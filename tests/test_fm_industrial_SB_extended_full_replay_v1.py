"""Small fixtures validate rejection/restore plumbing, never model performance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import audit_industrial_SB_extended_full_replay_v1 as audit
from scripts.flow_matching.timegan_native_common_v1 import read, sha, write


def test_independent_windows_and_alignment_include_overlapping_tail():
    from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores
    values = np.arange(46, dtype=np.float32).reshape(23, 2)
    a, starts = audit.native_windows(values, 10)
    b, original_starts = covering_windows(values, 10)
    assert starts == [0, 10, 13] == list(original_starts)
    assert np.array_equal(a, b)
    score = np.arange(30, dtype=np.float32).reshape(3, 10)/3
    audit.require_array_equal(audit.native_align(score, starts, 23, 10), align_scores(score, original_starts, 23, 10), 'tail')


@pytest.mark.parametrize('fault', ['tail', 'dtype', 'one_bit', 'signed_zero'])
def test_whole_payload_rejects_truncation_dtype_and_bitwise_changes(fault):
    a = np.array([0., 1., 2.], dtype=np.float64); b = a.copy()
    if fault == 'tail': b = b[:-1]
    elif fault == 'dtype': b = b.astype(np.float32)
    elif fault == 'one_bit': b[1] = np.nextafter(b[1], 2)
    else: b[0] = -0.
    with pytest.raises(ValueError, match='Exact full array'): audit.require_array_equal(a, b, 'score')


def layout():
    return dict(input_sha256='a'*64,
        training_groups=[dict(array_prefix='t', group_id='day1')],
        validation_groups=[dict(array_prefix='v', group_id='day2')],
        entities=[dict(array_prefix='a', group_id='day3', entity_id='first'),
                  dict(array_prefix='b', group_id='day4', entity_id='second')])


@pytest.mark.parametrize('fault', ['missing_role', 'duplicate_prefix', 'role_contamination'])
def test_full_role_layout_rejects_missing_or_leaking_groups(fault):
    dataset = layout()
    if fault == 'missing_role': dataset['validation_groups'] = []
    elif fault == 'duplicate_prefix': dataset['entities'][0]['array_prefix'] = 'v'
    else: dataset['entities'][0]['group_id'] = 'day1'
    with pytest.raises(ValueError): audit.roles(dataset)


def fixture_metrics():
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    dataset = layout()
    payload = dict(v_validation=np.array([0., 1., 2., 3., 4., 5.]),
        a_test=np.array([0., 3., 3., 8.]), a_labels=np.array([0, 1, 1, 1]),
        b_test=np.array([1., 2., 6., 7.]), b_labels=np.array([1, 1, 0, 1]))
    entities = {g['entity_id']:(payload[g['array_prefix']+'_labels'], payload[g['array_prefix']+'_test']) for g in dataset['entities']}
    settings = dict(strict_false_alarm_percent=[.1, 1, 5], primary_false_alarm_percent=1,
        bootstrap_resamples=2, bootstrap_block_points=2, tab_threshold_ratios_percent=[1], tab_commit='fixture')
    metrics = evaluate_saved(entities, payload['v_validation'], np.array([0., 1.]), 'a'*64, 'b'*64, settings, 1103)['strict']
    return dataset, payload, metrics


def test_independent_tie_ranks_micro_macro_and_per_segment_events_match_source():
    dataset, payload, metrics = fixture_metrics()
    actual = audit.strict_metrics(dataset, payload, metrics, dict(rtol=2e-6, atol=2e-6), 'b'*64)
    assert set(actual) == {'0.1', '1', '5'}
    # Two adjacent anomaly tails belong to separate source entities.
    assert actual['1']['event_metrics']['events'] == 3
    assert actual['1']['micro']['true_positives'] == 2
    assert actual['1']['micro']['false_positives'] == 1


@pytest.mark.parametrize('fault', ['test_calibration', 'PA', 'labels', 'threshold', 'checkpoint', 'data', 'count', 'missing_budget'])
def test_independent_metrics_reject_protocol_and_metric_corruption(fault):
    dataset, payload, metrics = fixture_metrics()
    if fault == 'test_calibration': metrics['1']['calibration']['calibration_split'] = 'test'
    elif fault == 'PA': metrics['1']['point_adjustment'] = True
    elif fault == 'labels': payload['a_labels'][0] = 2
    elif fault == 'threshold': metrics['1']['calibration']['threshold'] += .1
    elif fault == 'checkpoint': metrics['1']['calibration']['model_artifact_sha256'] = 'other'
    elif fault == 'data': metrics['1']['calibration']['calibration_dataset_sha256'] = 'other'
    elif fault == 'count': metrics['1']['micro']['true_positives'] += 1
    else: del metrics['5']
    with pytest.raises(ValueError): audit.strict_metrics(dataset, payload, metrics, dict(rtol=2e-6, atol=2e-6), 'b'*64)


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    base = tmp_path/audit.SOURCE
    job = dict(id='registered', dataset='pronto_day4')
    paths = {k:base/'full_job'/name for k,name in [('checkpoint','model.pt'),('scores','scores.npz'),('result','result.json'),('solver','complete_transport_solver_audit.json')]}
    paths['checkpoint'].parent.mkdir(parents=True)
    paths['checkpoint'].write_bytes(b'checkpoint'); paths['scores'].write_bytes(b'scores')
    result = dict(id='registered', job=job, status='completed', checkpoint_sha256=sha(paths['checkpoint']), scores_sha256=sha(paths['scores']),
        training_losses=[1.]*20, epoch_audit=dict(declared_epochs=20, recorded_epochs=20), unique_fit_windows=8, test_points=11700, validation_points=4320)
    write(paths['result'], result)
    solver = dict(passed=True, source_result_sha256=sha(paths['result']), optimizer_updates=20,
        all20_actual_training_couplings=[dict(regularization=2., tolerance=1e-6, actual_absolute_marginal_error=9e-7, uniform_or_exact_OT_fallback=False) for _ in range(20)])
    write(paths['solver'], solver)
    guard = base/'attempts/full_training/receipt.json'
    write(guard, dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=sha(paths['solver'])))
    return dict(job=job), paths, guard


def test_exact_full_source_gate_requires_successful_guard_and_all_twenty_epochs(tmp_path, monkeypatch):
    original, paths, guard = source_fixture(tmp_path, monkeypatch)
    assert audit.require_source_completion(original)[1]['unique_fit_windows'] == 8
    guard.unlink()
    with pytest.raises(FileNotFoundError): audit.require_source_completion(original)


@pytest.mark.parametrize('fault', ['guard_abort','exit','proof_hash','losses','couplings','marginal','score_hash'])
def test_failed_incomplete_or_unbound_source_never_enters_inference(tmp_path, monkeypatch, fault):
    original, paths, guard = source_fixture(tmp_path, monkeypatch)
    if fault in ('guard_abort','exit','proof_hash'):
        g = read(guard)
        g[{'guard_abort':'resource_guard_aborted','exit':'exit_code','proof_hash':'proof_sha256'}[fault]] = {'guard_abort':True,'exit':1,'proof_hash':'wrong'}[fault]
        write(guard, g)
    elif fault in ('losses','score_hash'):
        r = read(paths['result'])
        if fault == 'losses': r['training_losses'].pop()
        else: r['scores_sha256'] = 'wrong'
        write(paths['result'], r)
        s = read(paths['solver']); s['source_result_sha256'] = sha(paths['result']); write(paths['solver'], s)
        g = read(guard); g['proof_sha256'] = sha(paths['solver']); write(guard, g)
    else:
        s = read(paths['solver'])
        if fault == 'couplings': s['all20_actual_training_couplings'].pop()
        else: s['all20_actual_training_couplings'][0]['actual_absolute_marginal_error'] = 1.01e-6
        write(paths['solver'], s); g = read(guard); g['proof_sha256'] = sha(paths['solver']); write(guard, g)
    with pytest.raises(ValueError): audit.require_source_completion(original)


def test_checkpoint_restore_never_calls_fit_and_preserves_every_score(monkeypatch):
    import torch
    from iia_benchmark.models.fm_objective_stable_extended import ExtendedStableCFMObjectiveDetector
    # This short trained fixture tests restoration only; it is not a paper case.
    config = dict(entrypoint='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector',
        parameters=dict(objective='sb', sigma=1., window=3, hidden_size=4, epochs=1, batch_size=2, monte_carlo_samples=1, score_times=[.5]))
    overrides = dict(seed=1103, device='cpu')
    model = ExtendedStableCFMObjectiveDetector(**config['parameters'], **overrides)
    values = np.random.default_rng(3).normal(size=(2,3,2)).astype(np.float32)
    model.fit(values)
    checkpoint = dict(configuration=config, overrides=overrides, modules={'_model':model._model.state_dict()}, window=3)
    def forbidden(*args, **kwargs): raise AssertionError('Replay must never fit')
    monkeypatch.setattr(ExtendedStableCFMObjectiveDetector, 'fit', forbidden)
    restored = audit.restore(checkpoint, config, overrides, (3,2))
    audit.require_array_equal(model.score(values), restored.score(values), 'restored fixture')
    assert not restored.training_losses_ and not restored.transport_solver_audit_


def test_cold_registration_imports_no_numerical_libraries_and_retains_original_gates():
    code = ('import sys;from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import registration,verify;'
        'c=registration();verify(c);assert c["settings"]["minimum_commit_headroom_bytes"]==32*2**30;'
        'assert c["settings"]["emergency_commit_headroom_bytes"]==16*2**30;'
        'assert not any(n in sys.modules for n in ["torch","numpy","pandas","scipy"]);'
        'assert c["new_optimizer_updates"]==c["new_training_seed_slots"]==0')
    p = subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode == 0, p.stderr
