import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.prepare_data import normalize, parse_patient, physio_split, point_mask, reject_pointer
from scripts.flow_matching.compare_results import compare
from scripts.flow_matching.audit_author_code import corrected_giflow, patch_text
from scripts.flow_matching.metrics import ensemble_totals, finalize, apply_time_cutoffs
from scripts.flow_matching.evaluate_cfmi_air36 import adapt_source
from scripts.flow_matching.run_cfmi import summarize_author_metric, latest_checkpoint
from scripts.flow_matching.prepare_transfer import group_split, missing_mask
from scripts.flow_matching.compare_transfer import paired_summary
from scripts.flow_matching.phase_gate import check_queue
from scripts.flow_matching.register_transfer import write_frozen


def test_hourly_parser_matches_last_measurement_protocol():
    payload = b'Time,Parameter,Value\n00:10,HR,70\n00:50,HR,80\n01:00,Na,140\n48:00,HR,99\n00:00,Age,63\n'
    values, observed = parse_patient(payload)
    assert values.shape == (48, 35)
    assert values[0, 1] == 80 and values[1, 2] == 140
    assert observed.sum() == 2
    assert np.isfinite(values).all()


def test_scaling_fits_training_patients_only():
    values = np.array([[[1.], [3.]], [[10000.], [20000.]]])
    mask = np.ones_like(values, dtype=bool)
    normalized, mean, std = normalize(values, mask, [0])
    assert mean[0] == 2 and std[0] == 1
    assert np.array_equal(normalized[0, :, 0], [-1, 1])


def test_masks_hide_observations_only_and_are_shared_deterministically():
    observed = np.array([[[True, False], [True, True], [True, False]]])
    a = point_mask(observed, .5, 1)
    assert np.array_equal(a, point_mask(observed, .5, 1))
    assert not (a & ~observed).any()
    assert (observed & ~a).sum() == 2
    assert observed.sum() == 4


def test_five_patient_folds_are_disjoint_and_cover_every_patient():
    tests = []
    for fold in range(5):
        split = physio_split(4000, fold=fold)
        assert [len(split[k]) for k in ['train', 'val', 'test']] == [2800, 400, 800]
        assert not set(split['train']) & set(split['test'])
        assert not set(split['val']) & set(split['test'])
        tests.extend(split['test'])
    assert len(set(tests)) == 4000
    assert not np.array_equal(physio_split(4000)['test'], physio_split(4000, generator='default_rng')['test'])


def test_git_lfs_pointer_cannot_be_used_as_array(tmp_path):
    path = tmp_path / 'data.npz'
    path.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1000\n')
    with pytest.raises(ValueError, match='Git LFS'):
        reject_pointer(path)


def test_source_patch_refuses_ambiguous_or_changed_upstream():
    with pytest.raises(ValueError):
        patch_text('changed', 'expected', 'replacement')
    with pytest.raises(ValueError):
        patch_text('expected expected', 'expected', 'replacement')


def reference():
    return {'id': 'r', 'metric': 'mae', 'mean': .2, 'standard_error': .001,
            'runs': 5, 'num_samples': 100, 'rounding_half_unit': .0005, 'expected_folds': [0, 1, 2, 3, 4]}


def record(fold, value=.2, **overrides):
    cfg = {'reference_id': 'r', 'num_samples': 100, 'fold': fold, **overrides}
    return {'status': 'completed', 'config': cfg, 'metrics': {'mae': value}}


def test_reproduction_comparison_rejects_smoke_and_missing_folds():
    assert compare([record(0, diagnostic=True)], reference())['verdict'] == 'not_comparable'
    assert compare([record(0, num_samples=2)], reference())['verdict'] == 'not_comparable'
    assert compare([record(0)], reference())['verdict'] == 'incomplete_repeats'
    assert compare([record(0) for _ in range(5)], reference())['verdict'] == 'not_comparable'


def test_comparison_aggregates_all_results_without_selecting_best_test_score():
    assert compare([record(i, .2) for i in range(5)], reference())['verdict'] == 'numerically_compatible'
    result = compare([record(i, .2 if i == 0 else .8) for i in range(5)], reference())
    assert result['mean'] == pytest.approx(.68)
    assert result['verdict'] == 'differs_from_paper'


def test_reviewed_giflow_snapshot_is_syntactically_valid_and_preserved():
    original = Path('experiments/runs/flow_matching_campaign/sources/giflow/original/main.py')
    if not original.exists():
        pytest.skip('Optional downloaded integration snapshot')
    text = original.read_text(encoding='utf-8')
    patched = corrected_giflow(text)
    ast.parse(patched)
    assert 'for batch in test_loader:\n            batch = batch.to(device)\n            x0, x, y, u, eval_mask, t' not in patched
    assert 'torch.load(early_stopping.path' in patched
    assert patched.count('ema.register()') == 1
    assert original.read_text(encoding='utf-8') == text


def test_common_metrics_use_exact_crps_median_mae_and_mean_rmse():
    # Asymmetric ensemble distinguishes the mean from the median point estimate.
    target = np.array([[0., np.nan]])
    samples = np.array([[[0., np.nan], [0., np.nan], [3., np.nan]]])
    metrics = finalize(ensemble_totals(target, samples, np.array([[True, False]])))
    assert metrics['mae'] == 0 and metrics['rmse'] == 1
    assert metrics['crps'] == pytest.approx(1 / 3)
    assert metrics['normalized_crps'] is None
    assert metrics['coverage_90'] == 1
    assert metrics['interval_width_90'] == pytest.approx(2.7)


def test_common_metrics_aggregate_batches_by_target_count():
    target = np.array([[0., 1.], [2., 3.]])
    samples = np.stack([target - 1, target + 2], axis=1)
    mask = np.array([[True, False], [True, True]])
    full = ensemble_totals(target, samples, mask)
    chunks = [ensemble_totals(target[i:i + 1], samples[i:i + 1], mask[i:i + 1]) for i in range(2)]
    combined = {key: sum(chunk[key] for chunk in chunks) for key in full}
    assert finalize(full) == finalize(combined)
    with pytest.raises(ValueError, match='Non-finite'):
        ensemble_totals(target, samples * np.nan, mask)


def test_sparse_author_crps_nan_is_reported_but_other_invalid_scores_fail():
    mean, audit = summarize_author_metric(np.array([.2, np.nan, .4]), 'crps')
    assert mean == pytest.approx(.3) and audit['undefined_components'] == 1
    for values, key in [(np.array([np.inf, .2]), 'crps'), (np.array([np.nan]), 'crps'), (np.array([.2, np.nan]), 'mae')]:
        with pytest.raises(ValueError):
            summarize_author_metric(values, key)


def test_resume_selects_latest_checkpoint_numerically(tmp_path):
    for version in [2, 9, 10]:
        path = tmp_path / f'train/seed_m1_d1/lightning_logs/version_{version}/checkpoints/last.ckpt'
        path.parent.mkdir(parents=True)
        path.touch()
    assert latest_checkpoint(tmp_path).parents[1].name == 'version_10'


def test_cfmi_reuses_real_author_checkpoint_in_clean_process():
    import subprocess
    python = Path('.venv/fm-author-py310/Scripts/python.exe')
    training = Path('experiments/runs/flow_matching_campaign/cfmi_training_fold0')
    checkpoint = latest_checkpoint(training)
    if not python.exists() or checkpoint is None:
        pytest.skip('Optional author environment and trained checkpoint')
    script = ('from pathlib import Path; from scripts.flow_matching.run_cfmi import checkpoint_complete; '
              f'checkpoint=Path({str(checkpoint)!r}); '
              'source=Path("experiments/runs/flow_matching_campaign/sources/cfmi/corrected"); '
              'assert checkpoint_complete(checkpoint,source,200); '
              'assert not checkpoint_complete(checkpoint,source,201)')
    subprocess.run([str(python), '-c', script], check=True, capture_output=True, text=True)


def test_transfer_split_keeps_every_simulation_group_together():
    groups = np.repeat(np.arange(50), 3)
    splits = group_split(groups, 2026)
    assert [len(splits[k]) for k in ['train', 'val', 'test']] == [105, 15, 30]
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(groups[splits[a]]) & set(groups[splits[b]])


def test_block_and_channel_masks_preserve_native_missingness():
    observed = np.ones((3, 48, 8), dtype=bool)
    observed[:, 0, 0] = False
    for pattern in ['block', 'channel']:
        mask = missing_mask(observed, .5, 2026, pattern)
        assert np.array_equal(mask, missing_mask(observed, .5, 2026, pattern))
        assert not (mask & ~observed).any()
    channel = missing_mask(observed, .5, 2026, 'channel')
    assert np.array_equal((~channel).all(axis=1).sum(axis=1), [4, 4, 4])


def test_paired_transfer_summary_requires_all_seeds_and_reports_reversals():
    seeds = [2026, 2027, 2028, 2029, 2030]
    baseline = dict.fromkeys(seeds, .2)
    assert paired_summary(baseline, {2026: .1}, seeds)['verdict'] == 'incomplete_pairs'
    assert paired_summary(baseline, dict.fromkeys(seeds, .1), seeds)['verdict'] == 'candidate_lower_error'
    assert paired_summary(baseline, dict.fromkeys(seeds, .3), seeds)['verdict'] == 'candidate_higher_error'
    assert paired_summary(baseline, baseline, seeds)['verdict'] == 'insufficient_evidence'


def test_phase_gate_rejects_diagnostic_or_changed_prerequisite(tmp_path):
    import hashlib
    import json
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps({'id': 'job', 'output_root': 'out'}))
    checksum = hashlib.sha256(cfg.read_bytes()).hexdigest()
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'jobs': [{'config': 'cfg.json', 'config_sha256': checksum}]}))
    assert check_queue(queue, tmp_path) == ['job']
    (tmp_path / 'out').mkdir()
    result = tmp_path / 'out/result.json'
    result.write_text(json.dumps({'status': 'completed', 'config_sha256': checksum, 'config': {'diagnostic': True}}))
    with pytest.raises(ValueError, match='Diagnostic'):
        check_queue(queue, tmp_path)
    result.write_text(json.dumps({'status': 'completed', 'config_sha256': checksum, 'config': {}}))
    assert check_queue(queue, tmp_path) == []
    cfg.write_text('{}')
    with pytest.raises(ValueError, match='configuration changed'):
        check_queue(queue, tmp_path)


def test_frozen_registration_preserves_existing_order_and_rejects_changes(tmp_path):
    path = tmp_path / 'configuration.json'
    path.write_text('{"b": 2, "a": 1}\n')
    original = path.read_bytes()
    write_frozen(path, {'a': 1, 'b': 2})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='preserved'):
        write_frozen(path, {'a': 1, 'b': 3})
    assert path.read_bytes() == original


def test_registered_jobs_are_frozen_and_all_papers_have_cited_status():
    import hashlib
    import json
    campaign = json.loads(Path('configs/reproducibility/flow_matching_campaign.v1.json').read_text(encoding='utf-8'))
    assert len(campaign['papers']) == 20
    for paper in campaign['papers']:
        model = json.loads(Path(f"configs/models/fm_{paper['id']}.json").read_text(encoding='utf-8'))
        assert model['citation'] and model['reproduction_status']
    for filename, count in [('fm_original_queue.v1.json', 30), ('fm_transfer_queue.v1.json', 270), ('fm_air36_original_queue.v1.json', 10)]:
        queue = json.loads(Path('configs/experiments', filename).read_text(encoding='utf-8'))
        assert len(queue['jobs']) == count
        for job in queue['jobs']:
            assert hashlib.sha256(Path(job['config']).read_bytes()).hexdigest() == job['config_sha256']


def test_air36_overlap_cutoff_removes_timestamps_not_sensor_channels():
    mask = np.ones((2, 5, 3), dtype=bool)
    result = apply_time_cutoffs(mask, [2, 0])
    assert not result[0, :2].any() and result[0, 2:].all()
    assert result[1].all() and mask.all()
    with pytest.raises(ValueError):
        apply_time_cutoffs(mask, [6, 0])


def test_air36_dual_evaluation_keeps_legacy_scores_and_is_syntactically_valid():
    path = Path('experiments/runs/flow_matching_campaign/sources/cfmi/corrected/eval_imputation_timeseries.py')
    if not path.exists():
        pytest.skip('Optional author snapshot')
    original = path.read_text(encoding='utf-8')
    adapted = adapt_source(original)
    ast.parse(adapted)
    assert 'imputable_points[i, ..., 0:cut_off[i].item()] = False' in adapted
    assert 'corrected_loss' in adapted and 'apply_time_cutoffs' in adapted


def test_air36_prepared_cutoffs_match_author_evaluation_targets():
    path = Path('data/public_datasets/flow_matching/processed_campaign_v1/air36_original/csdi_fold0.npz')
    if not path.exists():
        pytest.skip('Optional downloaded real data')
    with np.load(path, allow_pickle=False) as data:
        indices = data['test']
        mask = apply_time_cutoffs(data['eval_mask'][indices], data['cut_length'][indices])
        assert mask.shape == (82, 36, 36)
        assert np.array_equal(data['cut_length'][indices][data['cut_length'][indices] > 0], [12, 12])
        assert mask.sum() > 0
