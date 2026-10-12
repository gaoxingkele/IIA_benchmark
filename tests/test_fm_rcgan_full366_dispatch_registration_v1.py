from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching import run_rcgan_full366_native_audit_dispatch_v1 as dispatch
from scripts.flow_matching import rcgan_segmented_checkpoint_evidence_v1 as evidence
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read


DISPATCH = 'configs/runtime/fm_rcgan_seed1104_full366_native_audit_dispatch.v1.json'


def test_actual_registered_checkpoint_and_trace_view_has_full_provenance():
    config = read(ROOT / DISPATCH)
    chain = read(ROOT / config['stages'][0]['configuration'])
    manifest = read(ROOT / chain['segmented_evidence_manifest'])
    view = ROOT / manifest['evidence_view']
    directories = sorted((view / 'native_checkpoints').iterdir())
    assert [d.name for d in directories] == ['epoch_%03d' % e for e in range(366)]
    assert all(len(list(d.iterdir())) == 6 for d in directories)
    trace = (view / 'experiments/traces' / (manifest['canonical_job']['id'] + '.trace.txt')).read_text().splitlines()
    assert [int(line.split()[0]) for line in trace[1:]] == list(evidence.TRACE_EPOCHS)
    assert trace[1:] == [line for part in manifest['trace_provenance']
                        for line in Path(part['source']).read_text().splitlines()[1:]]
    assert all(part['time_semantics'] == 'actual_source_segment_elapsed_seconds' for part in manifest['trace_provenance'])
    assert manifest['time_is_not_reconstructed_as_one_continuous_clock']
    assert manifest['view_is_recovery_evidence_not_completed500epoch_result']


def test_entire_actual_dispatch_preserves_original_family_settings_and_scope():
    config = read(ROOT / DISPATCH)
    chain = read(ROOT / config['stages'][0]['configuration'])
    state = read(ROOT / config['stages'][1]['configuration'])
    resume = read(ROOT / config['resume_configuration'])
    queue = read(ROOT / chain['queue'])
    original_chain = read(ROOT / 'configs/reproducibility/fm_rcgan_seed1104_full84_random_order_chain.2026-10-11_v2.json')
    assert chain['settings'] == state['settings'] == original_chain['settings']
    assert chain['resource_lock'] == state['resource_lock'] == original_chain['resource_lock']
    assert resume['settings'] == queue['settings']
    assert resume['resource_lock'] == queue['resource_lock']
    assert len(queue['jobs']) == 5
    assert resume['full_original_epochs'] == 500
    assert resume['verified_completed_epochs'] == 366
    assert resume['remaining_full_epochs'] == 134
    assert resume['canonical_job']['seed'] == resume['recovery_job']['seed'] == 1104
    assert resume['recovery_job']['id'] == resume['canonical_job']['id'] + '__state_recovery_v3'
    assert config['original_scope'] == dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


@pytest.fixture
def registration_fixture(monkeypatch):
    config = read(ROOT / DISPATCH)
    chain = read(ROOT / config['stages'][0]['configuration'])
    state = read(ROOT / config['stages'][1]['configuration'])
    resume = read(ROOT / config['resume_configuration'])
    queue = read(ROOT / chain['queue'])
    values = {config['stages'][0]['configuration']: chain, config['stages'][1]['configuration']: state,
              config['resume_configuration']: resume}
    monkeypatch.setattr(dispatch, 'verify_pins', lambda items: None)
    monkeypatch.setattr(dispatch, 'read', lambda p: values[Path(p).relative_to(ROOT).as_posix()])
    monkeypatch.setattr(evidence, 'verify_registration', lambda c: (queue, resume['canonical_job'], None, None))
    return deepcopy(config), values


@pytest.mark.parametrize('corruption', ['skip_gate', 'wrong_order', 'short_resume', 'extra_seed', 'lower_guard', 'scope_cut'])
def test_dispatch_rejects_narrowed_or_unverified_native_work(registration_fixture, corruption):
    config, values = registration_fixture
    resume = values[config['resume_configuration']]
    if corruption == 'skip_gate': config['stages'].pop()
    if corruption == 'wrong_order': config['stages'].reverse()
    if corruption == 'short_resume': resume['remaining_full_epochs'] = 2
    if corruption == 'extra_seed': resume['recovery_job']['seed'] = 888
    if corruption == 'lower_guard': resume['settings']['emergency_commit_headroom_bytes'] = 1
    if corruption == 'scope_cut': config['original_scope']['papers'] = 1
    with pytest.raises(ValueError): dispatch.verify_dispatch_registration(config)
