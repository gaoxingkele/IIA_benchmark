import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.register_tsad_constant_fit_recomparison_v2 import derive_job
from scripts.flow_matching.run_light_controller import ROOT


def test_derived_control_retains_complete_original_scientific_contract():
    original = dict(id='method__smap__seed1103', dataset='smap', seed=1103, window=100,
        model_config='old.json', dataset_manifest_path='old_manifest.json',
        parameter_overrides=dict(seed=1103, device='cpu'), output_directory='old_output',
        frozen_files=[dict(path='old.json', sha256='a' * 64), dict(path='old_input.npz', sha256='b' * 64)])
    control = derive_job(original, 'new.json', 'new_manifest.json', 'new_output',
        [dict(path='new.json', sha256='c' * 64), dict(path='new_manifest.json', sha256='d' * 64)])
    for key in ('seed', 'window', 'dataset', 'parameter_overrides'):
        assert control[key] == original[key]
    assert control['id'] == original['id'] + '__constant_fit_v2'
    assert original['frozen_files'] == control['frozen_files'][:2]
    assert control['output_directory'].startswith('new_output/jobs/')
    assert original['output_directory'] == 'old_output'


def test_reject_silent_change_of_original_data_or_source_receipt():
    original = dict(id='method', frozen_files=[dict(path='raw.csv', sha256='a' * 64)])
    with pytest.raises(ValueError, match='silently overwritten'):
        derive_job(original, 'new.json', 'manifest.json', 'output', [dict(path='raw.csv', sha256='b' * 64)])


def test_actual_every_affected_original_job_and_model_ablation_budget_present():
    base = ROOT / 'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2'
    report = json.loads((base / 'full_recomparison_registration.json').read_text('utf-8'))
    scope = json.loads((ROOT / 'configs/reproducibility/fm_tsad_constant_fit_recomparison_scope.v2.json').read_text('utf-8'))
    expected = {identity for group in scope['original_queue_groups']
                for identity in group['original_full_job_ids'] if '__smap__' in identity}
    actual, original_contracts = [], {}
    for binding in report['queue_bindings']:
        original = json.loads((ROOT / binding['original_source_queue']).read_text('utf-8'))
        queue = json.loads((ROOT / binding['path']).read_text('utf-8'))
        assert queue['evaluation'] == original['evaluation']
        assert queue['python'] == original['python']
        old = {j['id']: j for j in original['jobs']}
        for job in queue['jobs']:
            before = old[job['preprocessing_control_original_id']]
            actual.append(before['id'])
            allowed = {'id','model_config','dataset_manifest_path','output_directory','frozen_files','preprocessing_control_original_id'}
            assert {k:v for k,v in job.items() if k not in allowed} == {k:v for k,v in before.items() if k not in allowed}
            current_config = json.loads((ROOT / job['model_config']).read_text('utf-8'))
            previous_config = json.loads((ROOT / before['model_config']).read_text('utf-8'))
            assert current_config['parameters'] == previous_config['parameters']
            assert current_config['entrypoint'] == previous_config['entrypoint']
            assert current_config.get('citation') or current_config.get('citations')
            assert job['dataset_manifest_path'] == report['corrected_manifest']
            assert job['output_directory'] != before['output_directory']
            frozen = {s['path']:s['sha256'] for s in job['frozen_files']}
            assert all(frozen[s['path']] == s['sha256'] for s in before['frozen_files'])
    assert len(actual) == len(set(actual)) == 291
    assert set(actual) == expected
    assert sum(b['jobs'] for b in report['queue_bindings'] if b['all_original_devices'] == ['cpu']) == 121
    assert sum(b['jobs'] for b in report['queue_bindings'] if b['all_original_devices'] == ['cuda:0']) == 170
