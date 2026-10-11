from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching import run_grasp_tau05_postreplay_pipeline_v1 as pipeline


def fixture(monkeypatch):
    root = Path('/fixture')
    config = dict(source_receipts=[dict(path='frozen', sha256='registered')],
        producer_configuration='producer', preserved_thirteen_model_union='union',
        settings=dict(minimum_commit_headroom_bytes=22), resource_lock='original_lock', python='original_python')
    producer = dict(settings=deepcopy(config['settings']), resource_lock='original_lock', python='original_python',
        cpu_threads=2, full_job_count=1, full_results=[dict(canonical_id=pipeline.CANONICAL)],
        output_directory='report', state_root='state')
    union = dict(full_original_models=13, all_complete_arrays_point_mapping_and_metrics_independently_verified=True,
        original_full_native_worker_exit_success_verified=True, no_new_training_seed_slots=True,
        records=[dict(canonical_id=pipeline.CANONICAL.replace('1104', '1103'),
            recipes=[dict(recipe=dict(source_samples=5, flow_evaluations=10), passed=True)])])
    monkeypatch.setattr(pipeline, 'ROOT', root)
    monkeypatch.setattr(pipeline, 'sha', lambda path: 'registered')
    monkeypatch.setattr(pipeline, 'read', lambda path: {'producer': producer, 'union': union}.get(path.name, {}))
    monkeypatch.setattr(pipeline, 'check_terminal', lambda *args: None)
    return config, producer, union


def test_original_full_producer_and_first_seed_union_required(monkeypatch):
    config, producer, _ = fixture(monkeypatch)
    assert pipeline.verify_registration(config) is producer
    monkeypatch.setattr(pipeline, 'sha', lambda path: 'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        pipeline.verify_registration(config)


@pytest.mark.parametrize('field,value', [
    ('settings', dict(minimum_commit_headroom_bytes=1)), ('resource_lock', 'different_lock'),
    ('python', 'different_environment'), ('cpu_threads', 1), ('full_job_count', 0),
    ('full_results', [dict(canonical_id='other_seed')])])
def test_no_changed_resource_environment_or_producer_seed(monkeypatch, field, value):
    config, producer, _ = fixture(monkeypatch)
    producer[field] = value
    with pytest.raises(ValueError, match='resource guards'):
        pipeline.verify_registration(config)


@pytest.mark.parametrize('field,value', [
    ('full_original_models', 12), ('all_complete_arrays_point_mapping_and_metrics_independently_verified', False),
    ('original_full_native_worker_exit_success_verified', False), ('no_new_training_seed_slots', False)])
def test_partial_or_failed_first_seed_union_refused(monkeypatch, field, value):
    config, _, union = fixture(monkeypatch)
    union[field] = value
    with pytest.raises(ValueError, match='preserved complete'):
        pipeline.verify_registration(config)


@pytest.mark.parametrize('mutation', ['seed', 'recipe', 'passed', 'missing', 'duplicate'])
def test_first_seed_identity_and_complete_recipe_cannot_be_replaced(monkeypatch, mutation):
    config, _, union = fixture(monkeypatch)
    if mutation == 'seed':
        union['records'][0]['canonical_id'] = pipeline.CANONICAL
    elif mutation == 'recipe':
        union['records'][0]['recipes'][0]['recipe']['source_samples'] = 1
    elif mutation == 'passed':
        union['records'][0]['recipes'][0]['passed'] = False
    elif mutation == 'missing':
        union['records'] = []
    else:
        union['records'].append(deepcopy(union['records'][0]))
    with pytest.raises(ValueError, match='Complete first original seed'):
        pipeline.verify_registration(config)


def test_terminal_native_replay_failure_is_propagated(monkeypatch):
    config, _, _ = fixture(monkeypatch)
    def reject(*args):
        raise ValueError('native partial replay')
    monkeypatch.setattr(pipeline, 'check_terminal', reject)
    with pytest.raises(ValueError, match='native partial replay'):
        pipeline.verify_registration(config)
