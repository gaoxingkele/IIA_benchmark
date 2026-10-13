"""Exact full fMRI retry and unchanged numerical/GPU guard dispatch."""
import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_spectral_fmri43_full_retry_v1 as retry

CONFIG = 'configs/runtime/fm_spectral_fmri43_actual500_full_retry.v1.json'


def fixture():
    config = retry.read(retry.ROOT / CONFIG)
    previous = retry.read(retry.ROOT / config['original_resource_order_configuration'])
    original = retry.read(retry.ROOT / previous['queue'])
    queue = retry.read(retry.ROOT / config['queue'])
    return original, queue, config


def test_all70_jobs_and_original15000_updates_full_generation_retained():
    original, queue, config = fixture(); old, job = retry.exact_retry(original, queue, config)
    assert job['training_updates'] == 15000 and job['gradient_accumulate_every'] == 2
    assert job['generation_batch_size'] == 2001 and job['data_shape'] == [9977, 24, 50]
    assert len(queue['jobs']) == 70 and job['seed'] == old['seed'] == 43


@pytest.mark.parametrize('change', ['scope', 'budget', 'shape', 'chunk', 'other', 'overwrite'])
def test_no_shortened_or_changed_scientific_recipe(change):
    original, queue, config = fixture(); queue = copy.deepcopy(queue)
    job = next(j for j in queue['jobs'] if j['id'] == retry.IDENTITY)
    if change == 'scope': queue['jobs'].pop()
    if change == 'budget': job['training_updates'] = 14500
    if change == 'shape': job['data_shape'][0] = 9976
    if change == 'chunk': job['generation_field_chunk_size'] = 1
    if change == 'other': queue['jobs'][0]['seed'] += 1
    if change == 'overwrite': job['output_directory'] = next(j for j in original['jobs'] if j['id'] == retry.IDENTITY)['output_directory']
    with pytest.raises(ValueError, match='all70 full jobs'):
        retry.exact_retry(original, queue, config)


def test_only_stronger_admission_changes_original_settings():
    original, queue, config = fixture(); old, job = retry.exact_retry(original, queue, config)
    assert retry.admission_settings(queue, job, queue['settings']) == dict(original['settings'], minimum_commit_headroom_bytes=40 * 1024**3)
    assert queue['settings'] == original['settings']


def test_entire_original_dispatch_and_guard_body_unchanged():
    original = retry.ordering.ordered_main_ast(inspect.getsource(retry.ordering.previous.original.main))
    actual = retry.selected_ast(inspect.getsource(retry.ordering.previous.original.main))
    restored = copy.deepcopy(actual)
    loops = [n for n in ast.walk(restored) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'job']
    assert len(loops) == 1
    loops[0].iter = ast.parse('scheduler_jobs(queue)', mode='eval').body
    assert ast.dump(restored) == ast.dump(original)


def test_actual_terminal_partial_state_and_full_diagnostic_binding_verified():
    _, _, config = fixture(); queue, old, job = retry.verify(config, check_live=False)
    assert len(queue['jobs']) == 70 and config['credited_previous_optimizer_updates'] == 0
    assert old['training_updates'] == job['training_updates'] == 15000
