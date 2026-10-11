import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_spectral_original_queue as original
from scripts.flow_matching import run_spectral_resource_admission_v3 as previous
from scripts.flow_matching.run_spectral_resource_order_v4 import ordered_main_ast, scheduler_jobs


def queue():
    return dict(settings=dict(minimum_commit_headroom_bytes=20 * 1024**3,
        emergency_commit_headroom_bytes=12 * 1024**3),
        spectral_sampling_minimum_commit_headroom_bytes=48 * 1024**3,
        jobs=[dict(id=str(i), model='spectral_flow' if i % 2 == 0 else 'diffusion_ts',
            seed=i, training_updates=25000 if i % 2 == 0 else 10000) for i in range(70)])


def test_all70_original_scientific_slots_retained_without_mutation_or_lost_order():
    value = queue()
    before = copy.deepcopy(value)
    ordered = scheduler_jobs(value)
    assert value == before
    assert len(ordered) == 70 and len({j['id'] for j in ordered}) == 70
    assert sorted(ordered, key=lambda j: int(j['id'])) == before['jobs']
    assert ordered[:35] == before['jobs'][1::2]
    assert ordered[35:] == before['jobs'][0::2]
    assert all(j is value['jobs'][int(j['id'])] for j in ordered)


def test_entire_native_main_retained_after_only_dispatch_iterator_change():
    source = inspect.getsource(original.main)
    expected = previous.selected_main_ast(source)
    actual = ordered_main_ast(source)
    dispatch = next(n for n in ast.walk(actual) if isinstance(n, ast.For)
        and ast.dump(n.iter) == ast.dump(ast.parse('scheduler_jobs(queue)', mode='eval').body))
    dispatch.iter = ast.parse("queue['jobs']", mode='eval').body
    assert ast.dump(actual) == ast.dump(expected)


@pytest.mark.parametrize('fault', ['duplicate', 'unknown', 'weakened_spectral_guard'])
def test_invalid_scope_or_admission_is_rejected(fault):
    value = queue()
    if fault == 'duplicate': value['jobs'][1]['id'] = value['jobs'][0]['id']
    elif fault == 'unknown': value['jobs'][1]['model'] = 'reduced_surrogate'
    else: value['spectral_sampling_minimum_commit_headroom_bytes'] = 1
    with pytest.raises(ValueError):
        scheduler_jobs(value)
