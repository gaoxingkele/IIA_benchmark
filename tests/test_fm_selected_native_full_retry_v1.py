import ast
import copy
from pathlib import Path

import pytest

from scripts.flow_matching.run_selected_native_full_retry_v1 import full_queue_retry, selected_main_ast


@pytest.mark.parametrize('path', ['run_timegan_native_queue_v1.py', 'run_crossad_case_gated_queue.py'])
def test_only_dispatch_selection_changes_in_the_entire_original_controller(path):
    source = Path('scripts/flow_matching', path).read_text(encoding='utf-8')
    expected = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    actual = selected_main_ast(source).body[0]
    loop = next(n for n in ast.walk(actual) if isinstance(n, ast.For) and isinstance(n.iter, ast.ListComp))
    loop.iter = ast.parse("queue['jobs']", mode='eval').body
    assert ast.dump(actual) == ast.dump(expected)


@pytest.mark.parametrize('method', ['timegan', 'crossad'])
@pytest.mark.parametrize('fault', ['budget', 'seed', 'model', 'guard', 'removed', 'other_output'])
def test_whole_registry_and_scientific_resource_fields_cannot_change(method, fault):
    original = dict(jobs=[dict(id=str(i), seed=i, iterations=50000, model='full', output_directory='old'+str(i))
                         for i in range(45 if method == 'timegan' else 139)],
        settings=dict(emergency_commit_headroom_bytes=16, output_root='old-state'),
        queue_path='old-queue', state_root='old-state')
    registration = dict(method=method, selected_job_id='0', selected_output_directory='new-output',
        queue='new-queue', state_root='new-state')
    queue = copy.deepcopy(original)
    queue['jobs'][0]['output_directory'] = 'new-output'
    if method == 'timegan': queue.update(queue_path='new-queue', state_root='new-state')
    else: queue['settings']['output_root'] = 'new-state'
    assert full_queue_retry(original, queue, registration)[0] == original['jobs'][0]
    if fault == 'budget': queue['jobs'][0]['iterations'] = 10
    elif fault == 'seed': queue['jobs'][0]['seed'] = 100
    elif fault == 'model': queue['jobs'][0]['model'] = 'small'
    elif fault == 'guard': queue['settings']['emergency_commit_headroom_bytes'] = 1
    elif fault == 'removed': queue['jobs'].pop()
    else: queue['jobs'][1]['output_directory'] = 'new-other'
    with pytest.raises(ValueError): full_queue_retry(original, queue, registration)


@pytest.mark.parametrize('source', ['def main():\n    pass\n',
    "def main():\n    for job in queue['jobs']: pass\n    for job in queue['jobs']: pass\n"])
def test_missing_or_duplicate_dispatch_is_rejected(source):
    with pytest.raises(ValueError): selected_main_ast(source)
