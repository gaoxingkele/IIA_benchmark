import ast
import copy
import json
from types import SimpleNamespace

import pytest

from scripts.flow_matching.register_preserved_incident_recovery_v4 import ROOT, scientific, validate_case
from scripts.flow_matching.run_preserved_incident_recovery_v4 import capped_guard_ast, bind
from scripts.flow_matching.run_readonly_tab_range_queue_v1 import readonly_main_ast, OLD_LOCK


def config():
    return json.loads((ROOT / 'configs/runtime/fm_preserved_incident_recovery.v4.json').read_text(encoding='utf-8'))


def queue():
    return json.loads((ROOT / config()['queue']).read_text(encoding='utf-8'))


def test_all_original_cases_and_only_two_additional_stopped_original_seeds_retained():
    current = queue()
    parent = json.loads((ROOT / current['parent_queue']).read_text(encoding='utf-8'))
    prior = {c['original_id']: c for c in parent['jobs']}
    actual = {c['original_id']: c for c in current['jobs']}
    assert len(prior) == 11 and len(actual) == 13
    assert set(actual) - set(prior) == {'pi_author__source_runtime__psm__full__seed1104',
                                      'pi_author__source_runtime__psm__full__seed1105'}
    assert sum('superseded_preserved_attempt' in c for c in current['jobs']) == 2
    for identity, case in prior.items():
        assert actual[identity]['original_job'] == case['original_job']
        assert scientific(actual[identity]['job']) == scientific(case['job'])
        assert actual[identity]['settings'] == case['settings']
        assert actual[identity]['evaluation'] == case['evaluation']
    for case in current['jobs']: validate_case(ROOT, case)


@pytest.mark.parametrize('change', ['seed', 'model', 'source', 'epochs', 'settings', 'evaluation'])
def test_scientific_changes_cannot_be_called_an_exact_recovery(change):
    case = copy.deepcopy(next(c for c in queue()['jobs'] if c['kind'] == 'strict'))
    if change == 'seed': case['job']['seed'] += 1
    elif change == 'model': case['job']['model_config'] += '.changed'
    elif change == 'source': case['job']['frozen_files'][0]['sha256'] = 'changed'
    elif change == 'epochs': case['job']['parameter_overrides']['epochs'] = 1
    elif change == 'settings': case['settings']['changed'] = True
    elif change == 'evaluation': case['evaluation']['cpu_threads'] += 1
    with pytest.raises(ValueError): validate_case(ROOT, case)


def test_original_incident_main_changes_exactly_one_resource_mutex_literal():
    source = (ROOT / 'scripts/flow_matching/run_resource_recovery.py').read_text(encoding='utf-8')
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changed = readonly_main_ast(source, config()['resource_lock'])
    replacements = [n for n in ast.walk(changed) if isinstance(n, ast.Constant) and n.value == config()['resource_lock']]
    assert len(replacements) == 1
    replacements[0].value = OLD_LOCK
    assert ast.dump(changed) == ast.dump(original)


def test_original_guard_retains_every_statement_and_only_strengthens_conditions():
    source = (ROOT / 'scripts/flow_matching/heavy_resources.py').read_text(encoding='utf-8')
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'guarded_wait')
    changed = capped_guard_ast(source)
    guard = next(n for n in ast.walk(changed) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp))
    assert len(guard.test.values) == 3
    guard.test = guard.test.values[0]
    assert ast.dump(changed) == ast.dump(original)


@pytest.mark.parametrize('rss,private,commit,aborted', [(7,9,40,False),(11,11,40,True),(9,13,40,True),(7,9,15,True)])
def test_actual_original_function_enforces_rss_private_and_commit_guard(monkeypatch,rss,private,commit,aborted):
    import psutil
    source = (ROOT / 'scripts/flow_matching/heavy_resources.py').read_text(encoding='utf-8')
    process = SimpleNamespace(pid=123,code=None)
    calls = []
    def poll():
        calls.append(True)
        if not aborted and len(calls) > 1: process.code = 0
        return process.code
    process.poll = poll
    process.wait = lambda: process.code
    handle = SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss,private=private,vms=private),children=lambda recursive:[])
    monkeypatch.setattr(psutil,'Process',lambda pid:handle)
    stopped = []
    def terminate(child): stopped.append(child.pid);child.code=1
    namespace = dict(psutil=psutil,resource_snapshot=lambda:dict(commit_available_bytes=commit),
                     terminate_tree=terminate,time=SimpleNamespace(sleep=lambda s:None))
    exec(compile(ast.Module(body=[capped_guard_ast(source)],type_ignores=[]),'actual-original-guard','exec'),namespace)
    code,usage=namespace['guarded_wait'](process,dict(emergency_commit_headroom_bytes=16,
        maximum_worker_rss_bytes=10,maximum_worker_private_bytes=12),lambda v:None)
    assert code == (1 if aborted else 0) and usage['resource_guard_aborted'] is aborted
    assert stopped == ([123] if aborted else [])


def test_all_queue_science_and_frozen_orchestration_bind():
    main,current,binding=bind(config())
    assert current['settings']['minimum_available_memory_bytes'] >= 24*1024**3
    assert current['settings']['minimum_commit_headroom_bytes'] >= 32*1024**3
    assert current['settings']['emergency_commit_headroom_bytes'] >= 16*1024**3
    assert len(current['jobs']) == 13 and callable(main)
    assert binding['source_path'] == 'scripts/flow_matching/run_resource_recovery.py'
