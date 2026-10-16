import ast
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.flow_matching import continue_industrial_SB_math_diagnostic_lock_v4 as m
from scripts.flow_matching import continue_industrial_SB_transport_extension_v3 as original


def test_only_exact_math_settings_select_separate_lock_and_full_guard_stays_same():
    c=m.read(m.ROOT/original.CONFIG)
    assert m.selected_lock(c,c['settings'])==c['resource_lock']
    assert m.selected_lock(c,c['diagnostic_settings'])==m.DIAGNOSTIC_LOCK
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert c['diagnostic_settings']['minimum_commit_headroom_bytes']==22*2**30
    changed=deepcopy(c['diagnostic_settings']);changed['emergency_commit_headroom_bytes']=1
    with pytest.raises(ValueError):m.selected_lock(c,changed)


def test_original_controller_source_is_unchanged_and_only_one_path_expression_changes():
    before=(m.ROOT/(original.MODULE.replace('.','/')+'.py')).read_bytes()
    entry=m.original_controller_with_diagnostic_lock()
    assert (m.ROOT/(original.MODULE.replace('.','/')+'.py')).read_bytes()==before
    assert entry.__globals__['_selected_stage_lock'] is m.selected_lock
    # Undo precisely the one new lock-selection call, then compare the ENTIRE
    # compiled function AST. Code object identity/index offsets are irrelevant.
    tree=deepcopy(entry.scheduler_source_ast)
    target=ast.dump(ast.parse('_selected_stage_lock(c, settings)',mode='eval').body)
    changed=[]
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if ast.dump(node)==target:
                changed.append(node)
                return ast.copy_location(ast.parse("c['resource_lock']",mode='eval').body,node)
            return self.generic_visit(node)
    restored=Restore().visit(tree)
    full=ast.parse(before);old=next(x for x in full.body if isinstance(x,ast.FunctionDef) and x.name=='controller')
    assert len(changed)==1 and ast.dump(restored.body[0])==ast.dump(old)


def test_numerical_children_and_worker_invocations_remain_the_original_source_commands():
    import inspect
    tree=ast.parse(inspect.getsource(original.controller))
    calls=[ast.unparse(x) for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='run']
    assert len(calls)==2 and 'DIAGNOSTIC' in calls[0] and 'MODULE' in calls[1]
    before=m.sha(m.ROOT/(original.DIAGNOSTIC.replace('.','/')+'.py'))
    m.original_controller_with_diagnostic_lock()
    assert m.sha(m.ROOT/(original.DIAGNOSTIC.replace('.','/')+'.py'))==before


@pytest.mark.parametrize('fault',[None,'creation_time','command','child','active_pid','stage','state','shared_owned','no_own_lock','training_lock','attempt','report'])
def test_idle_handoff_refuses_every_live_numerical_or_identity_change(tmp_path,monkeypatch,fault):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(original,'STATE','state')
    monkeypatch.setattr(original,'REPORT',str(tmp_path/'real_report.json'))
    base=tmp_path/'state';base.mkdir()
    state=dict(pid=99,create_time=42.,active_pid=None,stage='coupling_diagnostic',state='waiting_for_heavy_lock',
        shared_lock_owned=False,full_training_complete=False)
    if fault=='active_pid':state['active_pid']=100
    elif fault=='stage':state['stage']='full_training'
    elif fault=='state':state['state']='running_coupling_diagnostic'
    elif fault=='shared_owned':state['shared_lock_owned']=True
    elif fault=='attempt':(base/'attempts').mkdir()
    elif fault=='report':(tmp_path/'real_report.json').write_text('{}')
    (base/'status.json').write_text(json.dumps(state))
    config=tmp_path/original.CONFIG;config.parent.mkdir(parents=True);config.write_text(json.dumps(dict(resource_lock='training.lock')))
    files=[] if fault=='no_own_lock' else [SimpleNamespace(path=str(base/'controller.lock'))]
    if fault=='training_lock':files.append(SimpleNamespace(path=str(tmp_path/'training.lock')))
    process=SimpleNamespace(pid=99,is_running=lambda:True,create_time=lambda:43. if fault=='creation_time' else 42.,
        cmdline=lambda:['other'] if fault=='command' else ['controller'],
        children=lambda recursive:([SimpleNamespace(name=lambda:'python.exe')] if fault=='child' else []),open_files=lambda:files)
    monkeypatch.setattr(m.psutil,'Process',lambda _:process)
    expected=dict(pid=99,create_time=42.,command=['controller'])
    if fault:
        with pytest.raises(ValueError):m.idle_snapshot(expected)
    else:
        snapshot,_=m.idle_snapshot(expected);assert snapshot['no_numerical_descendants'] and snapshot['owns_only_metadata_lane']
