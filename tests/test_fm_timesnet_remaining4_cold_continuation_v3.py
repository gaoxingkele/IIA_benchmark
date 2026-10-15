from copy import deepcopy
import ast
import inspect
import json
import subprocess
import sys
import pytest
from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as s


def test_all_original_stages_seeds_resources_and_real1104_proofs_retained():
    c=s.registration();old=s.verify(c)
    assert c['required_original_stages']==8 and c['required_original_seeds']==[1103,1104,1105,1106,1107]
    assert c['jobs']==old['jobs'] and c['modes']==old['modes']==['point','native']
    for key in ['settings','resource_lock','cpu_threads','original_scope']:assert c[key]==old[key]
    assert c['initial_completed']==s.COMPLETED and len(c['reused_actual_stages'])==2
    assert all(r['id']==s.original.identity(1104) for r in c['reused_actual_stages'])
    assert not c['all_original_experiments_complete']


@pytest.mark.parametrize('fault',['stages','seed','guard','lock','thread','certificate','source'])
def test_reduced_scope_changed_science_or_weakened_resources_rejected(fault):
    c=s.registration()
    if fault=='stages':c['required_original_stages']=6
    elif fault=='seed':c['required_original_seeds'].pop()
    elif fault=='guard':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='lock':c['resource_lock']='other.lock'
    elif fault=='thread':c['cpu_threads']=1
    elif fault=='certificate':c['reused_actual_stages'].pop()
    else:c['original_configuration_sha256']='0'*64
    with pytest.raises(ValueError):s.verify(c)


def state():
    return dict(pid=s.EXPECTED['pid'],create_time=s.EXPECTED['create_time'],active_pid=None,
        state='waiting_actual_original_training',completed=deepcopy(s.COMPLETED),failed_preserved={})


@pytest.mark.parametrize('fault',['active','child','lock','completed','failed','time','phase'])
def test_handoff_rejects_any_native_work_changed_identity_or_progress(fault):
    actual=state();children=[]
    if fault=='active':actual['active_pid']=123
    elif fault=='child':children=[dict(name='python.exe',exe='C:/Python/python.exe')]
    elif fault=='lock':actual['shared_lock_owned']=True
    elif fault=='completed':actual['completed'].append(dict(id='new_result',mode='point'))
    elif fault=='failed':actual['failed_preserved']={'a':'failed'}
    elif fault=='time':actual['create_time']+=1
    else:actual['state']='running_full_native'
    assert not s.idle_allowed(actual,s.EXPECTED,children)


def test_only_verified_idle_parent_with_allowed_console_is_eligible():
    children=[dict(name='conhost.exe',exe='C:/Windows/System32/conhost.exe')]
    assert s.idle_allowed(state(),s.EXPECTED,children)


@pytest.mark.parametrize('fault',['guard','proof','mode','native','source'])
def test_changed_existing_guard_or_certificate_is_never_reused(monkeypatch,fault):
    old=s.read(s.ROOT/s.original.CONFIG);job=next(j for j in old['jobs'] if j['seed']==1104)
    stage=s.original.stage_config(old,job,'native');proof=__import__('pathlib').Path(stage['output_directory'])/'independent_full_saved_score_verification.json'
    guard=s.ROOT/stage['state_root']/'receipt.json';read=s.read
    def changed(path):
        value=deepcopy(read(path))
        if path==guard and fault=='guard':value['passed']=False
        if path==proof:
            if fault=='proof':value['passed']=False
            elif fault=='mode':value['mode']='point'
            elif fault=='native':value['all_replayed_timestamp_scores_bitwise_equal']=False
            elif fault=='source':value['original_checkpoint_sha256']='0'*64
        return value
    monkeypatch.setattr(s,'read',changed)
    with pytest.raises(ValueError):s.checked_stage(old,job,'native')


def test_numerical_worker_entry_and_exact_all_original_arguments_remain():
    old=s.read(s.ROOT/s.original.CONFIG);job=next(j for j in old['jobs'] if j['seed']==1104)
    for mode in old['modes']:
        stage=s.original.stage_config(old,job,mode);path=s.ROOT/stage['state_root']/'frozen_stage.json'
        guard=s.read(path.parent/'receipt.json')
        assert s.worker_command(stage,path)==guard['exact_child_handle']['command']
        assert s.original.numerical_worker().__code__ is s.original.original.worker.__code__
    # Whole binding (including model libraries) is performed by that unchanged
    # numerical worker, rather than being retained in the waiting controller.
    tree=ast.parse(inspect.getsource(s.controller))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='binding' for n in ast.walk(tree))


def test_actual_retained_certificates_can_be_fully_rehashed_in_cold_process():
    code='from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as s;import sys,psutil,json;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules;print(json.dumps(dict(private_bytes=psutil.Process().memory_info().private,reused=len(c["reused_actual_stages"]))))'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout+p.stderr
    proof=json.loads(p.stdout);assert proof['reused']==2 and proof['private_bytes']<128*2**20
