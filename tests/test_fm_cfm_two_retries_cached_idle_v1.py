import json
import subprocess
import sys
from types import FunctionType
import pytest
from scripts.flow_matching import run_cfm_two_retries_cached_idle_v1 as s


def test_both_existing_full285_queues_all_science_budgets_and_guards_unchanged():
    c=s.registration()
    assert set(c['roles'])=={'node400_1104','node600_1107'}
    for name in s.ENTRIES:
        role=s.verify(c,name);runtime=s.read(s.ROOT/role['original_runtime']);q=s.read(s.ROOT/role['original_queue'])
        assert len(q['jobs'])==len({j['id'] for j in q['jobs']})==285
        assert role['selected_job'] in q['jobs']
        assert role['resource_policy']==q['resource_policy']
        assert role['resource_policy']['cpu_threads']==1
        assert role['resource_policy']['minimum_available_memory_bytes']==12*2**30
        assert role['resource_policy']['emergency_commit_headroom_bytes']==16*2**30
        assert role['stronger_admission_commit_bytes']==40*2**30
        assert role['family_lock']==runtime['family_lock']=='experiments/runs/fm_cfm_ts_original_cpu.lock'
        epochs=400 if name=='node400_1104' else 600
        assert role['selected_job']['epochs']==role['selected_job']['expected_optimizer_updates']==epochs
        assert role['selected_job']['test_points_per_trajectory']==500
        assert role['original_arguments']==['--runtime-config',role['original_runtime'],'--controller','recovery']
        index=role['previous_owner']['command'].index('-m')
        assert role['previous_owner']['command'][index+1:]==[s.ENTRY,*role['original_arguments']]


@pytest.mark.parametrize('fault',['scope','role','budget','memory','lock','args','hash'])
def test_scope_or_scientific_or_resource_or_identity_shortcut_rejected(fault):
    c=s.registration();r=c['roles']['node400_1104']
    if fault=='scope':c['original_scope']['papers']=1
    elif fault=='role':c['roles'].pop('node600_1107')
    elif fault=='budget':r['selected_job']['epochs']=1
    elif fault=='memory':r['resource_policy']['emergency_commit_headroom_bytes']=1
    elif fault=='lock':r['family_lock']='other.lock'
    elif fault=='args':r['original_arguments'].pop()
    else:r['queue_sha256']='changed'
    with pytest.raises(ValueError):s.verify(c,'node400_1104')


def state(role):
    return dict(pid=role['previous_owner']['pid'],queue_sha256=role['queue_sha256'],runtime_sha256=role['original_runtime_sha256'],
        jobs={},active_pid=None,active_job=None,shared_lock_owned=False,state='waiting_for_memory')


@pytest.mark.parametrize('fault',['active','child','shared','phase','jobs','hash','queue','time','output','job'])
def test_only_unstarted_exact_legacy_owner_without_any_native_child_is_eligible(monkeypatch,fault):
    role=s.registration()['roles']['node400_1104'];actual=state(role);desc=[]
    monkeypatch.setattr(s,'pending_unstarted',lambda role:True)
    if fault=='active':actual['active_pid']=9
    elif fault=='child':desc=[dict(name='python.exe',exe='C:/Python/python.exe')]
    elif fault=='shared':actual['shared_lock_owned']=True
    elif fault=='phase':actual['state']='running'
    elif fault=='jobs':actual['jobs']={'a':'completed'}
    elif fault=='hash':actual['runtime_sha256']='changed'
    elif fault=='queue':actual['queue_sha256']='changed'
    elif fault=='time':actual['create_time']=role['previous_owner']['create_time']+1
    elif fault=='job':actual['active_job']='a'
    else:monkeypatch.setattr(s,'pending_unstarted',lambda role:False)
    assert not s.safe_phase(role,actual,role['previous_owner'],desc)


def test_missing_legacy_timestamp_is_not_invented_os_identity_remains_required(monkeypatch):
    role=s.registration()['roles']['node400_1104'];actual=state(role)
    assert 'create_time' not in actual
    monkeypatch.setattr(s,'pending_unstarted',lambda role:True)
    assert s.safe_phase(role,actual,role['previous_owner'],[dict(name='conhost.exe',exe='C:/Windows/System32/conhost.exe')])
    assert FunctionType(s.original.main.__code__,s.namespace()).__code__ is s.original.main.__code__


@pytest.mark.parametrize('role',list(s.ENTRIES))
def test_actual_original_resource_api_is_cached_cold_and_keeps_all285_bindings(role):
    code='from scripts.flow_matching import run_cfm_two_retries_cached_idle_v1 as s;import json,sys;print(json.dumps(s.actual_guard_api_probe(s.registration(),sys.argv[1])))'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,role],cwd=s.ROOT,capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    proof=json.loads(p.stdout)
    assert proof['original_jobs']==285 and len(proof['selected'])==1
    assert proof['actual_guard_snapshot']=='cached_snapshot' and proof['stronger_commit']==40*2**30
    assert not proof['numpy_imported'] and not proof['torch_imported'] and not proof['numerical_execution']
    assert proof['private_bytes']<128*2**20
