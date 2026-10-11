import copy
from types import SimpleNamespace

import pytest

from scripts.flow_matching.transition_giflow_commit_reserve_v2 import compare_full_scope, eligible


def queues():
    old = dict(id='old',state_root='old_state',resource_lock='original_gpu_mutex',controller_yield_seconds=35,
        settings=dict(minimum_commit_headroom_bytes=25*1024**3, emergency_commit_headroom_bytes=12*1024**3,
            minimum_gpu_free_bytes=20*1024**3,emergency_gpu_free_bytes=4*1024**3,cpu_threads=4),
        jobs=[dict(id=str(i),seed=i,model_config='full_native',output_directory='original/'+str(i),
            arguments=dict(batch_size=128,training_epoch=300)) for i in range(140)])
    new=copy.deepcopy(old)
    new.update(id='new',state_root='new_state',parent_admission_queue='old_queue',
        parent_admission_queue_sha256='hash',commit_reserve_policy=dict(shared_CPU_emergency_floor_bytes=16*1024**3,
            registered_full_worker_private_bytes=13*1024**3,concurrent_growth_margin_bytes=13*1024**3))
    new['settings'].update(minimum_commit_headroom_bytes=42*1024**3,emergency_commit_headroom_bytes=16*1024**3)
    return old,new


def test_only_stronger_host_commit_admission_changes_all140_full_jobs_exact():
    old,new=queues()
    compare_full_scope(old,new)
    changes=[('settings','cpu_threads',1),('settings','minimum_gpu_free_bytes',1),
        ('settings','emergency_gpu_free_bytes',1),('settings','emergency_commit_headroom_bytes',12*1024**3),
        ('settings','minimum_commit_headroom_bytes',41*1024**3)]
    for target,key,value in changes:
        bad=copy.deepcopy(new)
        bad[target][key]=value
        with pytest.raises(ValueError): compare_full_scope(old,bad)
    for key,value in [('seed',999),('model_config','small'),('output_directory','overwrite'),
        ('arguments',dict(batch_size=1,training_epoch=1))]:
        bad=copy.deepcopy(new)
        bad['jobs'][0][key]=value
        with pytest.raises(ValueError): compare_full_scope(old,bad)
    bad=copy.deepcopy(new)
    bad['jobs'].pop()
    with pytest.raises(ValueError): compare_full_scope(old,bad)
    bad=copy.deepcopy(new)
    bad['controller_yield_seconds']=0
    with pytest.raises(ValueError): compare_full_scope(old,bad)


@pytest.mark.parametrize('case',['ready','active_model','unreported_child','active_job',
    'partial','failed','reused_pid','wrong_command','wrong_queue','wrong_state_pid'])
def test_only_complete_exact_childless_controller_can_handoff(case):
    command=['python','-m','original_native','--queue','frozen.json']
    expected=dict(pid=21,create_time=123.0,command=command,queue_sha256='qhash')
    parent=SimpleNamespace(pid=21,
        create_time=lambda:124.0 if case=='reused_pid' else 123.0,
        cmdline=lambda:['wrong'] if case=='wrong_command' else command,
        children=lambda recursive=True:[object()] if case=='unreported_child' else [])
    state=dict(pid=22 if case=='wrong_state_pid' else 21,
        queue_sha256='other' if case=='wrong_queue' else 'qhash',
        active_pid=22 if case=='active_model' else None,
        active_job='new_model' if case=='active_job' else None,
        jobs={'full': 'failed' if case=='failed' else 'completed'})
    actual=eligible(parent,state,expected,'full',lambda job:case!='partial',
        pid_exists=lambda pid:True,helper_check=lambda child:False)
    assert actual == (case=='ready')
