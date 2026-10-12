import ast
from copy import deepcopy
import inspect
import subprocess
import sys

import pytest

from scripts.flow_matching import run_full_image_training_execution_recovery_v5 as worker
from scripts.flow_matching import run_full_image_training_recovery_queue_v5 as queue


def test_resume_worker_changes_only_frozen813_checkpoint_identity():
    tree=worker.resume_run_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant):
            if type(node.value) is int and node.value==813:node.value=401
            elif isinstance(node.value,str):node.value=node.value.replace('saved813_resume','saved401_resume').replace('813','401')
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(worker.previous.run)))


def test_entire_original_full25_controller_changes_only_worker_entry():
    tree=queue.main_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_full_image_training_execution_recovery_v5':
            node.value='scripts.flow_matching.run_full_image_training_execution_recovery_v3'
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(queue.original.main)))


def test_cold_parent_import_does_not_initialize_numerical_or_CUDA_backend():
    code='import scripts.flow_matching.run_full_image_training_recovery_queue_v5,sys;assert not {"numpy","scipy","torch"}&set(sys.modules)'
    result=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')


def fixture(monkeypatch):
    prior=dict(settings=dict(minimum_commit_headroom_bytes=28*1024**3,emergency_commit_headroom_bytes=16*1024**3,cpu_threads=2),
        resource_lock='original_GPU.lock',lane_lock='original_lane.lock',full_cohort_required_free_disk_bytes=454685515552,
        original_runtimes={'FM':{},'WT':{}},ordered_original_jobs=['first']+['job'+str(i) for i in range(24)])
    config=deepcopy(prior);config['settings']['minimum_commit_headroom_bytes']=40*1024**3
    config.update(previous_configuration='prior',saved813_resume=dict(job_configuration='first',completed_epochs=813,remaining_epochs=1187,
        CPU_proof=dict(path='proof',sha256='sha'),CPU_terminal=dict(path='terminal',sha256='terminalsha'),archive=dict(path='archive',sha256='archivesha')))
    proof=dict(passed=True,completed_original_epochs=813,verified_original_optimizer_updates=92682,complete_epoch_receipts_verified=813,
        full_native_parameters=45289539,required_original_epochs=2000,remaining_original_epochs=1187,CUDA_initialized=False,
        all_model_and_StableAdamW_variables_bitwise_restored=True,Python_NumPy_native_CPU_and_order_RNG_states_bitwise_restored=True,
        all_model_and_optimizer_tensors_finite=True,new_optimizer_updates_executed=0,checkpoint_receipt=config['saved813_resume']['archive'])
    terminal=dict(exit_code=0,passed=True,resource_guard_aborted=False,report_sha256='sha')
    def read(path):return {'prior':prior,'proof':proof,'terminal':terminal}[path.name]
    monkeypatch.setattr(queue.original,'read',read)
    monkeypatch.setattr(queue.original,'verify_receipts',lambda _:None)
    def canonical(value):
        assert value['settings']==prior['settings']
        assert value['ordered_original_jobs']==prior['ordered_original_jobs']
        return {p:'FM' for p in value['ordered_original_jobs']}
    monkeypatch.setattr(queue.original,'verify_registration',canonical)
    return config,proof,terminal


@pytest.mark.parametrize('mutation',['margin','emergency','threads','epochs','remaining','GPU_lock','CPU_failed','CUDA','weights','updates','guard'])
def test_weaker_guards_incomplete_state_short_budget_and_changed_original_lanes_rejected(monkeypatch,mutation):
    config,proof,terminal=fixture(monkeypatch);assert len(queue.verify_registration(config))==25
    if mutation=='margin':config['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='emergency':config['settings']['emergency_commit_headroom_bytes']-=1
    elif mutation=='threads':config['settings']['cpu_threads']=1
    elif mutation=='epochs':config['saved813_resume']['completed_epochs']=500
    elif mutation=='remaining':config['saved813_resume']['remaining_epochs']=1000
    elif mutation=='GPU_lock':config['resource_lock']='other'
    elif mutation=='CPU_failed':proof['passed']=False
    elif mutation=='CUDA':proof['CUDA_initialized']=True
    elif mutation=='weights':proof['full_native_parameters']=4528953
    elif mutation=='updates':proof['verified_original_optimizer_updates']=57000
    else:terminal['resource_guard_aborted']=True
    with pytest.raises(ValueError):queue.verify_registration(config)
