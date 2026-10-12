import ast
from pathlib import Path
import subprocess
import sys
from types import CodeType

import pytest

from scripts.flow_matching.run_cfm_seed1105_stronger_commit_parent_v3 import parent_namespace


def test_cold_parent_loads_no_scientific_backend():
    code='from scripts.flow_matching.run_cfm_seed1105_stronger_commit_parent_v3 import parent_namespace;v=parent_namespace();import sys;assert not {"numpy","scipy","torch"}&set(sys.modules)'
    result=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')


def compare(a,b):
    assert a.co_code==b.co_code and a.co_names==b.co_names and a.co_varnames==b.co_varnames
    assert len(a.co_consts)==len(b.co_consts)
    for x,y in zip(a.co_consts,b.co_consts):
        if isinstance(x,CodeType):compare(x,y)
        else:assert x==y


def test_every_original_parent_guard_update_receipt_and_worker_entry_remains_identical():
    values=parent_namespace()
    for name,path in [('verify_registration','run_cfm_checkpointed_original_job_v1.py'),('main','run_cfm_checkpointed_full_retry_v1.py')]:
        tree=ast.parse(Path('scripts/flow_matching',path).read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),namespace)
        compare(values[name].__code__,namespace[name].__code__)


@pytest.mark.parametrize('mutation',['budget','seed','checkpoints','scope','policy'])
def test_registration_rejects_short_budget_changed_seed_or_inconsistent_resource_policy(mutation):
    values=parent_namespace();function=values['verify_registration']
    job=dict(id='retry',recovery_original_job_id='canonical',seed=1105,method='node',epochs=400,
        expected_optimizer_updates=400,test_points_per_trajectory=500,batch_size=258)
    canonical=dict(job,id='canonical');canonical.pop('recovery_original_job_id')
    queue=dict(jobs=[job]+[dict(id=str(i)) for i in range(284)],resource_policy={'minimum_commit_headroom_bytes':34359738368})
    config=dict(source_receipts=[],queue='x',job_id='retry',canonical_job=canonical,checkpoint_interval=10,
        resource_policy=dict(queue['resource_policy']),original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        resume_full_epoch_checkpoint=None,resume_full_epoch_checkpoint_sha256=None)
    old_read,old_verify=values['original'].read,values['original'].verify
    values['original'].read=lambda _:queue;values['original'].verify=lambda _:None
    try:
        function(config)
        if mutation=='budget':job['epochs']=399
        elif mutation=='seed':job['seed']=1106
        elif mutation=='checkpoints':config['checkpoint_interval']=100
        elif mutation=='scope':queue['jobs'].pop()
        else:config['resource_policy']['minimum_commit_headroom_bytes']=17179869184
        with pytest.raises(ValueError):function(config)
    finally:values['original'].read,values['original'].verify=old_read,old_verify
