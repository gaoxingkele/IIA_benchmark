from copy import deepcopy
from scripts.flow_matching.run_cfm_ready_review_windows_v3 import ready_source, verify, ROOT, read, fingerprint
import json
import subprocess
import sys


def test_actual_idle_imports_load_no_scientific_stack_and_are_bounded():
    command = [sys.executable, '-X', 'utf8', '-c',
        'import sys,json,psutil;from scripts.flow_matching import run_cfm_ready_review_windows_v3 as c;'
        'print(json.dumps({"scientific_imports":[n for n in ["numpy","scipy","torch"] if n in sys.modules],'
        '"private_bytes":psutil.Process().memory_info().private,"fingerprint":c.fingerprint({"seed":1103})}))']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['scientific_imports'] == []
    assert report['private_bytes'] < 64 * 1024**2
    assert len(report['fingerprint']) == 64


def test_only_full_original_source_budget_and_grid_can_enter_review_window():
    job={'id':'exact','epochs':400,'expected_optimizer_updates':6400,'test_trajectories':20,'test_points_per_trajectory':500,'observed_dimensions':1}
    result=dict(status='completed',job_sha256=fingerprint(job),epochs_completed=400,optimizer_updates=6400,
                test_prediction_shape=[20,500,1],diagnostic_smoke=False,author_equivalence_certified=False)
    receipt=dict(exit_code=0,complete=True,resource_observations={'resource_guard_aborted':False})
    assert ready_source(job,result,receipt)
    for key,value in [('epochs_completed',399),('optimizer_updates',6399),('test_prediction_shape',[19,500,1]),('diagnostic_smoke',True)]:
        changed=deepcopy(result);changed[key]=value;assert not ready_source(job,changed,receipt)
    assert not ready_source(job,result,dict(receipt,exit_code=1))
    assert not ready_source(job,result,dict(receipt,resource_observations={'resource_guard_aborted':True}))


def test_every_canonical_slot_and_all_existing_guards_are_preserved():
    config=read(ROOT/'configs/runtime/fm_cfm_ready_complete_review_windows.v3.json');runtimes=verify(config)
    assert len(runtimes)==2 and all(len(r['routes'])==285 for r in runtimes)
    assert all(r['settings']==config['review_settings'] for r in runtimes)
