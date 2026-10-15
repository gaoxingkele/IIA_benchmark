"""Read-only checks after322 success;never recreate its one-time registration."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import pytest
from scripts.flow_matching import run_moment_icm_finite_control_cold_metadata_v4 as p
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as closure


def test_completed_configuration_remains_numerically_cold_and_guard_can_install_once():
    code = '''
import sys
from scripts.flow_matching import run_moment_icm_finite_control_cold_metadata_v4 as p
p.pre_guard(p.read(p.ROOT/p.CONFIG))
from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
a=install_facades();a['resource_snapshot']()
assert not any(x in sys.modules for x in ['numpy','pandas','torch'])
'''
    result = subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr[-4000:]


@pytest.mark.parametrize('mutation',['worker','guard','lock'])
def test_frozen_runtime_cannot_change_worker_guard_or_lock_after_completion(mutation):
    runtime = p.read(p.ROOT/p.CONFIG); p.pre_guard(runtime)
    if mutation == 'worker': runtime['worker_configuration'] = 'different.json'
    elif mutation == 'guard': runtime['settings']['minimum_commit_headroom_bytes'] -= 1
    else: runtime['resource_lock'] = 'different.lock'
    with pytest.raises(ValueError): p.pre_guard(runtime)


def test_one_time_registration_refuses_to_replace_actual_finished_scientific_output():
    config = p.read(p.ROOT/p.CONFIG)
    report = p.ROOT/config['artifact_root']/p.worker.REPORT
    before = p.sha(report)
    with pytest.raises(ValueError,match='pre-science'):
        p.registration()
    assert p.sha(report) == before


def test_actual_full_preparation_and_independent_guards_are_complete():
    config = closure.read(closure.ROOT/closure.CONFIG)
    assert closure.preparation_ready(config)
    report_path = closure.Path(config['artifact_root'])/closure.REPORT
    report = closure.read(report_path)
    guard = closure.read(closure.ROOT/config['state_root']/'guard_receipt.json')
    assert report['passed'] and report['sources']==322 and report['cells']==966 and report['arrays']==6762
    assert report['all_model_inputs_finite'] and report['original1627_caches_and322failures_untouched']
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
    assert guard['proof_sha256']==closure.sha(report_path)
