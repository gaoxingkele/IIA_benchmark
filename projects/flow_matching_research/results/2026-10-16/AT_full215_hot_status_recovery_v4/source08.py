import ast
import inspect
import json
import os
import subprocess
import sys
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v4 as s


def test_only_duplicate_facade_initialization_changes_in_the_full_controller():
    old=ast.parse(inspect.getsource(s.previous.previous.original.controller));new=s.controller_tree()
    class Reverse(ast.NodeTransformer):
        def visit_Assign(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='existing_api':
                n.value=ast.Call(func=ast.Name(id='install_facades',ctx=ast.Load()),args=[],keywords=[])
            return n
    assert ast.dump(old)==ast.dump(Reverse().visit(new))


def test_real_cold_facade_and_actual_controller_initialize_successfully_once(tmp_path):
    # Empty selection is plumbing-only in a temporary fixture,never a benchmark.
    code='\n'.join([
        'import json',
        'from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v4 as s',
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades',
        'api=install_facades()',
        'runtime='+repr(dict(state_root=str(tmp_path/'cold_fixture'),original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273))),
        's.private_controller(api)(runtime,{}, {})',
        'print(json.dumps(dict(passed=True,controller_entry_exercised=True,numerical_workers_started=0)))'])
    env=dict(os.environ,PYTHONPATH=str(s.ROOT)+';'+str(s.ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['passed']


def test_all500_models_215_tasks_and_original_numeric_settings_stay_fixed():
    old=s.read(s.ROOT/s.previous.previous.OLD_CONFIG);c=s.registration()
    for key in ['tasks','settings','resource_lock','python','seeds','datasets','primary_endpoint',
                'paired_test_support','paired_calibration','macro_weighting','interval_scope']:
        assert c[key]==old[key]
    assert c['required_models']==500 and c['required_tasks']==215
    assert c['resource_facades_initialized_once'] and not c['completed_metadata_handoff_repeated']
    assert not c['preserved_duplicate_facade_failure']['previous_statistics_worker_started']
