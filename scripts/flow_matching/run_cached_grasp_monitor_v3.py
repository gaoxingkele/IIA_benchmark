"""Cache telemetry for an independently identified idle full9120 GRASP owner.

All original CUDA jobs,1500epochs,480cohorts,10seeds,score recipes, GPU guards
and35second fairness yield remain unchanged. Only fresh host telemetry binding
and runtime status creation-time metadata differ.
"""
import importlib
import os
from pathlib import Path
from types import FunctionType

from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as base


def historical_identity(config):
    evidence=config['historical_identity']
    if base.sha(base.ROOT/evidence['path'])!=evidence['sha256']:
        raise ValueError('Independent GRASP launch observation changed')
    document=base.read(base.ROOT/evidence['path'])
    matches=[r for r in document['controllers'] if r['name']==evidence['controller_name']]
    if len(matches)!=1:
        raise ValueError('Exact independently observed full GRASP owner required')
    row=matches[0];expected=config['handoff']['previous_owner']
    if (row['pid']!=expected['pid'] or row['create_time']!=expected['create_time']
            or row['actual_command']!=expected['command'] or row['command_token']!=config['entry_module']
            or row['state']['queue_sha256']!=config['expected_state_identity']['queue_sha256']):
        raise ValueError('Historical exact PID/ctime/whole command and full queue identity required')
    return True


def verify(config):
    for pin in config['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full GRASP monitor binding changed: '+pin['path'])
    if (config['entry_module']!='scripts.flow_matching.run_grasp_gpu_memory_fair_queue_v6'
            or config['original_arguments']!=['--runtime-config',config['original_runtime_configuration']]
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('Full original GRASP controller and research scope required')
    module=importlib.import_module(config['entry_module'])
    runtime=base.read(base.ROOT/config['original_runtime_configuration'])
    for pin in runtime['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Original full GRASP runtime changed')
    queue=base.read(base.ROOT/runtime['queue']);module.original.verify(queue)
    if (base.sha(base.ROOT/runtime['queue'])!=runtime['queue_sha256'] or len(queue['jobs'])!=9120
            or queue['cohort_count']!=480 or runtime['yield_seconds']!=35
            or config['settings']!=queue['settings'] or config['resource_lock']!=queue['resource_lock']
            or config['expected_state_identity']!=dict(queue_sha256=runtime['queue_sha256'])
            or config['handoff']['previous_state_path']!=queue['state_root']+'/status.json'):
        raise ValueError('Every original job/cohort/fairness yield/GPU guard and queue identity retained')
    historical_identity(config)
    return module,queue


def safe_phase(config,state,descendants):
    expected=config['handoff']['previous_owner']
    if (state.get('pid')!=expected['pid'] or state.get('active_pid') is not None
            or state.get('state')!='waiting_full_gpu_resources'
            or any(not base.allowed_console_child(p['name'],p['exe']) for p in descendants)
            or state.get('queue_sha256')!=config['expected_state_identity']['queue_sha256']):
        return False
    if state.get('create_time') is None:
        return historical_identity(config)
    return state['create_time']==expected['create_time']


def install_status_identity(module,config,created):
    original=module.original.write_json
    target=(base.ROOT/config['handoff']['previous_state_path']).resolve()
    def write_json(path,data):
        if Path(path).resolve()==target and isinstance(data,dict) and data.get('pid')==os.getpid():
            data=dict(data,create_time=created,resource_monitor_configuration=config['runtime_configuration'])
        return original(path,data)
    module.original.write_json=write_json
    return write_json


def runtime_namespace():
    namespace=dict(base.__dict__)
    for name,value in base.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==base.__name__:
            function=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            function.__kwdefaults__=value.__kwdefaults__;namespace[name]=function
    namespace.update(verify=verify,historical_identity=historical_identity,safe_phase=safe_phase,
        install_status_identity=install_status_identity)
    return namespace


def main():
    runtime_namespace()['main']()


if __name__=='__main__':main()
