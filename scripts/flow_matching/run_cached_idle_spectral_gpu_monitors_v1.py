"""Keep all180 LS4/SaShiMi jobs and full GPU guards; cache idle host telemetry."""
import importlib
import os
from pathlib import Path
from types import FunctionType

from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as base


ROLES={'ls4_fred_nn5':('scripts.flow_matching.run_ls4_cauchy_corrected_queue_v3',20),
    'ls4_solar_temperature':('scripts.flow_matching.run_ls4_cauchy_corrected_queue_v3',20),
    'ls4_ushcn':('scripts.flow_matching.run_ls4_native_ushcn_queue_v1',100),
    'sashimi':('scripts.flow_matching.run_sashimi_monash_gaussian_queue_v2',40)}


def verify(config):
    for pin in config['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen idle GPU monitor changed: '+pin['path'])
    if config['role'] not in ROLES or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Exact complete registered GPU family and original research scope required')
    module_name,count=ROLES[config['role']]
    if config['entry_module']!=module_name or config['original_arguments']!=['--queue',config['original_runtime_configuration']]:
        raise ValueError('Every original full GPU queue must stay selected')
    module=importlib.import_module(module_name); queue=base.read(base.ROOT/config['original_runtime_configuration'])
    for pin in queue['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Pinned original GPU scientific source changed')
    if config['role']!='ls4_ushcn': module.verify(queue)
    if (len(queue['jobs'])!=count or config['settings']!=queue['settings'] or config['resource_lock']!=queue['resource_lock']
            or config['expected_state_identity']!=dict(queue_sha256=base.sha(base.ROOT/config['original_runtime_configuration']))
            or config['handoff']['previous_state_path']!=queue['state_root']+'/status.json'):
        raise ValueError('All original jobs, data/diagnostic settings, GPU/commit gates and identity must remain')
    for pin in config['completed_diagnostic_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Complete earlier diagnostic changed')
    return module,queue


def pending_unstarted(config):
    if any((base.ROOT/p).exists() for p in config['pending_paths_that_must_remain_absent']): return False
    path=base.ROOT/config['original_log_directory']
    current={p.name:base.sha(p) for p in path.glob('*') if p.is_file()}
    return current==config['completed_log_snapshot']


def safe_phase(config,state,descendants):
    old=config['handoff']['previous_owner']
    return (state.get('pid')==old['pid'] and state.get('create_time')==old['create_time']
        and state.get('active_pid') is None and state.get('state')=='waiting_full_gpu_resources'
        and state.get('active_job') is None and state.get('queue_sha256')==config['expected_state_identity']['queue_sha256']
        and all(base.allowed_console_child(p['name'],p['exe']) for p in descendants) and pending_unstarted(config))


def install_status_identity(module,config,created):
    # guarded_api's function globals belong to the already imported spectral
    # host module. Rebind its future loader only in this fresh successor, so
    # original native dispatch/diagnostic/evaluator functions stay untouched.
    from scripts.flow_matching import run_spectral_original_queue as spectral
    from scripts.flow_matching import run_light_controller as loader
    spectral.load_functions=loader.load_functions
    attribute='write' if config['role']=='ls4_ushcn' else 'write_json'
    original=getattr(module,attribute); target=(base.ROOT/config['handoff']['previous_state_path']).resolve()
    def write(path,data):
        if Path(path).resolve()==target and isinstance(data,dict) and data.get('pid')==os.getpid():
            data=dict(data,create_time=created,resource_monitor_configuration=config['runtime_configuration'])
        return original(path,data)
    setattr(module,attribute,write)
    return write


def runtime_namespace():
    namespace=dict(base.__dict__)
    for name,value in base.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==base.__name__:
            function=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            function.__kwdefaults__=value.__kwdefaults__; namespace[name]=function
    namespace.update(verify=verify,safe_phase=safe_phase,install_status_identity=install_status_identity)
    return namespace


def main():
    runtime_namespace()['main']()


if __name__=='__main__':main()
