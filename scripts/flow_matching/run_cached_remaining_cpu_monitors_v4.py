"""Equivalent cached telemetry for three unstarted bounded CPU queue owners.

Every original105 entropic,1140 Pi and72 MOMENT job, all32GiB commit admission,
16GiB emergency reserve,10/12GiB worker caps and scientific dispatch are retained.
Prepared Pi empty append-only logs are preserved; no model artifact may exist.
"""
import importlib
import os
from pathlib import Path
from types import FunctionType

from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as base


def verify(config):
    for pin in config['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen remaining CPU monitor binding changed: '+pin['path'])
    if (config['entry_module']!='scripts.flow_matching.run_bounded_remaining_cpu_queues_v3'
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or config['historical_identity'] is not None):
        raise ValueError('Original complete CPU protocol and explicit creation-time identity required')
    module=importlib.import_module(config['entry_module'])
    runtime=base.read(base.ROOT/config['original_runtime_configuration'])
    role=config['role']
    if config['original_arguments']!=['--runtime-config',config['original_runtime_configuration'],'--controller',role]:
        raise ValueError('Exact original full CPU queue selection required')
    module.bind(runtime,role)
    counts={'entropic_cpu':105,'pi_cpu':1140,'moment_cpu':72}
    selected=runtime['controllers'][role];queue=base.read(base.ROOT/selected['queue'])
    if (len(queue['jobs'])!=counts[role] or config['settings']!=runtime['settings']
            or config['resource_lock']!=runtime['resource_lock']
            or config['expected_state_identity']!=dict(runtime_sha256=base.sha(base.ROOT/config['original_runtime_configuration']),
                queue_sha256=selected['queue_sha256'])
            or config['handoff']['previous_state_path']!=runtime['state_root']+'/'+role+'/status.json'):
        raise ValueError('All original jobs/unchanged bounded resource settings and status identity required')
    matches=[j for j in queue['jobs'] if j['id']==config['pending_job']['id']]
    if len(matches)!=1 or matches[0]!=config['pending_job']:
        raise ValueError('Exact originally requested unstarted scientific job required')
    return module,queue


def pending_unstarted(config):
    job=config['pending_job'];output=base.ROOT/job['output_directory']
    runtime=base.read(base.ROOT/config['original_runtime_configuration'])
    receipt=base.ROOT/runtime['state_root']/config['role']/'receipts'/(job['id']+'.json')
    if receipt.exists():return False
    if not output.exists():return True
    if config['role']!='pi_cpu':return False
    files=list(output.iterdir())
    # Original Pi creates these append-only logs before the CPU reservation.
    # Do not accept checkpoints, failures, configuration files or nonempty logs.
    return (len(files)==2 and {p.name for p in files}=={'stdout.log','stderr.log'}
        and all(p.is_file() and p.stat().st_size==0 for p in files))


def safe_phase(config,state,descendants):
    expected=config['handoff']['previous_owner']
    return (state.get('pid')==expected['pid'] and state.get('create_time')==expected['create_time']
        and state.get('active_pid') is None and state.get('state')=='waiting_for_memory'
        and state.get('requested_job')==config['pending_job']['id']
        and all(state.get(k)==v for k,v in config['expected_state_identity'].items())
        and all(base.allowed_console_child(p['name'],p['exe']) for p in descendants)
        and pending_unstarted(config))


def install_status_identity(module,config,created):
    original=module.original.write
    target=(base.ROOT/config['handoff']['previous_state_path']).resolve()
    def write(path,data):
        if Path(path).resolve()==target and isinstance(data,dict) and data.get('pid')==os.getpid():
            data=dict(data,create_time=created,resource_monitor_configuration=config['runtime_configuration'])
        return original(path,data)
    module.original.write=write
    return write


def runtime_namespace():
    namespace=dict(base.__dict__)
    for name,value in base.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==base.__name__:
            function=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            function.__kwdefaults__=value.__kwdefaults__;namespace[name]=function
    namespace.update(verify=verify,safe_phase=safe_phase,install_status_identity=install_status_identity)
    return namespace


def main():
    runtime_namespace()['main']()


if __name__=='__main__':main()
