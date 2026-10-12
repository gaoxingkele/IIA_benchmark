"""Carry proven unstarted MaelNet log stages through idle cached-monitor handoff."""
import importlib
import os
from pathlib import Path
import shutil
from types import FunctionType

from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as base


def verify(config):
    for pin in config['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen prepared-stage handoff changed: '+pin['path'])
    if config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Full original research scope required')
    if config['entry_module'] not in ('scripts.flow_matching.run_maelnet_stage_resource_queue_v1',
            'scripts.flow_matching.run_maelnet_selected_exact_recovery_v2'):
        raise ValueError('Only original complete MaelNet stage pipelines supported')
    module=importlib.import_module(config['entry_module']); runtime=base.read(base.ROOT/config['original_runtime_configuration'])
    for pin in runtime['source_receipts']:
        if base.sha(base.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen original MaelNet source changed')
    queue=base.read(base.ROOT/runtime['queue'])
    if (base.sha(base.ROOT/runtime['queue'])!=runtime['queue_sha256'] or len(queue['jobs'])!=150
            or config['settings']!=runtime['settings'] or config['resource_lock']!=runtime['resource_lock']
            or config['original_arguments']!=['--runtime-config',config['original_runtime_configuration']]
            or config['expected_state_identity']!=dict(runtime_sha256=base.sha(base.ROOT/config['original_runtime_configuration']),
                queue_sha256=base.sha(base.ROOT/runtime['queue']))):
        raise ValueError('Every150 canonical jobs, unchanged original stage selection and resource settings required')
    job=next(j for j in queue['jobs'] if j['id']==config['pending_job']['id'])
    if job!=config['pending_job']: raise ValueError('Exact requested original source job required')
    if config['entry_module'].endswith('selected_exact_recovery_v2') and runtime['selected_job_id']!=job['id']:
        raise ValueError('Original existing exact recovery selector must stay unchanged')
    return module,queue


def pending_unstarted(config):
    stage=base.ROOT/config['prepared_stage']['path']; output=base.ROOT/config['pending_job']['output_directory']
    if (not stage.is_relative_to(output/'stages') or not stage.is_dir()
            or (output/'result.json').exists() or len(list((output/'stages').iterdir()))!=1): return False
    files=list(stage.iterdir())
    if (len(files)!=2 or {p.name for p in files}!={'stdout.log','stderr.log'}
            or any(not p.is_file() or p.stat().st_size!=0 for p in files)): return False
    for pin in config['workspace_snapshot']:
        path=output/'author_workspace'/pin['relative_path']
        if not path.is_file() or base.sha(path)!=pin['sha256']: return False
    current={p.relative_to(output/'author_workspace').as_posix() for p in (output/'author_workspace').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    return current=={p['relative_path'] for p in config['workspace_snapshot']}


def safe_phase(config,state,descendants):
    old=config['handoff']['previous_owner']
    return (state.get('pid')==old['pid'] and state.get('create_time')==old['create_time']
        and state.get('active_pid') is None and state.get('state')=='waiting_for_memory'
        and state.get('requested_job')==config['pending_job']['id']
        and state.get('requested_command')==config['prepared_stage']['requested_command']
        and all(state.get(k)==v for k,v in config['expected_state_identity'].items())
        and all(base.allowed_console_child(p['name'],p['exe']) for p in descendants) and pending_unstarted(config))


def carry_prepared_stage(config,monitor_base):
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive
    if not pending_unstarted(config): raise ValueError('Only exact proven unstarted empty-log stage may be carried')
    source=(base.ROOT/config['prepared_stage']['path']).resolve(); output=(base.ROOT/config['pending_job']['output_directory']).resolve()
    target=Path(config['prepared_stage_backup']).resolve()
    backup_root=Path('F:/aicoding/IIA_Data/experiments/maelnet_cached_prepared_stage_monitors_v1').resolve()
    if not source.is_relative_to(output/'stages') or not target.is_relative_to(backup_root) or target.exists():
        raise ValueError('Verified bounded source/destination and unused backup path required')
    files={p.name:base.sha(p) for p in source.iterdir()}; target.parent.mkdir(parents=True,exist_ok=True)
    # No raw data, workspace code, trained checkpoint or failed stage is moved.
    # The only source files are both verified empty pre-admission logs. Preserve
    # them outside the stage scanner so the original code can recreate its exact
    # original scientific stage rather than misclassifying it as a failed job.
    shutil.move(str(source),str(target))
    if source.exists() or {p.name:base.sha(p) for p in target.iterdir()}!=files:
        raise ValueError('Original prepared log bytes were not preserved exactly')
    exclusive(monitor_base/'prepared_stage_carry_receipt.json',dict(source=str(source),backup=str(target),files=files,
        only_unstarted_empty_pre_admission_logs_moved=True,workspace_and_raw_data_modified=False,
        original_registered_stage_path_and_command_unchanged=True,training_or_budget_modified=False))


def install_status_identity(module,config,created):
    target=module.original if config['entry_module'].endswith('selected_exact_recovery_v2') else module
    old_write=target.write; path=(base.ROOT/config['handoff']['previous_state_path']).resolve()
    def write(candidate,data):
        if Path(candidate).resolve()==path and isinstance(data,dict) and data.get('pid')==os.getpid():
            data=dict(data,create_time=created,resource_monitor_configuration=config['runtime_configuration'])
        return old_write(candidate,data)
    target.write=write
    return write


def safe_handoff(config,monitor_base,update):
    # Use the existing exact PID/ctime/full-command suspend/recheck/terminate
    # transaction. This only stops a proven idle monitor, never a native child.
    namespace=runtime_namespace()
    namespace['_original_safe_handoff'](config,monitor_base,update)
    carry_prepared_stage(config,monitor_base)


def runtime_namespace():
    namespace=dict(base.__dict__)
    for name,value in base.__dict__.items():
        if isinstance(value,FunctionType) and value.__module__==base.__name__:
            function=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            function.__kwdefaults__=value.__kwdefaults__; namespace[name]=function
    namespace['_original_safe_handoff']=namespace['safe_handoff']
    namespace.update(verify=verify,safe_phase=safe_phase,install_status_identity=install_status_identity,safe_handoff=safe_handoff)
    return namespace


def main():
    runtime_namespace()['main']()


if __name__=='__main__':main()
