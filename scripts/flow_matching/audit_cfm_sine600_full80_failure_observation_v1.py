"""One full80-trajectory observation pass of the saved numerical failure.

Restore the existing complete600/37800 model. The exact native simulate call,
dopri5 tolerances and500-point grids remain unchanged. Observe RHS calls only;
record failures and continue coverage without treating finite subsets as a
benchmark result. CFM-TS: OpenReview Hqn4Aj7xrQ.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
IDENTITY='cfm_ts_original__appendix_d__bb_gaussian_consistent_e600__sine__seed1107'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):digest.update(block)
    return digest.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen diagnostic source changed: '+pin['path'])
    queue=read(ROOT/config['original_queue']);job=next(j for j in queue['jobs'] if j['id']==IDENTITY)
    model=read(ROOT/job['model_config'])
    original=read(ROOT/config['guard_reference_configuration'])
    for key in ['settings','resource_lock','python','cpu_threads','environment_lock','original_scope']:
        if config[key]!=original[key]:raise ValueError('Unchanged original readonly guard required: '+key)
    if config['canonical_job']!=job or job['test_trajectories']!=80 or job['test_points_per_trajectory']!=500:
        raise ValueError('Every original80 full500-point trajectory required')
    if config['solver_parameters']!={k:model['parameters'][k] for k in ['solver','rtol','atol','max_num_steps']}:
        raise ValueError('No original solver tolerance or budget may change')
    if sha(config['checkpoint_path'])!=config['checkpoint_sha256']:
        raise ValueError('Actual saved complete model must remain unchanged')
    return job,model


class RHSObserver:
    """A forward hook returns None and never replaces inputs or outputs."""
    def __init__(self,torch):
        self.torch=torch;self.tail=deque(maxlen=16);self.calls=0
        self.maximum_finite_state=0.;self.maximum_finite_velocity=0.
        self.nonfinite_state_seen=False;self.nonfinite_velocity_seen=False
    def __call__(self,module,inputs,output):
        torch=self.torch;clock,state=inputs
        finite_state=bool(torch.isfinite(state).all());finite_velocity=bool(torch.isfinite(output).all())
        state_max=float(state.detach().abs().max()) if finite_state else None
        velocity_max=float(output.detach().abs().max()) if finite_velocity else None
        if state_max is not None:self.maximum_finite_state=max(self.maximum_finite_state,state_max)
        if velocity_max is not None:self.maximum_finite_velocity=max(self.maximum_finite_velocity,velocity_max)
        self.nonfinite_state_seen|=not finite_state;self.nonfinite_velocity_seen|=not finite_velocity
        self.calls+=1
        value=float(clock.detach())
        self.tail.append(dict(call=self.calls,time=value if self.torch.isfinite(clock).all() else None,
            state_abs_max=state_max,velocity_abs_max=velocity_max,
            state_finite=finite_state,velocity_finite=finite_velocity))
        return None
    def evidence(self):
        return dict(RHS_calls=self.calls,last_RHS_observations=list(self.tail),
            maximum_finite_state_abs=self.maximum_finite_state,maximum_finite_velocity_abs=self.maximum_finite_velocity,
            nonfinite_state_seen=self.nonfinite_state_seen,nonfinite_velocity_seen=self.nonfinite_velocity_seen)


def validity_summary(records,required):
    if len(records)!=required or len({r['trajectory_id'] for r in records})!=required:
        raise ValueError('All distinct original held-out trajectories must be attempted exactly once')
    failed=[r['trajectory_id'] for r in records if not r['finite_full_grid_completed']]
    return dict(required_original_trajectories=required,attempted_original_trajectories=len(records),
        finite_full_grid_trajectories=required-len(failed),failed_trajectory_ids=failed,
        all_original_full_grids_finite=not failed,full_original_MSE=None,
        censored_finite_subset_promoted_as_original_result=False,original_scoring_certified_complete=False)


def worker(config):
    job,_=verify(config)
    import numpy as np
    import torch
    from scripts.flow_matching.run_cfm_ts_original_job import fingerprint
    from scripts.flow_matching.register_cfm_ts_original import environment
    from scripts.flow_matching.model_registry import build_configured_method
    from iia_benchmark.models.fm_cfm_ts_original import PaperTrajectoryField
    if environment()!=read(ROOT/config['environment_lock']):raise ValueError('Original numerical environment required')
    torch.set_num_threads(config['cpu_threads']);torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(config['checkpoint_path'],map_location='cpu',weights_only=False)
    if (checkpoint['job']!=job or checkpoint['job_sha256']!=fingerprint(job)
        or checkpoint['optimizer_updates']!=37800 or len(checkpoint['training_losses'])!=600
        or not np.isfinite(checkpoint['training_losses']).all()):
        raise ValueError('Actual full600/37800 trained checkpoint required')
    model=build_configured_method(ROOT/job['model_config'],job['model_overrides'])
    if model.network_parameters!=checkpoint['network_parameters']:raise ValueError('Original model architecture required')
    model.model=PaperTrajectoryField(job['observed_dimensions'],**model.network_parameters)
    model.model.load_state_dict(checkpoint['state_dict'],strict=True);model.model.eval()
    if not all(torch.equal(v,checkpoint['state_dict'][k]) and torch.isfinite(v).all() for k,v in model.model.state_dict().items()):
        raise ValueError('Every actual trained weight must remain unchanged')
    output=Path(config['output_root']);output.mkdir(parents=True,exist_ok=False)
    records=[]
    with np.load(ROOT/job['data_path'],allow_pickle=False) as data:
        ids=data['test_ids'];predictions=np.full((80,500,1),np.nan,dtype=np.float32)
        for position,index in enumerate(ids):
            observer=RHSObserver(torch);handle=model.model.register_forward_hook(observer)
            entry=dict(position=position,trajectory_id=int(index),initial_state=data['initial_states'][index].tolist(),
                finite_full_grid_completed=False)
            try:
                actual=model.simulate(data['initial_states'][index],np.concatenate([[0.],data['test_times'][index]]))[1:]
                if actual.shape!=(500,1) or not np.isfinite(actual).all():raise ValueError('Full finite original grid required')
                predictions[position]=actual
                entry.update(finite_full_grid_completed=True,MSE=float(np.square(actual.astype(np.float64)-data['test_states'][index]).mean()))
            except (AssertionError,FloatingPointError,RuntimeError,ValueError) as error:
                entry.update(error_type=type(error).__name__,error_message=str(error))
            finally:
                handle.remove()
            entry.update(observer.evidence());records.append(entry)
            write(output/('trajectory_%02d.json'%position),entry)
            write(output/'progress.json',dict(attempted=len(records),required=80,latest_trajectory_id=int(index)))
        np.savez_compressed(output/'all_original_grid_attempts.npz',predictions=predictions,
            truth=data['test_states'][ids],times=data['test_times'][ids],test_ids=ids,
            finite_trajectory_mask=np.array([r['finite_full_grid_completed'] for r in records]))
    verify(config)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_job=job,
        checkpoint_sha256=config['checkpoint_sha256'],solver_parameters=config['solver_parameters'],
        **validity_summary(records,80),records=records,original_native_simulate_code_unchanged=True,
        observer_does_not_replace_outputs=True,numeric_optimizer_updates_executed=0,
        additional_training_seed_slots=0,numerical_root_cause_certified=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
    write(output/'full80_native_failure_observation.json',report)
    print(json.dumps({k:report[k] for k in ['attempted_original_trajectories','finite_full_grid_trajectories','failed_trajectory_ids']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        print(dict(original_trajectories=80,points_each=500,optimizer_updates=0,
            original_solver_parameters=config['solver_parameters'],torch_imported='torch' in __import__('sys').modules));return
    if args.worker:worker(config);return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api();api['resource_snapshot']=resource_snapshot
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The one full80 diagnostic already has an owner')
    process=psutil.Process();state=dict(pid=process.pid,create_time=process.create_time(),active_pid=None,
        config_sha256=sha(ROOT/args.config),all_original_experiments_complete=False)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if Path(config['output_root']).exists() or (base/'receipt.json').exists():raise FileExistsError('Preserve the single diagnostic attempt')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            verify(config)
            command=[config['python'],'-X','utf8','-u','-m',__name__ if __name__!='__main__' else 'scripts.flow_matching.audit_cfm_sine600_full80_failure_observation_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                    PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1'),
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),command=command))
                code,usage=capped_wait(child,config,api,update)
        captured=code==0 and not usage['resource_guard_aborted'] and (Path(config['output_root'])/'full80_native_failure_observation.json').exists()
        write(base/'receipt.json',dict(exit_code=code,full80_failure_observation_captured=captured,
            original_benchmark_scoring_complete=False,**usage))
        update(dict(active_pid=None,state='diagnostic_captured_original_numerical_failure_still_unresolved' if captured else 'diagnostic_failure_preserved_no_repeat'))
    finally:owner.close()


if __name__=='__main__':main()
