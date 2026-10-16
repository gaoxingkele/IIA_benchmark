"""Verify three existing SKAB recoveries without overwriting or retraining."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as original
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.audit_industrial_three_recovery_independent_metrics_v1'
TEST='tests/test_fm_industrial_three_recovery_independent_metrics_v1.py'
CONFIG='configs/reproducibility/fm_industrial_three_recovery_independent_metrics.v1.json'
RECOVERY='configs/experiments/fm_resource_recovery_queue.v4.json'


def validate_identity(case,jobs):
    canonical=case['original_id'];old=jobs[canonical];new=case['job']
    if case['kind']!='industrial' or case['original_job']!=old or old['dataset']!='skab':
        raise ValueError('Exact original industrial SKAB identity required')
    ignored={'id','output_directory'}
    if ({k:v for k,v in new.items() if k not in ignored}!={k:v for k,v in old.items() if k not in ignored}
            or new['id']!=old['id']+'__resource_recovery_v2' or new['output_directory']==old['output_directory']):
        raise ValueError('Only explicit recovered identity and separate output may differ')
    return new


def registration():
    c=read(ROOT/original.CONFIG)
    pins=[MODULE.replace('.','/')+'.py',TEST,RECOVERY,original.CONFIG,original.MODULE.replace('.','/')+'.py']
    return dict(original_audit_configuration=original.CONFIG,recovery_queue=RECOVERY,required_recovery_cases=3,
        original_CPU_scope=550,original_GPU_scope=850,original_global_scope=c['original_global_scope'],
        state_root='experiments/runs/fm_industrial_three_recovery_independent_metrics_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_three_recovery_independent_metrics_v1',
        resource_lock=c['resource_lock'],settings=c['settings'],source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        no_original_or_recovery_output_writes=True,new_optimizer_updates=0,new_neural_forwards=0,new_training_seed_slots=0,
        source_checkpoint_neural_replay=False,source_historical_current_gate_certification=False,
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Frozen recovery mapping, complete scope or bounded audit guards changed')
    oc=read(ROOT/original.CONFIG);q,manifest=original.verify(oc);jobs={j['id']:j for j in q['jobs']}
    cases=[x for x in read(ROOT/RECOVERY)['jobs'] if x['kind']=='industrial']
    if len(cases)!=3 or len({x['original_id'] for x in cases})!=3:raise ValueError('All three distinct original recovered slots required')
    for case in cases:
        validate_identity(case,jobs)
        if case['evaluation']!=q['evaluation'] or case['original_queue']!=original.QUEUE or case['original_queue_sha256']!=sha(ROOT/original.QUEUE):
            raise ValueError('Original data, seed, full training/scoring and all evaluation settings required')
    return oc,q,manifest,cases


def worker(c):
    oc,q,manifest,cases=verify(c);records={}
    for case in cases:
        job=case['job'];usage_path=ROOT/job['output_directory']/'resource_usage.json';usage=read(usage_path)
        if usage['resource_guard_aborted']:raise ValueError('Preserve source recovery resource failure; no new attempt')
        report=original.audit(oc,q,manifest,job)
        records[case['original_id']]=dict(canonical_id=case['original_id'],actual_execution_id=job['id'],
            original_and_recovered_jobs_identical_except_id_and_output=True,
            source_recovery_resource_usage=usage,source_recovery_resource_usage_sha256=sha(usage_path),independent_report=report)
    target=Path(c['artifact_root']);target.mkdir(parents=True,exist_ok=False)
    write(target/'report.json',dict(passed=True,records=records,source_recovery_slots=3,original_CPU_scope=550,
        source_checkpoint_neural_replay=False,source_historical_current_gate_certification=False,
        new_optimizer_updates=0,new_neural_forwards=0,new_training_seed_slots=0,all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    if args.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if args.probe_only:
        print(json.dumps(dict(passed=True,required_recovery_cases=3,numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,numerical_workers_started=0)));return
    if args.worker:worker(c);return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Three-recovery audit already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,all_original_experiments_complete=False)
    def update(v):
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if (base/'worker.stdout.log').exists():raise RuntimeError('Preserve terminal audit attempt; no automatic retry')
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='independent_recovery_metrics',active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,c,api,update)
            proof=Path(c['artifact_root'])/'report.json'
            passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None))
            if not passed:raise RuntimeError('Preserved terminal independent recovery audit failure')
        update(dict(state='complete',active_pid=None,passed=True))
    except Exception as error:
        update(dict(state='terminal_failure_preserved',active_pid=None,error=str(error)));raise
    finally:owner.close()


if __name__=='__main__':main()
