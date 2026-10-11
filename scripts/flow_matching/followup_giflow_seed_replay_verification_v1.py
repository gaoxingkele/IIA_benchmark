"""Follow the actual full native replay to its independent all-array verification.

Never restart, stop, or modify the producer. A frozen source/result registration,
successful actual terminal receipt and complete proof are required before audit.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as out:
        out.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def verify_registration(config,root=ROOT):
    for pin in config['source_receipts']:
        if sha(root/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen actual replay followup source changed: '+pin['path'])
    replay=read(root/config['replay_config'])
    queue=read(root/replay['queue'])
    job=next(j for j in queue['jobs'] if j['id']==replay['job_id'])
    result=read(root/job['output_directory']/'result.json')
    if (job['id']!=config['canonical_id'] or job['seed']!=394 or job['track']!='reviewed_corrected'
            or result['job']!=job or result['epochs_completed']!=68 or result['optimizer_updates']!=3264
            or result['test_targets_with_window_repetitions']!=264837 or result['full_test_batches']!=14
            or result['data_binding']['windows']['test']!=1747):
        raise ValueError('Actual complete original corrected seed394 model/data/result required')
    return replay,job,result

def successful_terminal(receipt):
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('full_model_replay_passed') is not True):
        raise ValueError('Successful full native terminal model replay required')

def independent_success(proof):
    coverage=proof.get('coverage',{})
    return (proof.get('full_model_replay_and_independent_saved_arrays_verified') is True
            and proof.get('seed')==394
            and coverage.get('full_test_windows')==1747
            and coverage.get('full_test_batches')==14
            and coverage.get('full_masked_test_targets')==264837
            and coverage.get('observed_forward_calls')==280
            and proof.get('additional_training_seed_slots')==0)

def exact_producer_live(config,process=psutil.Process):
    p=process(config['expected_producer_pid'])
    return (abs(p.create_time()-config['expected_producer_create_time'])<.001
            and config['producer_module'] in p.cmdline()
            and config['replay_config'] in p.cmdline())

def clarified_review(raw,reviewed,result):
    if (raw['job_id']!=result['job']['id'] or raw['seed']!=394
            or not raw['full_checkpoint_contents_and_reinference_passed']
            or reviewed['original_raw_replay_proof_sha256']!=result['_raw_sha256']):
        raise ValueError('Complete original raw replay proof identity required')
    for k,v in raw.items():
        if k!='boundary' and reviewed.get(k)!=v:
            raise ValueError('Reviewed metadata may not change any actual replay field')
    corrected=copy.deepcopy(reviewed)
    targets=result['test_targets_with_window_repetitions']
    if targets!=264837 or reviewed['full_masked_test_targets']!=targets:
        raise ValueError('Actual seed-specific full masks must agree before wording clarification')
    corrected['boundary']=reviewed['boundary'].replace('268613 masked targets',str(targets)+' masked targets')
    if '268613 masked targets' in corrected['boundary']:
        raise ValueError('Other seed target count remains in reviewed wording')
    return corrected

def audit_after_terminal(config,replay,result,root=ROOT):
    successful_terminal(read(root/config['worker_receipt']))
    raw_path=root/config['raw_proof']
    original_review=root/config['producer_reviewed_proof']
    result=dict(result,_raw_sha256=sha(raw_path))
    clarified=clarified_review(read(raw_path),read(original_review),result)
    reviewed_path=root/config['reviewed_proof']
    write_new(reviewed_path,clarified)
    generated=dict(replay_config=config['replay_config'],worker_receipt=config['worker_receipt'],
        raw_proof=config['raw_proof'],reviewed_proof=config['reviewed_proof'],output_directory=config['output_directory'])
    pins={pin['path'] for pin in config['source_receipts']}
    pins.update([config['worker_receipt'],config['raw_proof'],config['producer_reviewed_proof'],config['reviewed_proof']])
    generated['source_receipts']=[dict(path=p,sha256=sha(root/p)) for p in sorted(pins)]
    generated_path=root/config['verification_config']
    write_new(generated_path,generated)
    return generated_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.config)
    replay,job,result=verify_registration(config)
    if args.probe_only:
        print(json.dumps(dict(canonical_id=job['id'],full_windows=1747,full_batches=14,
            full_masked_targets=264837,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            producer_live=exact_producer_live(config))))
        return
    from scripts.flow_matching.heavy_resources import exclusive_lock
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base=ROOT/config['state_root']
    base.mkdir(parents=True,exist_ok=True)
    owner=exclusive_lock(base/'controller.lock')
    if owner is None:
        raise RuntimeError('Independent seed394 replay followup already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),config_sha256=sha(ROOT/args.config),
        active_pid=None,state='waiting_actual_full_native_replay_terminal',additional_training_seed_slots=0)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat())
        (base/'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    try:
        while not (ROOT/config['worker_receipt']).exists():
            try:
                live=exact_producer_live(config)
            except psutil.NoSuchProcess:
                live=False
            except (psutil.AccessDenied,OSError) as error:
                update(dict(state='waiting_producer_observation_retry',observation_error=type(error).__name__))
                time.sleep(10)
                continue
            update(dict(state='waiting_actual_full_native_replay_terminal',producer_identity_live=live))
            if not live:
                # Receipt writer/process exit can race. Recheck the same producer;
                # this wait neither restarts work nor treats a timeout as success.
                time.sleep(10)
                if (ROOT/config['worker_receipt']).exists():
                    break
                try:
                    if exact_producer_live(config):
                        continue
                except psutil.NoSuchProcess:
                    pass
                raise RuntimeError('Producer identity missing and no terminal receipt; preserve incomplete replay')
            time.sleep(10)
        verify_registration(config)
        successful_terminal(read(ROOT/config['worker_receipt']))
        api=resource_api()
        with resource_slot(ROOT/config['resource_lock'],config['settings'],update):
            generated=audit_after_terminal(config,replay,result)
            command=[sys.executable,'-X','utf8','-u','-m',config['verifier_module'],
                     '--config',generated.relative_to(ROOT).as_posix()]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
                        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),
                    state='verifying_every_completed_native_replay_array',actual_command=command))
                code,usage=capped_wait(child,config,api,update)
        path=ROOT/config['output_directory']/'independent_full_replay_verification.json'
        passed=code==0 and path.exists() and independent_success(read(path))
        write_new(base/'receipt.json',dict(exit_code=code,independent_full_replay_passed=passed,**usage))
        update(dict(active_pid=None,state='completed' if passed else 'failed_preserved'))
        if not passed:
            raise RuntimeError('Actual full replay independent verification failed; preserve evidence')
    except BaseException as error:
        update(dict(state='failed_preserved',error=repr(error)))
        raise
    finally:
        owner.close()

if __name__=='__main__':
    main()
