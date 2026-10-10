"""Bounded real-data audit and integration for both observed GiFlow tracks."""
import json
import os
from pathlib import Path
import subprocess

import psutil

from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def main():
    path=ROOT/'configs/experiments/fm_giflow_native_queue.v4.json'
    queue=json.loads(path.read_text(encoding='utf-8')); verify(queue)
    base=ROOT/'experiments/runs/fm_giflow_update_audit_v2';base.mkdir(parents=True,exist_ok=True)
    api=guarded_api()
    settings=dict(minimum_available_memory_bytes=12*2**30, minimum_commit_headroom_bytes=22*2**30,
        emergency_commit_headroom_bytes=16*2**30, maximum_worker_rss_bytes=2*2**30,
        maximum_worker_private_bytes=5*2**30)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),queue_sha256=sha(path),cases=[])
    def update(values): state.update(values);write_json(base/'status.json',state)
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
        PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    jobs=[j for j in queue['jobs'] if j['dataset']=='air36' and j['recipe']=='main' and j['seed']==393]
    assert len(jobs)==2
    for job in jobs:
        for mode in ('audit-only','integration-check'):
            ident=job['track']+'__'+mode
            target=base/ident
            if target.exists(): raise FileExistsError('Preserve previous diagnostic attempt')
            cmd=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_giflow_native_job_v2',
                '--queue',str(path),'--job-id',job['id'],'--'+mode,'--audit-output',str(target.relative_to(ROOT))]
            with api['resource_slot'](ROOT/'experiments/runs/fm_readonly_checkpoint_audit_cpu.lock',settings,update):
                with (base/(ident+'.stdout.log')).open('xb') as out,(base/(ident+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),command=cmd))
                    code,peaks=capped_wait(child,{'settings':settings},api,update)
            if code: write_json(base/(ident+'.failure.json'),dict(exit_code=code,**peaks));raise RuntimeError('Actual GiFlow diagnostic failed')
            proof=target/('audit_result.json' if mode=='audit-only' else 'integration_result.json')
            result=json.loads(proof.read_text(encoding='utf-8'))
            if mode=='audit-only':
                assert result['observed_native_train_loader']==dict(samples=6267,batches=48,batch_size=128,drop_last=True)
            else:
                assert result['captured_predictions_match_native_errors'] and result['no_formal_result_published']
            state['cases'].append(dict(track=job['track'],mode=mode,proof=str(proof.relative_to(ROOT)),sha256=sha(proof),**peaks))
            update(dict(active_pid=None))
    verify(queue)
    write_json(base/'validation.json',dict(state,passed=True,diagnostic_only=True,full_native_budget_not_replaced=True))
    print(json.dumps(dict(passed=True,actual_native_loader_batches=48,expected_full300_epoch_updates=14400,
                         full_training_or_benchmark_results_added=0)))


if __name__=='__main__':
    main()
