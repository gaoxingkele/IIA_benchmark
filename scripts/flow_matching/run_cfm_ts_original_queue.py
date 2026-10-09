"""Serial CPU original-task lane; retain budgets and guard system commit."""
import argparse
from pathlib import Path
import subprocess
import time

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT, read, sha
from scripts.flow_matching.run_cfm_ts_original_job import complete, verify, write
from scripts.flow_matching.heavy_resources import guarded_wait, has_headroom, resource_snapshot
from scripts.flow_matching.run_tsad_queue import exclusive_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',default='configs/experiments/fm_cfm_ts_original_queue.v1.json')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args(); queue = read(ROOT / args.queue); verify(queue)
    if args.verify_only:
        print('Verified full285-job scope and immutable sources'); return
    base = ROOT / queue['state_root']; base.mkdir(parents=True,exist_ok=True)
    handle = exclusive_lock(base/'controller.lock')
    if handle is None:
        raise RuntimeError('Original-task controller already active')
    import os
    state = {'pid':os.getpid(),'queue_sha256':sha(ROOT / args.queue),'jobs':{},'active_pid':None}
    update = lambda p: write(base/'status.json',dict(state,**p))
    try:
        for job in queue['jobs']:
            if complete(job):
                state['jobs'][job['id']]='completed'; continue
            if (ROOT / job['output_directory']).exists() or (base/'receipts'/f"{job['id']}.json").exists():
                state['jobs'][job['id']]='failed_or_partial_preserved'; continue
            memory = resource_snapshot()
            while not has_headroom(memory,queue['resource_policy']):
                update(dict(memory,state='waiting_for_memory'))
                time.sleep(queue['resource_policy']['poll_seconds']); memory=resource_snapshot()
            logs = base/'logs'; logs.mkdir(exist_ok=True)
            command=[str(ROOT / queue['python']),'-u','-m','scripts.flow_matching.run_cfm_ts_original_job',
                     '--queue',args.queue,'--job-id',job['id']]
            with (logs/(job['id']+'.stdout.log')).open('w') as stdout, (logs/(job['id']+'.stderr.log')).open('w') as stderr:
                worker=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                state.update(active_pid=worker.pid,active_job=job['id']); update({'state':'running'})
                code, peaks=guarded_wait(worker,queue['resource_policy'],update=update)
            done=complete(job)
            state['jobs'][job['id']]='completed' if done else 'failed_or_partial_preserved'
            (base/'receipts').mkdir(exist_ok=True)
            write(base/'receipts'/f"{job['id']}.json",dict(exit_code=code,complete=done,resource_observations=peaks))
            state.update(active_pid=None,active_job=None); update({'state':'between_jobs'})
        update({'state':'completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})
    finally:
        handle.close()


if __name__ == '__main__':
    main()
