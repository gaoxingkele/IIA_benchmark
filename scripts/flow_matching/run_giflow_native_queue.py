"""Serialize full native GiFlow with original GPU ownership and resource guards."""
import argparse
import json
import os
import subprocess
import sys
import time

from scripts.flow_matching.giflow_native_protocol import ROOT, complete, sha, verify, write_json
from scripts.flow_matching.run_light_controller import load_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--verify-only', action='store_true')
    cli = parser.parse_args()
    path = ROOT / cli.queue
    queue = json.loads(path.read_text(encoding='utf-8'))
    verify(queue)
    runtime = json.loads((ROOT / queue['runtime_config']).read_text(encoding='utf-8'))
    _, namespace, _ = load_controller(ROOT, runtime, 'crossad')
    if cli.verify_only:
        print(json.dumps({'verified_jobs': len(queue['jobs']), 'queue_sha256': sha(path)}))
        return
    base = ROOT / queue['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = namespace['exclusive_lock'](base / 'worker.lock')
    if lane is None:
        raise RuntimeError('GiFlow queue already owned')
    state = {'pid': os.getpid(), 'queue_sha256': sha(path), 'jobs': {}, 'active_pid': None, 'active_job': None}
    def update(values):
        state.update(values)
        write_json(base / 'status.json', state)
    while True:
        pending = namespace['check_queue'](ROOT / queue['requires_complete_queue'])
        if not pending:
            break
        update({'state': 'waiting_original_grin', 'pending_original_jobs': len(pending)})
        time.sleep(30)
    for job in queue['jobs']:
        if complete(job):
            state['jobs'][job['id']] = 'completed'
            continue
        output = ROOT / job['output_directory']
        if output.exists():
            state['jobs'][job['id']] = 'failed_or_partial_preserved'
            continue
        gpu = None
        while gpu is None:
            gpu = namespace['exclusive_lock'](ROOT / 'experiments/runs/flow_matching_campaign/queue.lock')
            if gpu is None:
                update({'state': 'waiting_original_gpu_lock', 'active_job': None})
                time.sleep(10)
        try:
            with namespace['resource_slot'](ROOT / 'experiments/runs/fm_heavy_recovery_cpu.lock', queue['settings'], update):
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                command = [str(ROOT / queue['python']), '-u', '-m', 'scripts.flow_matching.run_giflow_native_job',
                           '--queue', str(path), '--job-id', job['id']]
                environment = {k.upper(): v for k, v in os.environ.items()}
                environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                                   OMP_NUM_THREADS=str(queue['settings']['cpu_threads']),
                                   MKL_NUM_THREADS=str(queue['settings']['cpu_threads']), WANDB_MODE='disabled')
                with (logs / (job['id'] + '.stdout.log')).open('a', encoding='utf-8') as out, (logs / (job['id'] + '.stderr.log')).open('a', encoding='utf-8') as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                             stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                    update({'state': 'running', 'active_job': job['id'], 'active_pid': child.pid})
                    code, usage = namespace['guarded_wait'](child, queue['settings'], update)
                done = complete(job) if not code else False
                if output.exists():
                    write_json(output / 'resource_usage.json', usage)
                    if not done:
                        write_json(output / 'failure.json', {'exit_code': code, 'resources': usage, 'no_budget_fallback': True})
                state['jobs'][job['id']] = 'completed' if done else 'failed_or_partial_preserved'
                update({'active_job': None, 'active_pid': None})
        finally:
            gpu.close()
    update({'state': 'completed' if all(v == 'completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})


if __name__ == '__main__':
    main()
