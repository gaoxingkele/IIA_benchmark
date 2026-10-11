"""Pure orchestration: unchanged shared GPU/memory guards, complete image jobs."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import uuid

import psutil

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT, completion_valid, read, sha, validate_job, verify_receipts,
)
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_light_controller import load_functions
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def queue_api():
    api = resource_api()
    load_functions(ROOT, 'scripts/flow_matching/run_grasp_protocol_queue.py',
                   ['gpu_snapshot', 'guarded_gpu_wait'], api)
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    runtime = read(ROOT / args.runtime_config)
    verify_receipts(runtime['source_receipts'])
    queue = read(ROOT / runtime['queue'])
    verify_receipts(queue['source_receipts'])
    if len(queue['jobs']) != 20:
        raise ValueError('Full preregistered two-dataset/two-variant/five-seed scope required')
    cohort = set()
    for entry in queue['jobs']:
        job = read(ROOT / entry['path'])
        verify_receipts(job['source_receipts'])
        validate_job(job, read(ROOT / job['model_configuration']),
                     read(ROOT / job['optimizer_configuration']),
                     read(ROOT / job['dataset_configuration']))
        cohort.add((job['dataset'], job['variant'], job['seed']))
    if cohort != {(d, v, s) for d in ('mvtec', 'visa')
                   for v in ('conditional', 'pooled') for s in range(1103, 1108)}:
        raise ValueError('Full distinct dataset/ablation/seed cohort differs')
    if args.validate_only:
        print(dict(full_jobs=20, epochs_per_job=2000, source_and_budget_bindings_verified=True,
                   controller_imported_torch='torch' in sys.modules,
                   controller_imported_numpy='numpy' in sys.modules,
                   original_author_equivalence_certified=False, benchmark_performance=False))
        return
    state_root = ROOT / runtime['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    state = dict(pid=process.pid, create_time=process.create_time(), active_pid=None,
                 registered_jobs=20, complete=False, benchmark_performance=False)

    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(state_root / 'status.json', state)

    api = queue_api()
    lane = api['exclusive_lock'](ROOT / runtime['lane_lock'])
    if lane is None:
        raise RuntimeError('Existing image training controller owns the lane')
    try:
        settings = runtime['settings']
        first_job = read(ROOT / queue['jobs'][0]['path'])
        first_dataset = read(ROOT / first_job['dataset_configuration'])
        disk_root = Path(first_dataset['array_receipts'][0]['path']).parent
        disk = shutil.disk_usage(disk_root)
        if disk.free < runtime['full_cohort_required_free_disk_bytes']:
            update(dict(state='waiting_for_full_checkpoint_disk_capacity', disk_free_bytes=disk.free))
            while disk.free < runtime['full_cohort_required_free_disk_bytes']:
                time.sleep(10)
                disk = shutil.disk_usage(disk_root)
                update(dict(state='waiting_for_full_checkpoint_disk_capacity', disk_free_bytes=disk.free))
        for entry in queue['jobs']:
            job = read(ROOT / entry['path'])
            output = ROOT / job['output_directory']
            final = output / 'training_completed.json'
            if final.exists():
                completion_valid(job, read(final))
                continue
            update(dict(job_id=job['id'], active_pid=None, state='queued'))
            while True:
                retry_gpu = False
                with api['resource_slot'](ROOT / runtime['resource_lock'], settings, update):
                    gpu = api['gpu_snapshot'](settings)
                    if gpu['gpu_free_bytes'] < settings['minimum_gpu_free_bytes']:
                        update(dict(gpu, state='waiting_for_gpu', shared_lock_owned=True))
                        retry_gpu = True
                    else:
                        attempt = output / 'attempts' / uuid.uuid4().hex
                        attempt.mkdir(parents=True, exist_ok=False)
                        command = [str((ROOT / runtime['python']).resolve()), '-X', 'utf8', '-u', '-m',
                            'scripts.flow_matching.run_flow_mismatching_full_training_v1',
                            '--config', entry['path'], '--attempt-directory', str(attempt.resolve())]
                        with (attempt / 'stdout.log').open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
                            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONHASHSEED': str(job['seed']),
                                     'CUDA_VISIBLE_DEVICES': settings['gpu_uuid'],
                                     'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                                     'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
                            child_identity = psutil.Process(child.pid)
                            update(dict(state='running', active_pid=child.pid,
                                active_create_time=child_identity.create_time(), command=command,
                                attempt_directory=str(attempt), shared_lock_owned=True))
                            code, usage = api['guarded_gpu_wait'](child, settings, api, update)
                        receipt = dict(returncode=code, resource_usage=usage,
                                       command=command, worker_pid=child.pid,
                                       worker_create_time=state['active_create_time'])
                        write(attempt / 'terminal.json', receipt)
                        update(dict(state='worker_terminal', active_pid=None, shared_lock_owned=False,
                                    last_terminal=receipt))
                        if code != 0 or usage['resource_guard_aborted']:
                            # Preserve all artifacts. No budget/batch changes or automatic repeated failed launch.
                            update(dict(state='failed_attempt_preserved', complete=False))
                            return
                        if not final.exists():
                            raise ValueError('Worker exited without complete 2,000-epoch receipt')
                        completion_valid(job, read(final))
                update(dict(shared_lock_owned=False))
                if retry_gpu:
                    time.sleep(10)
                    continue
                break
        update(dict(state='all_twenty_full_training_jobs_completed_scores_pending', complete=True,
                    active_pid=None, shared_lock_owned=False))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
