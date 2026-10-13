"""Lightweight single-owner dispatch with unchanged CrossAD science and guards."""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil
from scripts.flow_matching.crossad_gecco1103_full_retry_v1 import (
    ROOT, CANONICAL, read, verify, verify_completed,
)
from scripts.flow_matching.run_light_controller import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.config
    config = read(path)
    queue, job, api = verify(config)
    if args.probe_only:
        print(json.dumps({'required_original_jobs': len(queue['jobs']),
            'selected_original_id': CANONICAL, 'original_max_epochs': 20,
            'credited_previous_epochs_for_fresh_retry': 0, 'additional_seed_slots': 0,
            'minimum_commit_headroom_bytes': config['settings']['minimum_commit_headroom_bytes'],
            'emergency_commit_headroom_bytes': config['settings']['emergency_commit_headroom_bytes'],
            'numpy_imported': 'numpy' in sys.modules, 'torch_imported': 'torch' in sys.modules,
            'training_executed': False, 'all_original_experiments_complete': False}))
        return
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lock = api['exclusive_lock'](base / 'worker.lock')
    if lock is None:
        raise RuntimeError('This retry already has a live owner')
    state = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
        'configuration_sha256': digest(path), 'required_original_jobs': 139,
        'selected_original_id': CANONICAL, 'active_pid': None,
        'all_original_experiments_complete': False, 'original_max_epochs': 20,
        'credited_previous_epochs_for_fresh_retry': 0}

    def update(values):
        state.update(values)
        state['captured_unix'] = time.time()
        api['write_json'](base / 'status.json', state)

    try:
        output = ROOT / job['output_directory']
        if (output / 'result.json').exists():
            verify_completed(ROOT, job, api)
            update({'state': 'selected_full_native_pipeline_verified'})
            return
        if (output / 'workspace').exists() or (output / 'failure.json').exists():
            update({'state': 'failed_or_partial_preserved_no_automatic_repeat'})
            return
        with api['resource_slot'](ROOT / config['shared_lock'], config['settings'], update) as start:
            # Recheck the old identity and immutable evidence after any long wait.
            if digest(path) != state['configuration_sha256']:
                raise ValueError('Live retry configuration changed')
            queue, job, api = verify(config)
            output.mkdir(parents=True, exist_ok=True)
            api['write_json'](output / 'native_preflight_binding.json', config['native_preflight_binding'])
            command = [config['execution_python'], '-u', str(ROOT / job['runner']),
                '--queue', str(ROOT / config['native_queue']), '--job-id', CANONICAL]
            env = {k.upper(): v for k, v in os.environ.items()}
            env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                OMP_NUM_THREADS=str(queue['settings']['cpu_threads']),
                MKL_NUM_THREADS=str(queue['settings']['cpu_threads']))
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            with (output / 'stdout.log').open('xb') as out, (output / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
                created = psutil.Process(child.pid).create_time()
                update({'state': 'running_full_original_pipeline', 'active_pid': child.pid,
                    'active_create_time': created, 'actual_command': command, 'start_resources': start})
                code, usage = api['guarded_wait'](child, queue['settings'], update)
            api['write_json'](output / 'resource_usage.json', dict(usage, start_resources=start))
            receipt = {'exit_code': code, 'resource_usage': usage,
                'canonical_id': CANONICAL, 'actual_command': command,
                'native_identity': {'pid': child.pid, 'create_time': created},
                'original_failure_preserved': config['incident']['terminal_failure_path'],
                'full_native_pipeline_verified': False}
            if code or usage['resource_guard_aborted'] or not (output / 'result.json').exists():
                api['write_json'](output / 'failure.json', receipt)
                update({'state': 'failed_or_partial_preserved_no_automatic_repeat', 'active_pid': None})
            else:
                verify_completed(ROOT, job, api)
                receipt['full_native_pipeline_verified'] = True
                update({'state': 'selected_full_native_pipeline_verified', 'active_pid': None})
            api['write_json'](base / 'receipt.json', receipt)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
