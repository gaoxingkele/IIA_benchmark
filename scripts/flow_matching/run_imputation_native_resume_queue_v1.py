"""Resume frozen GRIN/industrial imputation queues without changing model commands."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.phase_gate import check_queue
from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_tsad_native_gpu_resource_queue_v1 import native_slot


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_registration(root, registration):
    for entry in registration['source_receipts']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('Frozen imputation source differs: ' + entry['path'])
    path = root / registration['queue']
    if digest(path) != registration['queue_sha256']:
        raise ValueError('Full original queue differs')
    queue = read(path)
    for job in queue['jobs']:
        config_path = root / job['config']
        if digest(config_path) != job['config_sha256']:
            raise ValueError('Frozen original configuration differs')
        config = read(config_path)
        cuda = (config.get('device') == 'cuda:0' or
                job['runner'] == 'scripts/flow_matching/run_cfmi.py' and 'device' not in config)
        if (job['runner'] not in registration['allowed_runners']
                or config.get('diagnostic') or config.get('limit_test_cases')
                or not cuda):
            raise ValueError('Full registered CUDA imputation job required')
    if len(queue['jobs']) != registration['full_job_count']:
        raise ValueError('Original full job coverage differs')
    return queue


def original_gates(root, queue):
    """Use the original run_queue gates; a pending gate does not authorize dispatch."""
    pending = []
    if queue.get('requires_complete_queue'):
        pending = check_queue(root / queue['requires_complete_queue'], root)
    scope = None
    if queue.get('requires_original_scope_review'):
        campaign = read(root / queue['requires_original_scope_review'])
        scope = campaign['execution'].get('original_scope_status')
    ready = not pending and (scope is None or scope == 'completed_or_documented_unavailable')
    return dict(ready=ready, prerequisite_pending=len(pending),
                pending_examples=pending[:3], original_scope_status=scope)


def command_for(root, queue, job):
    # Exactly the original run_queue child arguments; no budget/seed/data overrides.
    return [str(root / queue['python']), str(root / job['runner']),
            '--config', str(root / job['config'])]


def runner_result(root, job, author_budget=None):
    """Check original runner completion, keeping independent metric audit separate."""
    config = read(root / job['config'])
    output = root / config['output_root']
    result_path = output / 'result.json'
    if not result_path.exists():
        return False
    result = read(result_path)
    if (result.get('status') != 'completed' or result.get('config_sha256') != job['config_sha256']
            or result.get('config') != config or result.get('diagnostic')
            or result.get('config', {}).get('diagnostic')
            or result.get('config', {}).get('limit_test_cases')):
        raise ValueError('Invalid full original runner result')
    metrics = result.get('metrics', {})
    required = ('mae', 'mse', 'rmse', 'mre', 'mape') if config.get('paper_id') == 'grin' else ('mae', 'rmse', 'crps')
    if any(not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key])
           for key in required):
        raise ValueError('Full original metrics missing or non-finite')
    if config.get('paper_id') == 'grin':
        args = result['author_arguments']
        if (args['seed'] != config['seed'] or args['dataset_name'] != config['dataset']
                or args['model_name'] != 'grin' or not author_budget
                or any(args.get(key) != value for key, value in author_budget.items())
                or not 0 < result['completed_epochs'] <= args['epochs'] + 1):
            raise ValueError('Full GRIN author arguments differ')
        for path in (root / result['selected_checkpoint'], output / 'checkpoints/last.ckpt',
                     output / 'test_month_statistics.npz'):
            if not path.is_file():
                raise ValueError('Full GRIN training/evaluation artifact missing')
    return True


def preserve_inputs(root, config, base):
    output = root / config['output_root']
    archive = base / 'preserved_resume_inputs' / config['id']
    if archive.exists():
        raise FileExistsError('Prior exact imputation resume inputs remain intact')
    archive.mkdir(parents=True)
    entries = []
    if output.exists():
        # Copy existing checkpoints/logs before the unchanged author runner resumes.
        for source in output.rglob('*'):
            if source.is_file():
                target = archive / source.relative_to(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if digest(source) != digest(target):
                    raise ValueError('Preserved native resume artifact differs')
                entries.append(dict(original=str(source), archived=str(target), sha256=digest(target)))
    write(archive / 'manifest.json', dict(original_job=config['id'], preserved=entries,
          native_runner_resume_unchanged=True))


@contextmanager
def original_lane(api, path, update):
    while True:
        handle = api['exclusive_lock'](path)
        if handle is not None:
            break
        update(dict(state='waiting_original_imputation_lane'))
        time.sleep(10)
    try:
        yield
    finally:
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    runtime_path = ROOT / args.runtime_config
    registration = read(runtime_path)
    queue = verify_registration(ROOT, registration)
    if args.probe_only:
        print(json.dumps(dict(full_jobs=len(queue['jobs']), gates=original_gates(ROOT, queue),
            pending_original_results=len(check_queue(ROOT / registration['queue'])),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    api = guarded_api()
    base = ROOT / registration['state_root']
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('This full imputation resume controller is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        runtime_sha256=digest(runtime_path), queue_sha256=registration['queue_sha256'],
        state='waiting_original_phase_gates', active_job=None, active_pid=None, jobs={})
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    update({})
    environment = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        while True:
            gates = original_gates(ROOT, queue)
            if gates['ready']:
                break
            update(dict(state='waiting_original_phase_gates', gates=gates))
            time.sleep(15)
        with original_lane(api, ROOT / registration['original_queue_lock'], update):
            # Recheck after lane admission, matching original queue prerequisites.
            if not original_gates(ROOT, queue)['ready']:
                raise RuntimeError('Original prerequisite changed before lane admission')
            for job in queue['jobs']:
                config = read(ROOT / job['config'])
                identity = config['id']
                budget = registration['author_budgets'].get(identity)
                if runner_result(ROOT, job, budget):
                    state['jobs'][identity] = 'completed_original_runner_result'
                    update({})
                    continue
                command = command_for(ROOT, queue, job)
                for process in psutil.process_iter(['pid', 'cmdline']):
                    supplied = process.info['cmdline'] or []
                    if '--config' in supplied and any(job['runner'].replace('/', '\\') in x.replace('/', '\\') for x in supplied):
                        path = supplied[supplied.index('--config') + 1]
                        if (ROOT / path).resolve() == (ROOT / job['config']).resolve():
                            raise RuntimeError('Original full model already live: ' + str(process.pid))
                update(dict(state='waiting_native_gpu_slot', requested_job=identity))
                with native_slot(api, registration, update):
                    preserve_inputs(ROOT, config, base)
                    logs = base / 'logs'
                    logs.mkdir(exist_ok=True)
                    with (logs / (identity + '.stdout.log')).open('xb') as out, (logs / (identity + '.stderr.log')).open('xb') as err:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = psutil.Process(child.pid)
                        update(dict(state='running', active_job=identity, active_pid=child.pid,
                            active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                        code, usage = api['guarded_gpu_wait'](child, registration['settings'], api, update)
                    valid = code == 0 and runner_result(ROOT, job, budget)
                    write(base / 'receipts' / (identity + '.json'), dict(original_job=identity,
                        command=command, exit_code=code, original_runner_result_validated=valid,
                        independent_metric_audit_required=True, full_budget_retained=True, **usage))
                state['jobs'][identity] = 'completed_original_runner_result' if valid else 'failed_or_partial_preserved'
                update(dict(state='between_full_jobs', active_job=None, active_pid=None))
                if not valid:
                    raise RuntimeError('Original full imputation job failed; preserve exact native state')
                time.sleep(registration['settings']['controller_yield_seconds'])
            update(dict(state='completed_original_runner_results', independent_metric_audit_required=True))
    except BaseException as error:
        update(dict(state='failed_or_partial_preserved', error=repr(error)))
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
