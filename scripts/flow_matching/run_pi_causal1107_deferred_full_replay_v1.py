"""Wait for the exact original worker, then freeze and run its full replay.

No original numerical process is stopped or restarted. No incomplete source
result is promoted. All numerical stages retain their original guards/budgets.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching import register_pi_causal1107_completed_full_replay_v1 as registration
from scripts.flow_matching import run_pi_causal1107_completed_full_replay_v1 as replay
from scripts.flow_matching.run_light_controller import digest
from scripts.flow_matching.timegan_native_common_v1 import write

ROOT = replay.ROOT
REGISTER_MODULE = 'scripts.flow_matching.register_pi_causal1107_completed_full_replay_v1'
REPLAY_MODULE = 'scripts.flow_matching.run_pi_causal1107_completed_full_replay_v1'
RUNTIME = 'configs/runtime/fm_pi_causal1107_completed_full_replay.v1.json'
ORIGINAL_QUEUE = 'configs/experiments/fm_pi_transformer_author_queue.v1.json'
REVIEW_TEMPLATE = 'configs/reproducibility/fm_pi_original_completed_full_replay_review.2026-10-11_v1.json'


def verify(config):
    if (config['canonical_id'] != replay.CANONICAL or config['original_scope'] != replay.SCOPE
            or config['original_budget'] != replay.BUDGET or config['completed_replay_runtime'] != RUNTIME
            or config['original_queue'] != ORIGINAL_QUEUE or config['poll_seconds'] != 5):
        raise ValueError('Preserve exact original causal1107, full research scope and numerical budgets')
    if config['python'] != registration.read(REVIEW_TEMPLATE)['python']:
        raise ValueError('The unchanged full-replay interpreter is required')
    for pin in config['source_receipts']:
        if digest(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen deferred-replay source changed: ' + pin['path'])
    queue = registration.read(config['original_queue'])
    if (len(queue['jobs']) != 1140 or len({j['id'] for j in queue['jobs']}) != 1140
            or queue['settings']['seeds'] != [42, 1103, 1104, 1105, 1106, 1107]):
        raise ValueError('All1140 original jobs and six seeds remain required')
    job = next(j for j in queue['jobs'] if j['id'] == replay.CANONICAL)
    if (job['dataset'], job['branch'], job['seed']) != ('PSM', 'causal_axis_repaired', 1107):
        raise ValueError('Exact original causal PSM job required')
    command = config['source_worker']['command']
    if (command[command.index('--job-id') + 1] != replay.CANONICAL
            or (ROOT / command[command.index('--queue') + 1]).resolve()
            != (ROOT / config['original_queue']).resolve()):
        raise ValueError('Exact original numerical source command required')
    return queue, job


def ready_or_exact_worker_live(config, queue):
    try:
        registration.require_original_complete(queue)
        return True
    except (OSError, ValueError) as pending_error:
        expected = config['source_worker']
        try:
            worker = psutil.Process(expected['pid'])
        except psutil.NoSuchProcess:
            raise RuntimeError('Original worker is absent without a complete verified source; preserve the attempt, never retry training automatically') from pending_error
        if worker.create_time() != expected['create_time'] or worker.cmdline() != expected['command']:
            raise RuntimeError('Original source PID/time/command changed; preserve the source and inspect') from pending_error
        return False


def replay_command(python, runtime=RUNTIME):
    bootstrap = ("import sys,runpy;from scripts.flow_matching.run_cached_resource_monitor_controller_v1 "
                 "import install_cached_loader;install_cached_loader(lambda snapshot:None);"
                 "sys.argv=['full-native-replay','--config'," + repr(runtime) + "];"
                 "runpy.run_module(" + repr(REPLAY_MODULE) + ",run_name='__main__')")
    return [python, '-B', '-X', 'utf8', '-u', '-c', bootstrap]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = registration.read(args.config)
    configuration_sha = digest(ROOT / args.config)
    queue, job = verify(config)
    ready = ready_or_exact_worker_live(config, queue)
    if args.probe_only:
        print(json.dumps(dict(passed=True, canonical_id=replay.CANONICAL,
            required_original_jobs=1140, original_training_seeds=[42,1103,1104,1105,1106,1107],
            original_source_already_complete=ready, original_source_worker_exact_live=not ready,
            future_full_replay_runtime_created=(ROOT / RUNTIME).exists(),
            numerical_training_executed=False, original_numeric_worker_changed=False,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            **replay.BUDGET)))
        return
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda snapshot: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = resource_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Exact deferred source watcher already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        configuration_sha256=configuration_sha, canonical_id=replay.CANONICAL,
        active_pid=None, state='starting', all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)

    def launch(command, stage):
        with (base / (stage + '.stdout.log')).open('xb') as out, (base / (stage + '.stderr.log')).open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err,
                env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                         PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        update(dict(state=stage, active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
        return child

    try:
        while not ready:
            update(dict(state='waiting_exact_original_full_training_result',
                        original_source_worker=config['source_worker'], active_pid=None))
            time.sleep(config['poll_seconds'])
            ready = ready_or_exact_worker_live(config, queue)
        if digest(ROOT / args.config) != configuration_sha:
            raise ValueError('Deferred configuration changed while waiting')
        verify(config)
        registration.require_original_complete(queue)
        if (ROOT / RUNTIME).exists():
            raise FileExistsError('Preserve any existing registration; inspect its owner instead of rerunning it')
        update(dict(state='complete_original_source_verified', original_result_sha256=digest(ROOT/job['output_directory']/'result.json')))
        child = launch([config['python'], '-B', '-X', 'utf8', '-u', '-m', REGISTER_MODULE], 'freezing_completed_original_replay')
        while child.poll() is None:
            time.sleep(config['poll_seconds'])
        if child.returncode != 0:
            raise RuntimeError('Preserve registration failure and source; no numerical retry was made')
        runtime = registration.read(RUNTIME)
        replay.verify(runtime)
        child = launch(replay_command(config['python']), 'running_original_point_native_independent_replay')
        while child.poll() is None:
            time.sleep(config['poll_seconds'])
        pipeline = registration.read(runtime['state_root'] + '/status.json')
        passed = child.returncode == 0 and pipeline.get('verified_ready_original_ids') == [replay.CANONICAL]
        write(base/'receipt.json', dict(exit_code=child.returncode,
            requested_case_full_replay_pipeline_passed=passed, original_training_restarted=False,
            added_training_seed_slots=0, all_original_experiments_complete=False))
        update(dict(state='requested_original_full_replay_complete' if passed else 'failed_pipeline_preserved', active_pid=None))
        if not passed:
            raise RuntimeError('Full pipeline did not pass; preserve all actual guarded evidence')
    except Exception as error:
        update(dict(state='failed_preserved_no_numerical_retry', failure_type=type(error).__name__,
                    failure_message=str(error)))
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
