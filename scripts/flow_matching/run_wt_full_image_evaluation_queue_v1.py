"""Pure serial full-score/replay controller; never imports Torch or NumPy."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
import uuid

import psutil

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_training_queue_v1 import queue_api
from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import validate_queue, complete
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


@contextmanager
def full_gpu_slot(api, runtime, update):
    settings = runtime['settings']
    while True:
        with api['resource_slot'](ROOT / runtime['resource_lock'], settings, update):
            gpu = api['gpu_snapshot'](settings)
            if gpu['gpu_free_bytes'] >= settings['minimum_gpu_free_bytes']:
                yield
                return
            update(dict(gpu, state='waiting_for_gpu', shared_lock_owned=True))
        update(dict(shared_lock_owned=False))
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True); parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    runtime = read(ROOT / args.runtime_config); verify_receipts(runtime['source_receipts'])
    queue = read(ROOT / runtime['queue']); verify_receipts(queue['source_receipts']); validate_queue(queue)
    if args.validate_only:
        import sys
        print(dict(full_paper_case_seed_evaluations=30, original_paper_metric_targets=7,
            all_cases_and_full2000epoch_checkpoint_sources_verified=True,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules))
        return
    state_root = ROOT / runtime['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    state = dict(pid=process.pid, create_time=process.create_time(), active_pid=None,
                 required_case_seed_evaluations=30, completed_case_seed_evaluations=0, complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(state_root / 'status.json', state)
    api = queue_api(); lane = api['exclusive_lock'](ROOT / runtime['lane_lock'])
    if lane is None:
        raise RuntimeError('Existing full WT image evaluation controller owns the lane')
    completed = set()
    try:
        while len(completed) != len(queue['cases']):
            ready = None
            for case in queue['cases']:
                if case['id'] in completed:
                    continue
                if complete(case):
                    completed.add(case['id']); continue
                if (ROOT / case['checkpoint']).exists():
                    ready = case; break
            if ready is None:
                update(dict(state='waiting_for_full_original_training_checkpoints',
                    completed_case_seed_evaluations=len(completed), active_pid=None, shared_lock_owned=False))
                time.sleep(10); continue
            case = ready; output = ROOT / case['output_directory']
            candidate = output / 'score_candidate_location.json'
            audit = candidate.exists()
            module = 'run_wt_full_image_evaluation_v1'
            update(dict(case_id=case['id'], active_pid=None, stage='audit' if audit else 'score',
                        completed_case_seed_evaluations=len(completed)))
            settings = runtime['settings']
            with full_gpu_slot(api, runtime, update):
                attempt = output / 'attempts' / uuid.uuid4().hex; attempt.mkdir(parents=True, exist_ok=False)
                command = [runtime['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.' + module,
                    '--queue', runtime['queue'], '--case-id', case['id'], '--attempt-directory', str(attempt.resolve())]
                if audit:
                    command.append('--audit')
                with (attempt / 'stdout.log').open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                        env={**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                             'CUDA_VISIBLE_DEVICES': settings['gpu_uuid'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
                    identity = psutil.Process(child.pid)
                    update(dict(active_pid=child.pid, active_create_time=identity.create_time(),
                                command=command, state='running', shared_lock_owned=True))
                    code, usage = api['guarded_gpu_wait'](child, settings, api, update)
                terminal = dict(returncode=code, worker_pid=child.pid, worker_create_time=state['active_create_time'],
                                command=command, resource_usage=usage)
                write(attempt / 'terminal.json', terminal)
                update(dict(active_pid=None, last_terminal=terminal, shared_lock_owned=False))
                if code != 0 or usage['resource_guard_aborted']:
                    update(dict(state='failed_attempt_preserved', complete=False))
                    return
            # Score and replay take separate guarded reservations, preserving shared-queue fairness.
            if audit and not complete(case):
                raise ValueError('Independent full-native audit exited without complete proof')
        for case in queue['cases']:
            if not complete(case):
                raise ValueError('Every original full score/replay completion must remain verified')
        update(dict(state='all30_full_native_wt_score_and_replay_cases_verified', complete=True,
                    completed_case_seed_evaluations=30, active_pid=None, shared_lock_owned=False,
                    all_original_experiments_complete=False, original_WT_best_feature_setting_reproduced=False,
                    original_target_solver_and_step_identity_certified=False,solver_selected_using_test_labels=False))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
