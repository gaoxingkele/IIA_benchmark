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
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def validate_queue(queue):
    protocol = read(ROOT / queue['original_protocol'])
    expected = {(key, index, seed)
        for key in ('Table6_compute', 'Table7_path_aggregation', 'Table8_conditional_mean', 'Table9_epoch_checkpoints', 'Table2_same_backbone_FM_comparison')
        for index, row in enumerate(protocol[key])
        if key != 'Table2_same_backbone_FM_comparison' or row['method'] == 'Flow Mismatching'
        for seed in range(1103, 1108)}
    actual = {(case['paper_table_key'], case['paper_row_index'], case['seed']) for case in queue['cases']}
    if actual != expected or len(actual) != len(queue['cases']) or len(actual) != 325:
        raise ValueError('Every original runnable paper table case and all five full training seeds required')
    for case in queue['cases']:
        row = protocol[case['paper_table_key']][case['paper_row_index']]
        parameters = case['score_parameters']
        if (row['paper_reported'] != case['paper_reported'] or row['dataset'] != case['original_dataset_name']
                or parameters['paths'] != row.get('K', 10) or parameters['times'] != row.get('T', 10)
                or parameters['aggregation'] != row.get('aggregation', 'minimum')
                or parameters['percentile'] != (row.get('percentile') or .1)
                or parameters['weighted'] is not True or parameters['top_fraction'] != .01
                or case['checkpoint_epoch'] != row.get('checkpoint_epoch', 2000)
                or (case['variant'] == 'conditional') != row.get('category_conditioning', True)):
            raise ValueError('Original complete image score recipe differs')
        training = read(ROOT / case['training_configuration'])
        if (training['seed'] != case['seed'] or training['dataset'] != case['dataset']
                or training['variant'] != case['variant'] or training['training']['epochs'] != 2000
                or case['checkpoint'] != training['output_directory'] + '/epoch_%04d.pt' % case['checkpoint_epoch']):
            raise ValueError('Full original-budget model/checkpoint identity differs')
    return True


def complete(case):
    output = ROOT / case['output_directory']
    path = output / 'fully_verified_score_result.json'
    if not path.exists():
        return False
    location = read(path)
    if sha(location['report']) != location['report_sha256']:
        raise ValueError('Full native independent audit changed')
    report = read(location['report'])
    if (not report['passed'] or report['case'] != case or not report['every_native_image_model_replayed']
            or not report['every_native_path_probe_seed_equal'] or not report['all_native_heatmaps_and_image_scores_equal']
            or not report['original_full_state_dictionary_loaded_strictly']
            or report['original_author_equivalence_certified'] or report['strict_TAB_result']):
        raise ValueError('Complete original image model replay and independent metric audit required')
    if sha(report['native_score_candidate']) != report['native_score_candidate_sha256']:
        raise ValueError('Native full original score candidate changed')
    candidate = read(report['native_score_candidate'])
    training = read(ROOT / case['training_configuration'])
    dataset = read(ROOT / training['dataset_configuration'])
    if (candidate['case'] != case or candidate['complete_original_test_images'] != dataset['expected_native_splits']['test']
            or report['complete_original_test_images'] != dataset['expected_native_splits']['test']
            or report['complete_original_categories'] != dataset['expected_original_category_count']
            or sha(ROOT / case['checkpoint']) != report['full_original_checkpoint_sha256']
            or report['queue_sha256'] != sha(ROOT / report['queue_configuration'])):
        raise ValueError('Every native original image/category, checkpoint and frozen evaluator required')
    verify_receipts(candidate['output_receipts'])
    return True


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
        print(dict(full_paper_case_seed_evaluations=325, original_paper_metric_targets=520,
            runnable_paper_metric_targets=485, remaining_comparator_targets=35,
            all_cases_and_full_budget_checkpoint_sources_verified=True,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules))
        return
    state_root = ROOT / runtime['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    state = dict(pid=process.pid, create_time=process.create_time(), active_pid=None,
                 required_case_seed_evaluations=325, completed_case_seed_evaluations=0, complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(state_root / 'status.json', state)
    api = queue_api(); lane = api['exclusive_lock'](ROOT / runtime['lane_lock'])
    if lane is None:
        raise RuntimeError('Existing full image evaluation controller owns the lane')
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
            module = ('audit_flow_mismatching_full_image_evaluation_v1' if candidate.exists()
                      else 'run_flow_mismatching_full_image_evaluation_v1')
            update(dict(case_id=case['id'], active_pid=None, stage=module,
                        completed_case_seed_evaluations=len(completed)))
            settings = runtime['settings']
            with full_gpu_slot(api, runtime, update):
                attempt = output / 'attempts' / uuid.uuid4().hex; attempt.mkdir(parents=True, exist_ok=False)
                command = [runtime['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.' + module,
                    '--queue', runtime['queue'], '--case-id', case['id'], '--attempt-directory', str(attempt.resolve())]
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
            if module.startswith('audit_') and not complete(case):
                raise ValueError('Independent full-native audit exited without complete proof')
        for case in queue['cases']:
            if not complete(case):
                raise ValueError('Every original full score/replay completion must remain verified')
        update(dict(state='all325_full_native_score_and_replay_cases_verified', complete=True,
                    completed_case_seed_evaluations=325, active_pid=None, shared_lock_owned=False,
                    all_original_experiments_complete=False, original_comparator_metric_targets_remaining=35))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
