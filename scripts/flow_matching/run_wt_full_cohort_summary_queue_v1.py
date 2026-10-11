"""Pure CPU summary/review controller; wait for all30 actual full-image proofs."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
import time
import uuid
import psutil
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, verify_receipts
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
from scripts.flow_matching.wt_full_cohort_summary_contract_v1 import (
    validate_configuration, score_locations_ready, complete_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True); parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(); runtime = read(ROOT / args.runtime_config)
    verify_receipts(runtime['source_receipts'])
    config = read(ROOT / runtime['summary_configuration']); queue = validate_configuration(config)
    if args.validate_only:
        print(dict(required_full_score_replay_cases=30, independent_training_seeds=5,
            aggregate_metric_rows=42, per_category_metric_rows=630, paired_metric_rows=84,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            native_performance_measured=False)); return
    api = resource_api(); state_root = ROOT / runtime['state_root']
    lane = api['exclusive_lock'](ROOT / runtime['lane_lock'])
    if lane is None: raise RuntimeError('Existing complete WT cohort summary controller owns the lane')
    process = psutil.Process()
    state = dict(pid=process.pid, create_time=process.create_time(), active_pid=None,
        required_full_verified_score_cases=30, complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(state_root / 'status.json', state)
    try:
        while not complete_summary(config):
            if not score_locations_ready(config, queue):
                update(dict(state='waiting_for_all30_full_native_score_and_replay_proofs', active_pid=None,
                    shared_lock_owned=False, available_score_proof_locations=sum(
                        (ROOT / case['output_directory'] / 'fully_verified_score_result.json').is_file() for case in queue['cases'])))
                time.sleep(10); continue
            output = ROOT / config['output_directory']; audit = (output / 'cohort_candidate_location.json').exists()
            with api['resource_slot'](ROOT / runtime['resource_lock'], runtime['settings'], update):
                attempt = output / 'attempts' / uuid.uuid4().hex; attempt.mkdir(parents=True, exist_ok=False)
                command = [runtime['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.summarize_wt_full_image_cohort_v1',
                    '--config', runtime['summary_configuration'], '--attempt-directory', str(attempt.resolve())]
                if audit: command.append('--audit')
                with (attempt / 'stdout.log').open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                        env={**os.environ, 'PYTHONUTF8':'1', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'CUDA_VISIBLE_DEVICES':'-1'})
                    identity = psutil.Process(child.pid)
                    update(dict(state='running', stage='independent_review' if audit else 'summary', active_pid=child.pid,
                        active_create_time=identity.create_time(), actual_command=command, shared_lock_owned=True))
                    code, usage = api['guarded_wait'](child, runtime['settings'], update)
                terminal = dict(returncode=code, worker_pid=child.pid, worker_create_time=state['active_create_time'],
                    command=command, resource_usage=usage)
                write(attempt / 'terminal.json', terminal)
                update(dict(active_pid=None, last_terminal=terminal, shared_lock_owned=False))
                if code != 0 or usage['resource_guard_aborted']:
                    update(dict(state='failed_attempt_preserved', complete=False)); return
            if audit and not complete_summary(config):
                raise ValueError('Independent full cohort review exited without complete proof')
        update(dict(state='all30_full_native_cases_and_five_seed_statistics_verified', complete=True,
            active_pid=None, shared_lock_owned=False, additional_training_seed_slots=0,
            all_original_experiments_complete=False, author_equivalence_certified=False))
    finally: lane.close()


if __name__ == '__main__':
    main()
