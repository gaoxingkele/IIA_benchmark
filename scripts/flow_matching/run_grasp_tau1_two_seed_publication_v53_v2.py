"""Publish and independently review the two completed original tau1 SWAN seeds."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import subprocess

import psutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def verify_registration(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen original two-seed publication source changed')
    producer = read(ROOT/config['producer_configuration'])
    for key in ('python', 'cpu_threads', 'settings', 'resource_lock', 'score_tolerance', 'metric_tolerance'):
        if config[key] != producer[key]:
            raise ValueError('Keep every original native admission, emergency and numerical guard')
    receipt = read(ROOT/config['completed_pipeline_receipt'])
    if receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['complete_native_and_independent_review_passed']:
        raise ValueError('Actual complete original native replay and independent review must finish first')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        values = list(reader)
        return reader.fieldnames, values


def independent_publication_review(config):
    """Check all slots and independently compute n=2 Student-t uncertainty."""
    before_headers, before = rows(ROOT/config['base_publication']/'all_algorithm_dataset_metrics.csv')
    target = ROOT/config['output_directory']
    headers, after = rows(target/'all_algorithm_dataset_metrics.csv')
    if headers != before_headers or len(headers) != 31 or len(before) != len(after) or len(after) != 8626:
        raise ValueError('Preserve every original metric slot and field')
    protocol_fields = {'paper_test_label_oracle_no_PA': ('paper_retrospective', ('AP', 'ROC', 'Best_F1')),
        'validation_only_1_percent_no_PA_control': ('validation_only_control', ('precision', 'recall', 'f1'))}
    results = [read(ROOT/path) for path in config['original_seed_results']]
    if [r['job']['seed'] for r in results] != [1103, 1104]:
        raise ValueError('Two distinct original seeds required')
    changed = []
    permitted = {'n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high', 'captured_utc', 'source', 'comparison', 'uncertainty_scope'}
    combinations = set()
    for i, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        field, names = protocol_fields[new['protocol']]
        if (new['algorithm'] != 'GRASP/tau_1' or new['dataset'] != 'swan'
                or new['metric'] not in names or old['n'] != '1' or new['n'] != '2'
                or new['required_repeats'] != '10' or new['source'] != config['cohort_summary_path']
                or old.keys() != new.keys()
                or any(old[k] != new[k] for k in old if k not in permitted)
                or new['author_equivalence_certified'].lower() != 'false'
                or new['strict_TAB_result'].lower() not in ('', 'false')):
            raise ValueError('Only six unchanged original partial tau1 seed metrics may advance')
        values = [r['metrics'][0]['metrics'][field][new['metric']] for r in results]
        average = (values[0] + values[1]) / 2
        sample_std = abs(values[1] - values[0]) / math.sqrt(2)
        se = abs(values[1] - values[0]) / 2
        # Student-t(df=1) is a Cauchy distribution; two-sided 95% quantile.
        halfwidth = math.tan(math.pi * (.975 - .5)) * se
        expected = dict(mean=average, std=sample_std, se=se,
            ci95_low=average-halfwidth, ci95_high=average+halfwidth)
        if not math.isclose(float(old['mean']), values[0], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Existing first-seed result must remain the same original experiment')
        for key, value in expected.items():
            if not math.isfinite(float(new[key])) or not math.isclose(float(new[key]), value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('Independent complete original seed statistics differ: ' + key)
        combinations.add((new['protocol'], new['metric']))
        changed.append(i)
    expected_combinations = {(protocol, metric) for protocol, (_, metrics) in protocol_fields.items() for metric in metrics}
    if len(changed) != 6 or combinations != expected_combinations or sum(bool(r['mean']) for r in after) != 2131:
        raise ValueError('Keep the full 8626-row/2131-measured scope and both separate protocols')
    exclusive_json(target/'independent_publication_review.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), passed=True, original_rows=8626, fields=31,
        changed_rows=6, changed_positions=changed, other8620rows_and_capture_times_unchanged=True,
        measured_rows=2131, pending_rows=6495, original_seed_count=2, required_original_seeds=10,
        original_seed_means_sample_std_se_and_unclipped_closed_form_student_t95CI_verified=True,
        original_scope=config['original_scope'], additional_training_seed_slots=0,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        reviewed_csv_sha256=sha(target/'all_algorithm_dataset_metrics.csv')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    verify_registration(config)
    if args.worker:
        from scripts.flow_matching.publish_grasp_tau1_two_seed_listing_v53_v2 import publish
        publish(args.config)
        independent_publication_review(config)
        return
    api = resource_api()
    base = ROOT/config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('This exact original-seed publication already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        configuration_sha256=sha(ROOT/args.config), state='starting', active_pid=None,
        all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json', state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                'scripts.flow_matching.run_grasp_tau1_two_seed_publication_v53_v2', '--config', args.config, '--worker']
            with (base/'worker.stdout.log').open('xb') as out, (base/'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            handle = psutil.Process(child.pid)
            update(dict(state='independently_rechecking_and_publishing', active_pid=child.pid,
                active_create_time=handle.create_time(), actual_command=handle.cmdline()))
            code, usage = capped_wait(child, config, api, update)
        proof_path = ROOT/config['output_directory']/'independent_publication_review.json'
        passed = bool(code == 0 and not usage['resource_guard_aborted'] and proof_path.exists() and read(proof_path)['passed'])
        write(base/'receipt.json', dict(exit_code=code, complete_independent_publication_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
