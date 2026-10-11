"""Independently review completed tau=0.5 native replay and publish its partial seed pair."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import math
import os
import statistics
import subprocess

import psutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_grasp_tau025_postreplay_pipeline_v1 import register_completed_review
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
from scripts.flow_matching.verify_grasp_completed_full_replay_v1 import check_terminal

CANONICAL = 'grasp_protocol_v3__tau_0p5__swan__swan_sf__seed1104'


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen tau0.5 postreview evidence changed')
    producer = read(ROOT/config['producer_configuration'])
    if (any(config[k] != producer[k] for k in ('settings', 'resource_lock', 'python'))
            or producer['full_job_count'] != 1 or producer['cpu_threads'] != 2
            or producer['full_results'][0]['canonical_id'] != CANONICAL):
        raise ValueError('Keep the original environment, resource guards and complete seed1104 producer')
    union = read(ROOT/config['preserved_thirteen_model_union'])
    if (union['full_original_models'] != 13
            or not union['all_complete_arrays_point_mapping_and_metrics_independently_verified']
            or not union['original_full_native_worker_exit_success_verified']
            or not union['no_new_training_seed_slots']):
        raise ValueError('The first seed requires the preserved complete original native replay union')
    first = [r for r in union['records'] if r['canonical_id'] == CANONICAL.replace('1104', '1103')]
    if (len(first) != 1 or len(first[0]['recipes']) != 1
            or first[0]['recipes'][0]['recipe'] != dict(source_samples=5, flow_evaluations=10)
            or not first[0]['recipes'][0]['passed']):
        raise ValueError('Complete first original seed and unchanged five-source/ten-flow recipe required')
    check_terminal(read(ROOT/producer['output_directory']/'checkpoint_replay_audit.json'),
                   read(ROOT/producer['state_root']/'receipt.json'), producer)
    return producer


def register_publication(config, producer):
    from scripts.flow_matching.publish_grasp_tau05_two_seed_listing_v47 import publish
    proof = config['second_seed_review_directory']+'/independent_full_replay_verification.json'
    sources = [config['preserved_thirteen_model_union'], proof, config['second_seed_review_configuration'],
        config['base_publication']+'/all_algorithm_dataset_metrics.csv',
        config['base_publication']+'/publication_validation.json', producer['queue'],
        'scripts/flow_matching/publish_grasp_tau05_two_seed_listing_v47.py',
        'scripts/flow_matching/audit_grasp_complete_replay_union_v1.py',
        'scripts/flow_matching/audit_grasp_checkpoint_replay_v1.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'scripts/flow_matching/publish_pi_original_source_listing_v42.py',
        'scripts/flow_matching/publish_brits_two_seed_listing_v27.py',
        'tests/test_fm_grasp_tau05_two_seed_listing_v47.py']
    publication = dict(base_publication=config['base_publication'], grasp_queue=producer['queue'],
        preserved_thirteen_model_union=config['preserved_thirteen_model_union'],
        completed_new_seed_proofs={'tau_0p5': proof}, score_tolerance=producer['score_tolerance'],
        metric_tolerance=producer['metric_tolerance'], output_directory=config['publication_directory'],
        cohort_summary_path=config['cohort_summary_path'], original_scope=config['original_scope'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in dict.fromkeys(sources)])
    exclusive_json(ROOT/config['publication_configuration'], publication)
    publish(config['publication_configuration'])


def independent_publication_review(config):
    from scripts.flow_matching.publish_grasp_tau05_two_seed_listing_v47 import check_delta
    with (ROOT/config['base_publication']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig') as stream:
        before = list(csv.DictReader(stream))
    with (ROOT/config['publication_directory']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        after, fields = list(reader), reader.fieldnames
    if len(before) != 8626 or len(after) != 8626 or len(fields) != 31:
        raise ValueError('Keep every original metric row and field')
    changed = check_delta(before, after)
    summary = read(ROOT/config['cohort_summary_path'])
    checks = []
    for item in summary['metric_statistics']:
        field = 'paper_retrospective' if item['protocol'] == 'paper_test_label_oracle_no_PA' else 'validation_only_control'
        values = [read(ROOT/'experiments/runs/fm_grasp_protocol_v3/jobs'/
            CANONICAL.replace('1104', str(seed))/'result.json')['metrics'][0]['metrics'][field][item['metric']]
            for seed in (1103, 1104)]
        if item['original_seed_values'] != values or item['n'] != 2:
            raise ValueError('Two unique complete original training seeds required')
        avg = statistics.mean(values)
        std = statistics.stdev(values)
        se = std/math.sqrt(2)
        if any(not math.isclose(item[k], v, rel_tol=1e-12, abs_tol=1e-12)
               for k, v in [('mean', avg), ('std', std), ('se', se)]):
            raise ValueError('Independent seed mean/sampleSTD/SE differ')
        if se:
            q = (item['ci95_high']-avg)/se
            if (abs(.5+math.atan(q)/math.pi-.975) > 1e-12
                    or not math.isclose(item['ci95_low'], avg-q*se, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Independent df1 Student-t confidence interval differs')
        elif item['ci95_low'] != avg or item['ci95_high'] != avg:
            raise ValueError('Zero-variance confidence interval differs')
        row = next(r for r in after if r['algorithm'] == 'GRASP/tau_0p5' and r['dataset'] == 'swan'
                   and r['protocol'] == item['protocol'] and r['metric'] == item['metric'])
        if any(float(row[k]) != item[k] for k in ('n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high')):
            raise ValueError('Publication must preserve independently verified statistics')
        checks.append(dict(metric=item['metric'], protocol=item['protocol'],
                           complete_original_seed_values_and_closed_form_CI_independently_verified=True))
    exclusive_json(ROOT/config['publication_directory']/'independent_publication_review.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), passed=True, original_rows=8626,
        fields=31, changed_rows=len(changed), other8620rows_and_capture_times_unchanged=True,
        measured_rows=sum(bool(r['mean']) for r in after), pending_rows=sum(not bool(r['mean']) for r in after),
        statistics=checks, original_seed_count=2, required_original_seeds=10,
        additional_training_seed_slots=0, all_original_experiments_complete=False))


def worker(config, path):
    producer = verify_registration(config)
    register_completed_review(config, producer)
    register_publication(config, producer)
    independent_publication_review(config)
    verify_registration(config)
    exclusive_json(ROOT/config['output_directory']/'postreplay_pipeline_completed.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        configuration=path, configuration_sha256=sha(ROOT/path),
        both_original_native_full_model_replays_and_independent_reviews_verified=True,
        six_existing_numeric_rows_updated_to_two_original_seeds=True,
        required_original_training_seeds=10, completed_original_training_seeds=2,
        additional_model_inference_executed_by_this_postreview=False, additional_training_seed_slots=0,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    verify_registration(config)
    if args.probe_only:
        print(dict(full_native_replay_terminal_verified=True, original_seeds=[1103, 1104],
                   required_original_seeds=10, new_training_seeds=0))
        return
    if args.worker:
        worker(config, args.config)
        return
    api = resource_api()
    base = ROOT/config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('The exact complete tau0.5 postreview controller is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json', state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.run_grasp_tau05_postreplay_pipeline_v1', '--config', args.config, '--worker']
            with (base/'worker.stdout.log').open('xb') as out, (base/'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=dict(os.environ,
                    PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                    CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT/config['output_directory']/'postreplay_pipeline_completed.json'
        passed = bool(code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed'])
        write(base/'receipt.json', dict(exit_code=code, complete_postreplay_publication_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None))
        if not passed:
            raise RuntimeError('Preserve failed complete native postreview evidence')
    finally:
        lane.close()


if __name__ == '__main__':
    main()
