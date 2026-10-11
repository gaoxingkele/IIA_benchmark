"""Publish one fully trained/replayed original seed without completing its five-seed cohort."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.verify_brits_electricity_full_replay_v3 import ROOT, read, digest, check_arrays

IDENTITY = ('saits', 'electricity', 'imputation', 'author', 'paper_mask', '0.2')
KEYS = ('algorithm', 'dataset', 'task', 'protocol', 'pattern', 'missing_ratio')
METRICS = ('mae', 'rmse', 'mre')
JOB = 'saits_paper_saits_electricity_author_missing0.2_seed26'


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def check_delta(before, after):
    if len(before) != len(after):
        raise ValueError('Keep every original metric slot')
    changed = []
    permitted = {'n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high', 'source',
        'captured_utc', 'comparison', 'uncertainty_scope', 'strict_TAB_result'}
    for i, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (tuple(new.get(k) for k in KEYS) != IDENTITY or new.get('metric') not in METRICS
                or any(old.get(k) != new.get(k) for k in (*KEYS, 'metric', 'required_repeats', 'paper_mean'))
                or str(old['n']) != '0' or str(new['n']) != '1'
                or str(old['required_repeats']) != '5'
                or any(old.get(k) != new.get(k) for k in old.keys() | new.keys() if k not in permitted)
                or any(new[k] != '' for k in ('std', 'se', 'ci95_low', 'ci95_high'))
                or str(new['author_equivalence_certified']).lower() != 'false'
                or str(new['strict_TAB_result']).lower() != 'false'
                or not math.isfinite(float(new['mean']))):
            raise ValueError('Only complete SAITS/Electricity20% seed26 metrics 0/5to1/5 may change')
        changed.append(i)
    if len(changed) != 3 or {after[i]['metric'] for i in changed} != set(METRICS):
        raise ValueError('Exactly all three original metrics must advance')
    return changed


def verify_registration(config):
    for pin in config['source_receipts']:
        if digest(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen native publication input differs: ' + pin['path'])
    proof, review = (read(ROOT / config[k]) for k in ('native_model_proof', 'independent_model_proof'))
    if (proof['full_models_replayed'] != 11 or not proof['full_checkpoint_contents_and_re_inference_passed']
            or not review['passed'] or review['full_native_models_verified'] != 11
            or review['native_replay_report_sha256'] != digest(ROOT / config['native_model_proof'])
            or config['original_training_seeds'] != [26] or config['required_training_seeds'] != [26,27,28,29,30]):
        raise ValueError('Actual complete full-model replay and independent review required')
    for receipt_path in config['native_guard_receipts']:
        receipt = read(ROOT / receipt_path)
        if receipt['exit_code'] != 0 or receipt['resource_guard_aborted']:
            raise ValueError('Successful unchanged native guards required')
        if not (receipt.get('full_native_model_replay_passed') or receipt.get('full_native_batch_independent_review_passed')):
            raise ValueError('Native full replay/review must pass')
    records = [r for r in proof['records'] if r['id'] == JOB]
    checks = [r for r in review['records'] if r['id'] == JOB]
    if len(records) != 1 or len(checks) != 1:
        raise ValueError('One actual completed canonical seed required')
    record, checked = records[0], checks[0]
    native = read(ROOT / record['config'])
    result = read(ROOT / record['result_path'])
    completion = record['full_training_completion']
    if (native['id'] != JOB or native['seed'] != 26 or native['missing_ratio'] != .2
            or native['method'] != 'saits' or native['dataset'] != 'electricity'
            or native['variant'] != 'author' or native['diagnostic'] or native['corrected_masking']
            or result['status'] != 'completed' or result['config'] != native
            or result['parameters'] != 11511000 or result['test_windows'] != 290
            or result['target_count'] != 1944331 or result['bootstrap_group'] != 'month'
            or result['bootstrap_groups'] != 10 or not completion['full_training_finished']
            or completion['benchmark_smoke'] or completion['original_maximum_epochs'] != 10000
            or (completion['next_epoch'] < 10000 and completion['author_patience_remaining'] != 0)
            or not checked['actual_full_checkpoint_reinference_completed']
            or not checked['full_truth_masks_units_groups_equal']
            or not checked['full_group_statistics_independently_recomputed']
            or proof['metric_tolerance'] != {'rtol': 2e-6, 'atol': 2e-6}):
        raise ValueError('Preserve full original architecture/data/earlystop/grouped evaluation')
    for path_key, hash_key in [('config','config_sha256'),('result_path','result_sha256'),
        ('checkpoint_path','checkpoint_sha256'),('replay_arrays','replay_arrays_sha256')]:
        if digest(ROOT / record[path_key]) != record[hash_key]:
            raise ValueError('Actual full native artifact changed')
    if digest(ROOT / native['dataset_root'] / 'datasets.h5') != record['original_full_dataset_sha256']:
        raise ValueError('Full original HDF5 differs')
    return record, native, result


def publish(path):
    config = read(ROOT / path)
    record, native, result = verify_registration(config)
    base = ROOT / config['base_publication']
    headers, before = csv_rows(base / 'all_algorithm_dataset_metrics.csv')
    if len(headers) != 31 or len(before) != 8626 or sum(bool(r['mean']) for r in before) != 2128:
        raise ValueError('Keep full existing8626slots/31fields/2128measured rows')
    rows = copy.deepcopy(before)
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if tuple(row[k] for k in KEYS) == IDENTITY and row['metric'] in METRICS:
            row.update(n='1', mean=str(result['metrics'][row['metric']]), std='', se='',
                ci95_low='', ci95_high='', source=config['cohort_summary_path'], captured_utc=now,
                comparison='one_of_five_full_native_models_replayed_author_alignment_unverified',
                strict_TAB_result='False', uncertainty_scope=
                'One complete original training seed26 of5. Across-seed sampleSTD/SE/CI undefined. Ten-month block bootstrap is within-seed temporal sensitivity only and is recorded separately; epochs/windows/held-out points/retries are not additional seeds.')
    changed = check_delta(before, rows)
    target = ROOT / config['output_directory']
    target.mkdir(parents=True, exist_ok=False)
    exclusive(ROOT / config['cohort_summary_path'], dict(captured_utc=now, canonical_id=JOB,
        original_training_seeds=[26], required_original_training_seeds=[26,27,28,29,30],
        full_trained_and_replayed_original_models=1, full_five_seed_cohort_complete=False,
        original_metrics=result['metrics'], metric_rows=[rows[i] for i in changed],
        original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],
        month_block_bootstrap_scope='Within one trained seed; not cross-seed robustness or independent months guarantee',
        original_protocol_boundaries=result['protocol_boundaries'], config=native,
        native_model_proof=config['native_model_proof'], independent_model_proof=config['independent_model_proof'],
        full_original_scope=config['original_scope'], additional_training_seed_slots=0,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    with (target / 'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers); writer.writeheader(); writer.writerows(rows)
    from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy
    from scripts.flow_matching.publish_pi_original_source_listing_v42 import interactive_table
    boundary = ('Only three SAITS/Electricity original20% missing-rate metrics advance from0/5to1/5 original seeds. '
        'All8623other rows and timestamps remain unchanged. Seed26 finished unchanged10000epoch-cap/author early stopping '
        'at418epochs and the validation-selected11511000-parameter checkpoint was fully replayed on all290test windows '
        'and1944331held-out coordinates, with independent original HDF5/mask/unit/month verification. '
        'Across-seed sampleSTD, SE and CI remain undefined. Saved ten-month block-bootstrap intervals are within-seed '
        'temporal sensitivity only. Historical preprocessing/per-rate tuning/model-source alignment is unverified; '
        'no paper or TAB equivalence. All45papers/681method-baseline-ablation/273original-data-task obligations remain.')
    (target / 'all_metrics.md').write_text('# Algorithm-dataset metrics v50\n\n' + boundary + '\n\n' +
        legacy.table(headers, [[r.get(k) or '-' for k in headers] for r in rows]) + '\n', encoding='utf-8')
    page = interactive_table(rows, headers).replace('v42', 'v50')
    import re
    page = re.sub(r'<p>[^<]*</p>(?=<input)', '<p>' + boundary + '</p>', page)
    (target / 'results.html').write_text(page, encoding='utf-8')
    (target / 'README.rst').write_text('Algorithm-dataset metrics v50\n============================\n\n' +
        boundary + '\n\n8626metric rows;2131measured;6495pending. This SAITS cohort remains1/5seeds.\n', encoding='utf-8')
    exclusive(target / 'publication_validation.json', dict(published_utc=now,
        base_publication=config['base_publication'], base_receipt_sha256=digest(base / 'publication_validation.json'),
        aggregate_metric_rows=8626, fields=31, measured_metric_rows=2131, missing_metric_rows=6495,
        changed_metric_rows=3, newly_measured_metric_rows=3, changed_positions=changed,
        other8623rows_and_capture_times_unchanged=True, original_full_seed_count=1, required_original_seeds=5,
        full_original_seed_cohort_complete=False, additional_training_seed_slots=0,
        all_original_experiments_complete=False, author_equivalence_certified=False, strict_TAB_result=False,
        original_scope=config['original_scope'], source_receipts=config['source_receipts'],
        outputs={p.name:digest(p) for p in target.iterdir() if p.is_file()}))


def review(path):
    import h5py
    import numpy as np
    from scripts.flow_matching.run_saits import patient_bootstrap
    config = read(ROOT / path)
    record, native, result = verify_registration(config)
    base, target = (ROOT / config[k] for k in ('base_publication','output_directory'))
    old_headers, before = csv_rows(base / 'all_algorithm_dataset_metrics.csv')
    headers, rows = csv_rows(target / 'all_algorithm_dataset_metrics.csv')
    if headers != old_headers or len(headers) != 31 or len(rows) != 8626:
        raise ValueError('Preserve complete original metric schema and slots')
    changed = check_delta(before, rows)
    receipt = read(target / 'publication_validation.json')
    for name, value in receipt['outputs'].items():
        if digest(target / name) != value:
            raise ValueError('Published artifact changed')
    data = ROOT / native['dataset_root']
    with h5py.File(data / 'datasets.h5','r') as hf:
        truth, mask = hf['test']['X'][:], hf['test']['indicating_mask'][:]
    with np.load(data / 'audit_arrays.npz', allow_pickle=False) as a:
        ids, groups = a['test_ids'], a['test_groups']
    with np.load(record['replay_arrays'], allow_pickle=False) as a:
        actual, grouped = check_arrays(a, truth, mask, ids, groups, 1944331, dict(rtol=2e-6,atol=2e-6))
    with np.load(ROOT / native['output_root'] / 'group_statistics.npz', allow_pickle=False) as a:
        if not np.allclose(grouped, a['statistics'], rtol=2e-6, atol=2e-6):
            raise ValueError('Full original monthly statistics differ')
    bootstrap = patient_bootstrap(grouped, 10026)
    for name in METRICS:
        if not math.isclose(actual[name], result['metrics'][name], rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError('Full original metric differs')
        if not np.allclose(bootstrap[name], result['month_block_bootstrap_95_ci'][name], rtol=2e-6,atol=2e-6):
            raise ValueError('Independent full grouped bootstrap differs')
    for i in changed:
        if float(rows[i]['mean']) != result['metrics'][rows[i]['metric']]:
            raise ValueError('Published metric must preserve the original complete training result')
    if receipt['measured_metric_rows'] != sum(bool(r['mean']) for r in rows) or receipt['full_original_seed_cohort_complete']:
        raise ValueError('Do not complete a one-of-five cohort')
    exclusive(target / 'independent_publication_review.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, original_rows=8626, fields=31, changed_rows=3, other8623rows_and_capture_times_unchanged=True,
        full_test_shape=list(truth.shape), held_out_coordinates=1944331, month_groups=len(grouped),
        full_native_HDF5_truth_mask_unit_month_identity_and_metrics_verified=True,
        independent_metrics=actual, independent_month_block_bootstrap_95_ci=bootstrap,
        across_seed_uncertainty_remains_undefined=True, original_training_seeds=[26], required_original_seeds=5,
        full_original_seed_cohort_complete=False, author_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False, additional_training_seed_slots=0,
        publication_receipt_sha256=digest(target / 'publication_validation.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', choices=['publish','review'])
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify_registration(config)
    if args.probe_only:
        print(json.dumps(dict(full_original_seed=26, required_original_seeds=5, original_metric_rows=8626,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    if args.worker:
        return (publish if args.worker == 'publish' else review)(args.config)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
    api = resource_api()
    state_root = ROOT / config['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](state_root / 'controller.lock')
    if owner is None:
        raise RuntimeError('Publication pipeline already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat());write(state_root/'status.json',state)
    try:
        for phase in ('publish','review'):
            with api['resource_slot'](ROOT / config['resource_lock'],config['settings'],update):
                command = [config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.publish_saits_electricity20_seed26_v50','--config',args.config,'--worker',phase]
                with (state_root/(phase+'.stdout.log')).open('xb') as out, (state_root/(phase+'.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                            PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    p=psutil.Process(child.pid);update(dict(state='running_'+phase,active_pid=p.pid,
                        active_create_time=p.create_time(),actual_command=p.cmdline()))
                    code, usage = capped_wait(child,config,api,update)
            exclusive(state_root/(phase+'_execution_receipt.json'),dict(exit_code=code,**usage))
            if code != 0 or usage['resource_guard_aborted']:
                update(dict(state='failed_preserved',active_pid=None));return
            update(dict(state='between_phases',active_pid=None))
        update(dict(state='completed',active_pid=None,newly_measured_metric_rows=3,
            full_original_seed_cohort_complete=False,all_original_experiments_complete=False))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
