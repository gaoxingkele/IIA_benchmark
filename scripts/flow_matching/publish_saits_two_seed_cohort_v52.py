"""Publish two fully replayed canonical seeds with existing Student-t uncertainty."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
import subprocess

import psutil

from scripts.flow_matching.publish_saits_electricity20_seed26_v50 import (
    ROOT, read, digest, exclusive, csv_rows, KEYS, IDENTITY, METRICS,
)


def check_delta(before, after):
    if len(before) != len(after):
        raise ValueError('Every original metric slot must remain')
    permitted = {'n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high', 'source',
                 'captured_utc', 'comparison', 'uncertainty_scope'}
    changed = []
    for i, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (tuple(new.get(k) for k in KEYS) != IDENTITY or new.get('metric') not in METRICS
                or old.get('n') != '1' or new.get('n') != '2'
                or old.get('required_repeats') != '5' or new.get('required_repeats') != '5'
                or old.keys() != new.keys()
                or any(old[k] != new[k] for k in old if k not in permitted)
                or new.get('author_equivalence_certified', '').lower() != 'false'
                or new.get('strict_TAB_result', '').lower() != 'false'
                or any(not math.isfinite(float(new[k])) for k in
                       ('mean', 'std', 'se', 'ci95_low', 'ci95_high'))
                or float(new['std']) < 0 or float(new['se']) < 0
                or not float(new['ci95_low']) <= float(new['mean']) <= float(new['ci95_high'])):
            raise ValueError('Only three unchanged original SAITS slots may advance from1/5to2/5')
        changed.append(i)
    if len(changed) != 3 or {after[i]['metric'] for i in changed} != set(METRICS):
        raise ValueError('Exactly all three original metrics required')
    return changed


def check_record(record, checked, native, result, seed):
    wanted = 'saits_paper_saits_electricity_author_missing0.2_seed%d' % seed
    completion = record['full_training_completion']
    if (record['id'] != wanted or checked['id'] != wanted or native['id'] != wanted
            or native['seed'] != seed or record['seed'] != seed or checked['seed'] != seed
            or native['method'] != 'saits' or native['dataset'] != 'electricity'
            or native['missing_ratio'] != .2 or native['variant'] != 'author'
            or native['corrected_masking'] or native['diagnostic']
            or native.get('limit_test_cases') or result['config'] != native
            or result['status'] != 'completed' or result['parameters'] != 11511000
            or record['model_parameters'] != 11511000 or result['test_windows'] != 290
            or record['full_test_shape'] != [290, 100, 370]
            or checked['full_test_shape'] != [290, 100, 370]
            or result['target_count'] != 1944331 or record['held_out_coordinates'] != 1944331
            or checked['held_out_coordinates'] != 1944331 or checked['months'] != 10
            or result['bootstrap_groups'] != 10 or result['bootstrap_group'] != 'month'
            or not completion['full_training_finished'] or completion['benchmark_smoke']
            or completion['original_maximum_epochs'] != 10000
            or (completion['next_epoch'] < 10000 and completion['author_patience_remaining'] != 0)
            or not record['full_original_checkpoint_and_data_identity_verified']
            or not record['full_test_grid_and_grouped_statistics_replayed']
            or not checked['actual_full_checkpoint_reinference_completed']
            or not checked['full_truth_masks_units_groups_equal']
            or not checked['full_group_statistics_independently_recomputed']
            or record['original_metrics'] != result['metrics']):
        raise ValueError('Complete original canonical model, data, budget and independent replay required')
    for metric in METRICS:
        for values in (record['replayed_metrics'], record['independently_recomputed_metrics'],
                       checked['independently_recomputed_metrics']):
            if not math.isfinite(values[metric]) or not math.isclose(
                    values[metric], result['metrics'][metric], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError('Every full original metric must independently agree')


def verify_registration(config):
    if (config['original_training_seeds'] != [26, 27]
            or config['required_training_seeds'] != [26, 27, 28, 29, 30]
            or len(config['model_evidence']) != 2):
        raise ValueError('Keep the actual two canonical seeds and full required five-seed cohort')
    for pin in config['source_receipts']:
        if digest(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen original two-seed publication input changed: ' + pin['path'])
    for path in config['native_guard_receipts']:
        receipt = read(ROOT / path)
        if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
                or not (receipt.get('full_native_model_replay_passed')
                        or receipt.get('full_native_batch_independent_review_passed'))):
            raise ValueError('All actual original native replay/review guards must pass')
    records = []
    for seed, paths in zip(config['original_training_seeds'], config['model_evidence']):
        proof, review = read(ROOT / paths['native']), read(ROOT / paths['independent'])
        if (not proof['full_checkpoint_contents_and_re_inference_passed']
                or proof['full_models_replayed'] != len(proof['records'])
                or len({r['id'] for r in proof['records']}) != len(proof['records'])
                or proof['cuda_context_initialized'] or proof['additional_training_seed_slots'] != 0
                or proof['metric_tolerance'] != dict(rtol=2e-6, atol=2e-6)
                or not review['passed'] or review['full_native_models_verified'] != len(proof['records'])
                or review['native_replay_report_sha256'] != digest(ROOT / paths['native'])
                or not review['all_original_grid_coordinates_independently_checked']):
            raise ValueError('Actual complete native batches and their independent reviews required')
        wanted = 'saits_paper_saits_electricity_author_missing0.2_seed%d' % seed
        matches = [r for r in proof['records'] if r['id'] == wanted]
        checks = [r for r in review['records'] if r['id'] == wanted]
        if len(matches) != 1 or len(checks) != 1:
            raise ValueError('Exactly one native model and independent review per original seed')
        record, checked = matches[0], checks[0]
        native, result = read(ROOT / record['config']), read(ROOT / record['result_path'])
        check_record(record, checked, native, result, seed)
        for path_key, hash_key in [('config', 'config_sha256'), ('result_path', 'result_sha256'),
                                  ('checkpoint_path', 'checkpoint_sha256'), ('replay_arrays', 'replay_arrays_sha256')]:
            if digest(ROOT / record[path_key]) != record[hash_key]:
                raise ValueError('Actual canonical full native artifact changed')
        if digest(ROOT / native['dataset_root'] / 'datasets.h5') != record['original_full_dataset_sha256']:
            raise ValueError('Retain the complete frozen native HDF5')
        records.append((record, native, result))
    if len(records) != 2 or len({r[0]['original_full_dataset_sha256'] for r in records}) != 1:
        raise ValueError('Two unique original seeds sharing the exact frozen full split required')
    return records


def publish(path):
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
    from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy
    from scripts.flow_matching.repair_saits_publication_display_v51 import render
    config = read(ROOT / path); records = verify_registration(config)
    base = ROOT / config['base_publication']; fields, before = csv_rows(base / 'all_algorithm_dataset_metrics.csv')
    if len(fields) != 31 or len(before) != 8626 or sum(bool(r['mean']) for r in before) != 2131:
        raise ValueError('Keep all original measured and pending slots')
    base_review = read(base / 'independent_publication_review.json')
    if not base_review['passed'] or base_review['publication_receipt_sha256'] != digest(base / 'publication_validation.json'):
        raise ValueError('Require successful preceding full publication review')
    summaries = {m: independent_stats([r[2]['metrics'][m] for r in records]) for m in METRICS}
    rows = copy.deepcopy(before); captured = datetime.now(timezone.utc).isoformat()
    uncertainty = ('Two original training seeds26/27 on one frozen split; sampleSTD(ddof1),SE and existing '
        'two-sided95%Student-t CI(df1). Normal independent seed-error assumption is unverified with n2; '
        'wide untruncated intervals are not five-seed or cross-dataset robustness. Ten-month bootstrap '
        'is a separate within-seed temporal sensitivity; windows/epochs/replays are not training seeds.')
    for row in rows:
        if tuple(row[k] for k in KEYS) == IDENTITY and row['metric'] in METRICS:
            row.update({k: str(v) for k, v in summaries[row['metric']].items()})
            row.update(source=config['cohort_summary_path'], captured_utc=captured,
                comparison='two_of_five_full_native_models_replayed_author_alignment_unverified',
                uncertainty_scope=uncertainty)
    changed = check_delta(before, rows)
    target = ROOT / config['output_directory']; target.mkdir(parents=True, exist_ok=False)
    exclusive(ROOT / config['cohort_summary_path'], dict(captured_utc=captured,
        original_training_seeds=[26, 27], required_original_training_seeds=[26, 27, 28, 29, 30],
        full_trained_and_replayed_original_models=2, full_five_seed_cohort_complete=False,
        summaries=summaries, uncertainty_scope=uncertainty,
        per_seed_original_metrics={str(r[1]['seed']): r[2]['metrics'] for r in records},
        per_seed_month_block_bootstrap={str(r[1]['seed']): r[2]['month_block_bootstrap_95_ci'] for r in records},
        protocol_boundaries=records[0][2]['protocol_boundaries'], model_evidence=config['model_evidence'],
        metric_rows=[rows[i] for i in changed], full_original_scope=config['original_scope'],
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    with (target / 'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    boundary = ('SAITS Electricity20%已完整核验原始seed26/27，当前2/5个训练种子。仅三项指标更新两种子均值、样本标准差和Student-t区间；'
        '其余8623条记录保持原样。n=2的区间较宽，独立正态种子误差假设尚未验证；月份bootstrap只描述单种子的时间敏感性。'
        '论文历史条件对齐与严格TAB等价性未认证；45论文/681方法与消融/273原始数据任务的完整目标仍待完成。')
    (target / 'results.html').write_text(render(rows, fields, 52, boundary), encoding='utf-8')
    (target / 'all_metrics.md').write_text('# Algorithm-dataset metrics v52\n\n' + boundary + '\n\n' +
        legacy.table(fields, [[r.get(k) or '-' for k in fields] for r in rows]) + '\n', encoding='utf-8')
    (target / 'README.rst').write_text('Algorithm-dataset metrics v52\n============================\n\n' +
        boundary + '\n\n' + uncertainty + '\n', encoding='utf-8')
    exclusive(target / 'publication_validation.json', dict(published_utc=captured,
        aggregate_metric_rows=8626, fields=31, measured_metric_rows=2131, missing_metric_rows=6495,
        changed_metric_rows=3, newly_measured_metric_rows=0, changed_positions=changed,
        other8623rows_and_capture_times_unchanged=True, original_full_seed_count=2, required_original_seeds=5,
        full_original_seed_cohort_complete=False, additional_training_seed_slots=0,
        original_scope=config['original_scope'], author_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False, source_receipts=config['source_receipts'],
        cohort_summary_sha256=digest(ROOT / config['cohort_summary_path']),
        outputs={p.name: digest(p) for p in target.iterdir() if p.is_file()}))


def review(path):
    import h5py
    import numpy as np
    from scripts.flow_matching.run_saits import patient_bootstrap
    from scripts.flow_matching.verify_brits_electricity_full_replay_v3 import check_arrays
    config = read(ROOT / path); records = verify_registration(config)
    base, target = (ROOT / config[k] for k in ('base_publication', 'output_directory'))
    old_fields, before = csv_rows(base / 'all_algorithm_dataset_metrics.csv')
    fields, rows = csv_rows(target / 'all_algorithm_dataset_metrics.csv')
    if fields != old_fields or len(fields) != 31 or len(rows) != 8626:
        raise ValueError('The full original metric schema and scope are required')
    changed = check_delta(before, rows); receipt = read(target / 'publication_validation.json')
    if digest(ROOT / config['cohort_summary_path']) != receipt['cohort_summary_sha256']:
        raise ValueError('Actual complete original-seed cohort summary changed')
    for name, value in receipt['outputs'].items():
        if digest(target / name) != value:
            raise ValueError('Actual published artifact differs')
    with h5py.File(ROOT / records[0][1]['dataset_root'] / 'datasets.h5', 'r') as hf:
        truth, mask = hf['test']['X'][:], hf['test']['indicating_mask'][:]
    with np.load(ROOT / records[0][1]['dataset_root'] / 'audit_arrays.npz', allow_pickle=False) as a:
        ids, groups = a['test_ids'], a['test_groups']
    verified = []
    for record, native, result in records:
        with np.load(record['replay_arrays'], allow_pickle=False) as a:
            actual, grouped = check_arrays(a, truth, mask, ids, groups, 1944331, dict(rtol=2e-6, atol=2e-6))
        with np.load(ROOT / native['output_root'] / 'group_statistics.npz', allow_pickle=False) as a:
            if not np.allclose(grouped, a['statistics'], rtol=2e-6, atol=2e-6):
                raise ValueError('Every complete original month statistic must agree')
        bootstrap = patient_bootstrap(grouped, 10000 + native['seed'])
        for m in METRICS:
            if (not math.isclose(actual[m], result['metrics'][m], rel_tol=2e-6, abs_tol=2e-6)
                    or not np.allclose(bootstrap[m], result['month_block_bootstrap_95_ci'][m], rtol=2e-6, atol=2e-6)):
                raise ValueError('Independent full native metric/month bootstrap differs')
        verified.append(dict(seed=native['seed'], metrics=actual, month_bootstrap=bootstrap))
    # Independent two-sample closed form: df1 Student-t is the Cauchy distribution.
    # This checks the existing scipy estimator without reusing its implementation.
    summaries = {}
    for m in METRICS:
        a, b = [r[2]['metrics'][m] for r in records]
        mean = (a + b) / 2; sd = abs(a - b) / math.sqrt(2); se = abs(a - b) / 2
        margin = math.tan(.475 * math.pi) * se
        summaries[m] = dict(mean=mean, std=sd, se=se, ci95_low=mean-margin, ci95_high=mean+margin)
    for i in changed:
        for k, value in summaries[rows[i]['metric']].items():
            if not math.isclose(float(rows[i][k]), value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Original-seed summary or untruncated df1 uncertainty differs')
    page = (target / 'results.html').read_text(encoding='utf-8')
    if json.loads(page.split('<script>const rows=', 1)[1].split(', cols=', 1)[0]) != rows:
        raise ValueError('All embedded values and actual source identities must match the complete CSV')
    if (receipt['original_full_seed_count'] != 2 or receipt['full_original_seed_cohort_complete']
            or receipt['newly_measured_metric_rows'] != 0 or sum(bool(r['mean']) for r in rows) != 2131):
        raise ValueError('Two of five is incomplete and advances existing measured slots only')
    exclusive(target / 'independent_publication_review.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), original_rows=8626, fields=31,
        changed_rows=3, other8623rows_and_capture_times_unchanged=True,
        full_test_shape=list(truth.shape), per_seed_held_out_coordinates=1944331, per_seed_month_groups=10,
        full_original_HDF5_truth_mask_unit_month_identity_and_metrics_verified=True,
        independently_recomputed_original_seed_summaries=summaries, full_per_seed_array_reviews=verified,
        original_training_seeds=[26, 27], required_original_seeds=5, full_original_seed_cohort_complete=False,
        all8626embedded_source_records_exact=True, author_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False, additional_training_seed_slots=0,
        publication_receipt_sha256=digest(target / 'publication_validation.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--worker', choices=['publish', 'review']); args = parser.parse_args()
    config = read(ROOT / args.config); verify_registration(config)
    if args.worker:
        return (publish if args.worker == 'publish' else review)(args.config)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Two-seed publication controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    try:
        for phase in ('publish', 'review'):
            with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                cmd = [config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.publish_saits_two_seed_cohort_v52', '--config', args.config, '--worker', phase]
                with (base / (phase + '.stdout.log')).open('xb') as out, (base / (phase + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
                            PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    p = psutil.Process(child.pid); update(dict(state='running_' + phase, active_pid=p.pid,
                        active_create_time=p.create_time(), actual_command=p.cmdline()))
                    code, usage = capped_wait(child, config, api, update)
            exclusive(base / (phase + '_execution_receipt.json'), dict(exit_code=code, **usage))
            if code != 0 or usage['resource_guard_aborted']:
                update(dict(state='failed_preserved', active_pid=None)); return
            update(dict(state='between_phases', active_pid=None))
        update(dict(state='completed', active_pid=None, original_full_seeds=2,
            full_original_seed_cohort_complete=False, all_original_experiments_complete=False))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
