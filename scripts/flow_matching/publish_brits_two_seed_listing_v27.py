"""Publish only the three fully audited original BRITS Electricity seed updates."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import math
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

ROOT, legacy = previous.ROOT, previous.legacy
METRICS = ('mae', 'rmse', 'mre')


def validated_cohort(proofs, seeds):
    records = []
    for proof in proofs:
        if (not proof['full_checkpoint_contents_and_re_inference_passed']
                or proof['full_models_replayed'] != len(proof['records'])
                or proof['additional_training_seed_slots'] != 0
                or proof['cuda_context_initialized']):
            raise ValueError('Actually completed full original CPU model replays required')
        for record in proof['records']:
            if not record['id'].startswith('saits_paper_brits_'):
                continue
            if (record['id'] != 'saits_paper_brits_electricity_author_missing0.1_seed%d' % record['seed']
                    or record['model_parameters'] != 6993328
                    or record['full_test_shape'] != [290, 100, 370]
                    or record['held_out_coordinates'] != 1020954
                    or not record['full_original_checkpoint_and_data_identity_verified']
                    or not record['full_test_grid_and_grouped_statistics_replayed']
                    or not record['full_training_completion']['full_training_finished']
                    or record['full_training_completion']['benchmark_smoke']):
                raise ValueError('Original complete BRITS architecture, data, seed and budget required')
            for metric in METRICS:
                value = record['original_metrics'][metric]
                if not math.isfinite(value) or any(not math.isclose(
                        record[field][metric], value, rel_tol=2e-6, abs_tol=2e-6)
                        for field in ('replayed_metrics', 'independently_recomputed_metrics')):
                    raise ValueError('Original full native metrics must independently agree')
            records.append(record)
    if sorted(record['seed'] for record in records) != seeds or len({r['id'] for r in records}) != len(seeds):
        raise ValueError('Exactly the registered unique original training seeds required')
    if len({r['original_full_dataset_sha256'] for r in records}) != 1:
        raise ValueError('Every seed must use the identical original frozen dataset')
    return sorted(records, key=lambda record: record['seed'])


def updates(original, records, source, captured):
    rows = []
    for metric in METRICS:
        matches = [row for row in original if row['algorithm'] == 'brits'
            and row['dataset'] == 'electricity' and row['task'] == 'imputation'
            and row['protocol'] == 'author' and row['pattern'] == 'paper_mask'
            and row['missing_ratio'] == '0.1' and row['metric'] == metric]
        if len(matches) != 1 or int(matches[0]['required_repeats']) != 5 or int(matches[0]['n']) != 1:
            raise ValueError('Exactly the frozen original five-seed BRITS slots required')
        row = dict(matches[0])
        row.update(independent_stats([record['original_metrics'][metric] for record in records]))
        row.update(source=source, captured_utc=captured, author_equivalence_certified=False,
            strict_TAB_result=False, comparison='full_native_models_audited_2_of_5_author_alignment_unverified')
        rows.append(row)
    return rows


def check_increment(original, rows, changed_rows):
    changed_keys = {previous.key(row) for row in changed_rows}
    differences = [i for i, (a, b) in enumerate(zip(original, rows)) if a != b]
    if (len(rows) != len(original) or len(changed_keys) != 3 or len(differences) != 3
            or any(previous.key(rows[i]) not in changed_keys for i in differences)):
        raise ValueError('Only three original BRITS metric slots may change')
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full native publication evidence differs')
    proofs = [legacy.read(ROOT / path) for path in config['full_model_proofs']]
    records = validated_cohort(proofs, config['original_training_seeds'])
    for path in config['full_model_guard_receipts']:
        receipt = legacy.read(ROOT / path)
        if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
                or not receipt['full_native_model_replay_passed']):
            raise ValueError('Every full original model replay must finish successfully')
    for record in records:
        for field in ('config', 'result', 'checkpoint', 'replay_arrays'):
            path = record[field] if field in ('config', 'replay_arrays') else record[field + '_path']
            if legacy.sha(ROOT / path) != record[field + '_sha256']:
                raise ValueError('Actual original full BRITS configuration or artifact differs')
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, original = reader.fieldnames, list(reader)
    captured = max(proof['captured_utc'] for proof in proofs)
    changed_rows = updates(original, records, config['cohort_summary_path'], captured)
    rows = previous.merge_audited_rows(original, changed_rows)
    changed = check_increment(original, rows, changed_rows)
    measured = sum(row['mean'] not in ('', None) for row in rows)
    if len(rows) != 7283 or measured != old['measured_metric_rows']:
        raise ValueError('Preserve every registered metric and all existing measured slots')
    target = ROOT / config['output_directory']
    proof_path = ROOT / config['cohort_summary_path']
    if target.exists() or proof_path.exists():
        raise FileExistsError('Published complete inventories and cohort captures are immutable')
    target.mkdir(parents=True)
    legacy.write(proof_path, dict(captured_utc=captured,
        original_training_seeds=config['original_training_seeds'], required_training_seeds=5,
        actual_full_native_models_replayed=2, full_test_windows_per_seed=290,
        held_out_coordinates_per_seed=1020954, full_group_identity_and_statistics_verified=True,
        source_proofs=config['full_model_proofs'], original_metric_values=[r['original_metrics'] for r in records],
        metric_rows=changed_rows, additional_training_seed_slots=0, author_equivalence_certified=False,
        all_original_experiments_complete=False))
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    for name in ('overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '评分配方',
        '指标', '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集时间UTC']
    values = [[row['task'], row['algorithm'], row['dataset'], row['protocol'], row.get('epochs', ''),
        str(row.get('pattern', '')) + '/' + str(row.get('missing_ratio', '')), row.get('score_recipe', ''),
        row['metric'], '%s/%s' % (row['n'], row['required_repeats']),
        *[legacy.fmt(row.get(key)) for key in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
        row['captured_utc']] for row in rows]
    (target / 'all_metrics.md').write_text('# 完整算法—数据集实验技术指标\n\n'
        '本版仅更新BRITS/Electricity两个完整原始训练种子的三项指标；其余每行采集时间保留。'
        '—表示缺结果，0表示实测零。SMAP历史预处理缺口及修正数据重跑仍未完成；'
        '清单完整不表示全部实验完成。\n\n' + legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    summary = [[row['metric'], '2/5', *[legacy.fmt(row.get(key)) for key in
        ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')]] for row in changed_rows]
    (target / 'README.rst').write_text('完整算法—数据集指标 v27\n==========================\n\n'
        f'{len(rows)}项指标；{measured}项有实测值；{len(rows)-measured}项缺结果。仅更新BRITS/Electricity'
        '作者协议、名义10%缺失率的三个指标：2/5原始独立训练种子（26、27），'
        '每份完整模型均已实际重推断，并与冻结HDF5真值/掩码/单位/月份分组及误差统计独立核验。\n\n' +
        legacy.table(['指标', '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值'], summary) +
        '\n\nSTD、SE和Student-t95%区间使用两个独立训练种子，样本不足使区间较宽；'
        '不能作为完整五种子结论。原作者10000轮上限及patience30保持，实际原早停分别为'
        'next_epoch742/949；两份检查点覆盖290窗口、1020954遮蔽坐标和十个月分组。'
        '重推断不增加训练种子，既有2e-6容差未放宽。1397与论文1400预处理窗口及历史调参缺口仍保留。\n\n'
        '其余7280项指标和采集时间与v26逐字段一致；异常检测总览及其余三个既有导出逐字节复制。'
        'v26之后完成的Anomaly Transformer/SMD、MOMENT-base/SWAN与GRASP/Lin增量未混入本次发布，'
        '各自核验材料及运行状态保留在研究项目中。SMAP历史结果存在已查明的常数特征预处理问题；'
        '修正数据上的291项完整对照仍在排队，不能把历史列视为修正后的结果。\n\n'
        '45篇论文、681方法/基线/消融、273原始数据任务及采用基线仍未全部完成；'
        '作者等价和完整TAB训练框架未认证。完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。\n',
        encoding='utf-8')
    published = datetime.now(timezone.utc).isoformat()
    inventory_keys = ('aggregate_metric_rows', 'task_rows', 'strict_groups', 'strict_complete_groups',
        'strict_partial_groups', 'cfm_completed_canonical_slots', 'cfm_complete_groups',
        'grasp_complete_cohorts', 'algorithm_configuration_names')
    receipt = {key: old[key] for key in inventory_keys if key in old}
    receipt.update(published_utc=published, base_publication=config['base_publication'],
        base_publication_utc=old['published_utc'], base_receipt_sha256=legacy.sha(base / 'publication_validation.json'),
        measured_metric_rows=measured, missing_metric_rows=len(rows)-measured,
        measured_task_rows=dict(Counter(row['task'] for row in rows if row['mean'] not in ('', None))),
        replaced_metric_rows=3, changed_numeric_metric_rows=3, changed_row_positions=changed,
        newly_measured_metric_rows=0, newly_completed_canonical_training_slots=1,
        brits_electricity_completed_training_seeds=2, brits_electricity_required_training_seeds=5,
        brits_full_native_models_replayed=2, other_track_rows_and_capture_times_preserved=True,
        all_numeric_tables_byte_identical_to_base=False, base_numeric_csv_prefix_byte_identical=False,
        performance_rows_changed=True, additional_training_seed_slots=0, other_tracks_recaptured=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, checks=dict(all_existing_rows_retained=True,
            exactly_three_BRITS_metric_updates=True, all_other_7280_rows_and_timestamps_unchanged=True,
            original_five_seed_obligations_retained=True, all_two_full_native_checkpoint_replays_finished=True),
        source_receipts=old['source_receipts'] + config['source_receipts'] +
            [dict(path=args.config, sha256=legacy.sha(ROOT / args.config)),
             dict(path=config['cohort_summary_path'], sha256=legacy.sha(proof_path))])
    receipt['outputs'] = {path.name: legacy.sha(path) for path in target.iterdir() if path.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, missing=len(rows)-measured, brits_seeds=2,
        changed_original_slots=3, newly_measured_slots=0))


if __name__ == '__main__':
    main()
