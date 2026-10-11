"""Publish the third original fully replayed BRITS Electricity training seed."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import shutil

from scripts.flow_matching.publish_brits_two_seed_listing_v27 import (
    ROOT, legacy, previous, validated_cohort, METRICS)
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

SEEDS = [26, 27, 28]


def complete_delta(before, after):
    if len(before) != len(after):
        raise ValueError('Preserve all original registered slots')
    changed = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        identity = ('algorithm', 'dataset', 'task', 'protocol', 'pattern', 'missing_ratio', 'metric')
        if any(k not in old or k not in new for k in identity):
            raise ValueError('Every changed slot must retain its complete original identity')
        if (any(old[k] != new[k] for k in identity)
                or new['algorithm'] != 'brits' or new['dataset'] != 'electricity'
                or new['task'] != 'imputation' or new['protocol'] != 'author'
                or new['pattern'] != 'paper_mask' or new['missing_ratio'] != '0.1'
                or new['metric'] not in METRICS
                or (str(old['n']), str(new['n']), str(old['required_repeats']),
                    str(new['required_repeats'])) != ('2', '3', '5', '5')):
            raise ValueError('Only the three original BRITS metrics may advance from2/5to3/5')
        changed.append(index)
    if len(changed) != 3 or {after[i]['metric'] for i in changed} != set(METRICS):
        raise ValueError('Every one of the three original metrics must advance')
    return changed


def publish(config_path):
    config = legacy.read(ROOT / config_path)
    for source in config['source_receipts']:
        if legacy.sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen original publication source differs')
    if config['original_training_seeds'] != SEEDS:
        raise ValueError('Exactly the three original unique training seeds required')
    proofs = [legacy.read(ROOT / path) for path in config['full_model_proofs']]
    records = validated_cohort(proofs, SEEDS)
    for path in config['full_model_guard_receipts']:
        receipt = legacy.read(ROOT / path)
        if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
                or not receipt['full_native_model_replay_passed']):
            raise ValueError('Every complete model replay must succeed under its original guard')
    for record in records:
        for field in ('config', 'result', 'checkpoint', 'replay_arrays'):
            path = record[field] if field in ('config', 'replay_arrays') else record[field + '_path']
            if legacy.sha(ROOT / path) != record[field + '_sha256']:
                raise ValueError('Actual full original native artifact changed')
        completion = record['full_training_completion']
        if (completion['original_maximum_epochs'] != 10000
                or completion['full_test_units'] != 290
                or completion['held_out_coordinates'] != 1020954
                or (completion['next_epoch'] < 10000 and completion['author_patience_remaining'] > 0)):
            raise ValueError('Original full training budget and native early stop must remain')
    base = ROOT / config['base_publication']
    old_receipt = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, before = reader.fieldnames, list(reader)
    rows = [dict(row) for row in before]
    captured = datetime.now(timezone.utc).isoformat()
    changed_rows = []
    for row in rows:
        if (row['algorithm'], row['dataset'], row['task'], row['protocol'], row['pattern'],
                row['missing_ratio']) != ('brits', 'electricity', 'imputation', 'author', 'paper_mask', '0.1'):
            continue
        if row['metric'] not in METRICS:
            continue
        row.update(independent_stats([r['original_metrics'][row['metric']] for r in records]))
        row.update(source=config['cohort_summary_path'], captured_utc=captured,
            author_equivalence_certified=False, strict_TAB_result=False,
            comparison='full_native_models_audited_3_of_5_author_alignment_unverified')
        changed_rows.append(row)
    positions = complete_delta(before, rows)
    measured = sum(bool(row['mean']) for row in rows)
    if len(rows) != 7283 or len(headers) != 31 or measured != old_receipt['measured_metric_rows']:
        raise ValueError('All7283records and31fields and existing measured coverage required')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve prior publications')
    target.mkdir(parents=True)
    summary = dict(captured_utc=captured, original_training_seeds=SEEDS, required_training_seeds=5,
        source_proofs=config['full_model_proofs'], original_metric_values=[r['original_metrics'] for r in records],
        metric_rows=changed_rows, full_native_checkpoint_models_replayed=3,
        additional_training_seed_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False)
    legacy.write(ROOT / config['cohort_summary_path'], summary)
    with (target / 'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    for path in base.iterdir():
        if path.is_file() and path.name not in ('all_algorithm_dataset_metrics.csv', 'all_metrics.md',
                'results.html', 'README.rst', 'publication_validation.json'):
            shutil.copyfile(path, target / path.name)
    boundary = ('本版仅将原始 BRITS/Electricity、10%缺失率的 MAE/RMSE/MRE 从2/5更新为3/5训练种子。'
        '其余7280项指标及采集时间保持不变。三个完整模型均已重放全部测试窗口与遮蔽坐标。'
        '全部登记任务尚未完成；作者及完整TAB等价未认证，旧SMAP结果仍为历史记录。')
    (target / 'all_metrics.md').write_text('# 完整算法—数据集实验技术指标 v36\n\n' + boundary + '\n\n' +
        legacy.table(headers, [[row.get(k) if row.get(k) not in ('', None) else '—' for k in headers]
            for row in rows]) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    (target / 'README.rst').write_text('完整算法—数据集指标 v36\n==========================\n\n' +
        boundary + '\n\n' + '原始训练种子26、27、28；每个模型6993328参数、290测试窗口、1020954待插补坐标。'
        '原作者10000轮上限和patience30保留；实际停止轮数742、949、1028。'
        '所有三项种子均值、STD、SE、Student-t95%区间按三个独立训练种子计算，完整五种子结论尚未成立。'
        '1397与论文1400预处理窗口及历史调参缺口继续保留。原210项SAITS/BRITS时间序列队列未缩小。\n\n'
        f'全部7283项登记指标；实测{measured}项，缺结果{7283-measured}项。'
        '45论文、681方法/基线/消融、273原始数据任务及采用基线保持完整范围。\n', encoding='utf-8')
    receipt = dict(old_receipt)
    receipt.update(published_utc=captured, base_publication=config['base_publication'],
        base_publication_utc=old_receipt['published_utc'],
        base_receipt_sha256=legacy.sha(base / 'publication_validation.json'),
        replaced_metric_rows=3, changed_numeric_metric_rows=3, changed_row_positions=positions,
        newly_measured_metric_rows=0, newly_completed_canonical_training_slots=1,
        brits_electricity_completed_training_seeds=3, brits_electricity_required_training_seeds=5,
        brits_full_native_models_replayed=3, other_track_rows_and_capture_times_preserved=True,
        all_numeric_tables_byte_identical_to_base=False, base_numeric_csv_prefix_byte_identical=False,
        performance_rows_changed=True, additional_training_seed_slots=0, other_tracks_recaptured=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False,
        measured_task_rows=dict(Counter(row['task'] for row in rows if row['mean'])),
        checks=dict(all7283_original_slots_and31fields_retained=True,
            exactly_three_original_BRITS_metrics_2_to_3_of_5=True,
            all_other7280_rows_and_capture_times_exact=True, three_original_full_model_replays_succeeded=True),
        source_receipts=old_receipt['source_receipts'] + config['source_receipts'] + [
            dict(path=config_path, sha256=legacy.sha(ROOT / config_path)),
            dict(path=config['cohort_summary_path'], sha256=legacy.sha(ROOT / config['cohort_summary_path']))])
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(json.dumps(dict(rows=len(rows), measured=measured, original_training_seeds=SEEDS,
        changed_metrics=3, original_five_seed_group_complete=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    publish(parser.parse_args().config)
