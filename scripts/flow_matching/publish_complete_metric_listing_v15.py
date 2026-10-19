"""Publish the full registered metric inventory with a verified CFM increment."""
import argparse
import copy
import csv
from collections import Counter
from datetime import datetime, timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT, legacy = previous.ROOT, previous.legacy


def cfm_updates(capture, source):
    rows = []
    for record in capture['seed_aggregates']:
        for metric in ('MSE', 'MAE', 'RMSE'):
            rows.append(dict(
                algorithm='CFM-TS/' + record['variant'], dataset=record['dataset'],
                task='continuous_time_dynamics', protocol=record['track'],
                epochs=record['epochs'], metric=metric, n=record[metric + '_n'],
                required_repeats=record['required_seeds'],
                **{k: record[metric + '_' + k] for k in
                   ('mean', 'std', 'se', 'ci95_low', 'ci95_high')},
                paper_mean=record['paper_MSE_mean'] if metric == 'MSE' else None,
                paper_std=record['paper_MSE_std'] if metric == 'MSE' else None,
                comparison=record['comparison'], captured_utc=capture['captured_utc'],
                source=source, author_equivalence_certified=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen publication evidence changed')
    base, capture = ROOT / config['base_publication'], ROOT / config['cfm_capture']
    old = previous.verified_capture(base, 'publication_validation.json')
    audit = legacy.read(capture / 'independent_full_metrics_audit.json')
    cfm = legacy.read(capture / 'execution_audit.json')
    if (not audit['all_completed_full_arrays_and_metrics_verified']
            or not audit['all_seed_aggregates_independently_verified']
            or audit['source_capture_sha256'] != legacy.sha(capture / 'execution_audit.json')
            or audit['audited_full_canonical_slots'] != cfm['job_counts']['completed']):
        raise ValueError('Complete independent CFM prediction-array audit required')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers, original = reader.fieldnames, list(reader)
    updates = cfm_updates(cfm, config['cfm_capture'] + '/algorithm_dataset_metrics.csv')
    rows = previous.merge_audited_rows(original, updates)
    update_keys = {previous.key(r) for r in updates}
    if any(a != b for a, b in zip(original, rows) if previous.key(a) not in update_keys):
        raise ValueError('Other tracks or their capture timestamps changed')
    changed_numeric = sum(
        any(str(a.get(k) or '') != str(b.get(k) or '') for k in
            ('n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high'))
        for a, b in zip(original, rows))
    measured = sum(r['mean'] not in ('', None) for r in rows)
    if len(rows) != old['aggregate_metric_rows'] or measured != old['measured_metric_rows']:
        raise ValueError('This increment must update existing metric slots only')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Published result tables are immutable')
    target.mkdir(parents=True)
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    for name in ('overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率',
               '评分配方', '指标', '完成／预定', '均值', 'STD', 'SE',
               '95%下界', '95%上界', '论文值', '采集时间UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs', ''),
               str(r.get('pattern', '')) + '/' + str(r.get('missing_ratio', '')),
               r.get('score_recipe', ''), r['metric'], f"{r['n']}/{r['required_repeats']}",
               *[legacy.fmt(r.get(k)) for k in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
               r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text(
        '# 完整算法—数据集实验指标清单\n\n'
        '覆盖已登记的全部指标槽位，含方法、消融、协议、预算、论文值和未完成项。'
        '—表示缺结果，0表示实测零；完整清单不表示全部实验完成。'
        '其他轨道保留各自采集时间，本版只更新CFM-TS。\n\n' +
        legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    counts = Counter(r['task'] for r in rows)
    measured_counts = Counter(r['task'] for r in rows if r['mean'] not in ('', None))
    summary = [[task, count, measured_counts[task], count-measured_counts[task]]
               for task, count in sorted(counts.items())]
    (target / 'README.md').write_text(
        '# 完整算法—数据集实验技术指标\n\n'
        f'发布UTC：{now}。共{len(rows)}条指标：{measured}条有实测值，'
        f'{len(rows)-measured}条缺完整结果，覆盖{len({r["algorithm"] for r in rows})}个算法／配置名称。'
        '指标条数不是独立训练次数；数据集别名和不同协议保留原名称。\n\n'
        '[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；'
        '[全部文字指标](all_metrics.md)；[异常检测F1总览](overview.md)；'
        '[异常检测全部技术指标](strict_algorithm_dataset.md)。\n\n' +
        legacy.table(['任务', '登记指标', '有实测值', '缺完整结果'], summary) + '\n\n'
        '本版只更新CFM-TS的171条既有指标槽位；84/285个规范任务已完成，'
        '16个五种子组完整。所有84项的全量预测、原始数据身份、训练预算、'
        'MSE/MAE/RMSE与种子统计已独立重算。Pendulum/NODE/附录D/10轮由3/5增至4/5种子，'
        '其均值MSE=6.625421、MAE=1.887639、RMSE=2.261529；新增种子误差较高，未筛掉。'
        '[完整数组核验](../cfm_current_listing_v6/independent_full_metrics_audit.json)。'
        '本次未重新执行模型重推断；完整数组核验不等于作者实现等价认证。\n\n'
        '异常检测采用各行明示的协议。验证集99百分位定阈值、无PA的工程结果，'
        '尚未获完整TAB实验预算或作者等价认证；测试标签选阈值、PA和原生作者协议分别列出。'
        '固定发布权重的1/1表示一次评估，不能当作一个重新训练种子。\n\n'
        'LS4/PhysioNet原完整500轮训练曾在25轮后触发内存保护，部分结果未计入完整性能；'
        '新尝试和其他排队任务均不能作为实测完整指标。'
        '[保留中断和恢复记录](../ls4_physionet_memory_guard_recovery_v3/README.md)。'
        '本版不重新核验任何运行进程，不将发布日期视为各轨道采集时间。\n', encoding='utf-8')
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=now, base_publication_utc=old['published_utc'],
                   aggregate_metric_rows=len(rows), measured_metric_rows=measured,
                   missing_metric_rows=len(rows)-measured, task_rows=dict(counts),
                   measured_task_rows=dict(measured_counts), replaced_metric_rows=len(updates),
                   changed_numeric_metric_rows=changed_numeric, newly_measured_metric_rows=0,
                   cfm_completed_canonical_slots=cfm['job_counts']['completed'],
                   cfm_complete_groups=audit['full_five_seed_groups'],
                   cfm_all_completed_arrays_independently_recomputed=True,
                   other_track_rows_and_capture_times_preserved=True,
                   other_tracks_recaptured=False, all_original_experiments_complete=False,
                   additional_training_seed_slots=0)
    receipt['source_receipts'] += config['source_receipts'] + [
        dict(path=args.config, sha256=legacy.sha(ROOT / args.config))]
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, missing=len(rows)-measured,
               changed_numeric_rows=changed_numeric, cfm_full_slots=cfm['job_counts']['completed']))


if __name__ == '__main__':
    main()
