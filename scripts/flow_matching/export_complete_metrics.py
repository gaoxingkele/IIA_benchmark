"""Read-only, source-verified export of all registered experiment metrics."""
from __future__ import annotations

import csv
import argparse
import hashlib
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.build_result_table import ROOT, build


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def numeric_leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from numeric_leaves(child, f'{prefix}[{i}]')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def write_csv(target, rows, headers):
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reproducibility/fm_result_table.2026-10-09.json')
    args = parser.parse_args()
    config_path = ROOT / args.config
    settings = read(config_path)
    snapshot_path = ROOT / settings['tsad_snapshot']
    snapshot = read(snapshot_path)
    imputation = build(ROOT, read(ROOT / settings['imputation_settings']))
    target = ROOT / settings['output_directory']
    target.mkdir(parents=True, exist_ok=True)
    archived_snapshot = target / 'tsad_source_snapshot.json'
    archived_snapshot.write_bytes(snapshot_path.read_bytes())
    ranges = {r['id']: r for r in snapshot['tab_range_evaluations']['records']}
    groups = defaultdict(list)
    for job in snapshot['new_jobs']:
        groups[job['model_config'], job['dataset']].append(job)
    seed_stats = {(r['model_config'], r['dataset']): r for r in snapshot['seed_aggregates']}
    range_stats = {(r['model_config'], r['dataset']): r for r in snapshot['tab_range_evaluations']['seed_aggregates']}
    summary, strict_runs, diagnostic, leaves, entity_runs = [], [], [], [], []
    sources = [{'path': archived_snapshot.relative_to(ROOT).as_posix(), 'sha256': sha(archived_snapshot)},
               {'path': str(config_path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(config_path)}]
    author_tracks = snapshot.get('author_pipeline_experiments', [])
    author_jobs, author_leaves = [], []
    for track in author_tracks:
        sources.append({'path': track['queue'], 'sha256': track['queue_sha256']})
        for job in track['jobs']:
            author_jobs.append(dict(job, algorithm=track['name'], protocol='author_pipeline',
                                    strict_TAB_result=False, boundary=track['boundary']))
            if job['status'] == 'completed':
                source = ROOT / job['result_path']
                if sha(source) != job['result_sha256']:
                    raise ValueError(f'Author result changed: {source}')
                sources.append({'path': job['result_path'], 'sha256': job['result_sha256']})
                for name, value in numeric_leaves(job['metrics']):
                    author_leaves.append({'algorithm': track['name'], 'dataset': job['dataset'],
                                          'author_recipe': job['author_recipe'], 'seed': job['seed'],
                                          'run_id': job['id'], 'metric_path': name, 'value': value,
                                          'result_path': job['result_path'], 'result_sha256': job['result_sha256']})
    for (model, dataset), jobs in groups.items():
        metrics = seed_stats.get((model, dataset), {}).get('metrics', {})
        range_metrics = range_stats.get((model, dataset), {}).get('metrics', {})
        counts = Counter(j['status'] for j in jobs)
        row = {'algorithm_config': Path(model).stem, 'dataset': dataset.upper(), 'required_seeds': len(jobs),
               'completed_seeds': counts['completed'], 'running': counts['running'],
               'partial_or_failed': counts['partial_or_failed'], 'pending': counts['pending'],
               'status': 'completed' if counts['completed'] == len(jobs) else 'partial' if counts['completed'] else 'no_complete_result',
               'protocol': 'validation_only_1_percent_no_PA', 'model_config': model, 'paper_equivalence': False}
        for key in ('precision', 'recall', 'f1', 'auroc', 'average_precision'):
            for statistic in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'):
                row[f'{key}_{statistic}'] = metrics.get(key, {}).get(statistic)
        for key in ('VUS_ROC', 'VUS_PR', 'affiliation_f'):
            for statistic in ('n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high'):
                row[f'{key}_{statistic}'] = range_metrics.get(key, {}).get(statistic)
        summary.append(row)
    for record in snapshot['completed_new_runs']:
        source = ROOT / record['result_path']
        if sha(source) != record['result_sha256']:
            raise ValueError(f"Result changed: {source}")
        result = read(source)
        common = {'run_id': record['id'], 'algorithm_config': Path(record['model_config']).stem,
                  'dataset': record['dataset'].upper(), 'seed': record['seed'],
                  'result_path': record['result_path'], 'result_sha256': record['result_sha256'],
                  'checkpoint_sha256': record['checkpoint_sha256'], 'scores_sha256': record['scores_sha256']}
        sources.append({'path': record['result_path'], 'sha256': record['result_sha256']})
        range_record = ranges.get(record['id'])
        if range_record:
            rp = ROOT / range_record['evaluation_path']
            if sha(rp) != range_record['evaluation_sha256']:
                raise ValueError(f'Range evaluation changed: {rp}')
            sources.append({'path': range_record['evaluation_path'], 'sha256': range_record['evaluation_sha256']})
            for name, value in numeric_leaves(read(rp)):
                leaves.append(dict(common, metric_path='TAB_ranges.' + name, value=value,
                                   metric_source=range_record['evaluation_path']))
        for name, value in numeric_leaves(result['metrics']):
            leaves.append(dict(common, metric_path=name, value=value, metric_source=record['result_path']))
        for ratio, data in result['metrics']['strict'].items():
            row = dict(common, calibration_false_alarm_percent=float(ratio), threshold=data['calibration']['threshold'],
                       protocol=data['protocol'], **data['micro'], macro_f1=data['macro_f1'], **data['event_metrics'],
                       f1_ci95_low=data['f1_ci95']['low'], f1_ci95_high=data['f1_ci95']['high'],
                       ci_unit=data['f1_ci95']['unit'], training_seconds=result['training_seconds'],
                       score_seconds=result['score_seconds'], parameter_count=result['parameter_count'],
                       peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
                       epochs_completed=len(result['training_losses']) if result['training_losses'] is not None else None)
            if range_record:
                for key, data_range in range_record['VUS_entity_macro'].items():
                    row[key] = data_range['mean_over_defined_entities']
                    row[key + '_defined_entities'] = data_range['defined_entities']
                    row[key + '_undefined_entities'] = data_range['undefined_entities']
                for key, data_range in range_record['strict_affiliation_entity_macro'][ratio].items():
                    row[key] = data_range['mean_over_defined_entities']
                    row[key + '_defined_entities'] = data_range['defined_entities']
                    row[key + '_undefined_entities'] = data_range['undefined_entities']
            strict_runs.append(row)
            for entity, values in data['per_entity'].items():
                entity_runs.append(dict(common, calibration_false_alarm_percent=float(ratio), entity=entity, **values))
        tab = result['metrics']['tab_core_diagnostic']
        for item in tab['ratio_grid']:
            diagnostic.append(dict(common, **item, auc_roc=tab['auc_roc'], auc_pr=tab['auc_pr'],
                                   test_scores_used_in_calibration=True, test_labels_used_to_select_best_grid=True))
    history = []
    for record in snapshot['legacy_mtsad_records']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError(f"Historical source changed: {record['path']}")
        for protocol, metrics in record['protocols'].items():
            history.append({'model': record['model'], 'dataset': record['dataset'].upper(), 'seed': record['seed'],
                            'record_id': record['path'], 'protocol': protocol,
                            'parameters': json.dumps(record['parameters'], ensure_ascii=False), **metrics,
                            'source_sha256': record['sha256'], 'status': record['status']})
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'tsad_capture_utc': snapshot['captured_utc'],
              'summary': {'registered_tsad_attempts': len(snapshot['new_jobs']), 'tsad_counts': snapshot['new_job_counts'],
                          'tsad_model_dataset_groups': len(summary), 'complete_tsad_runs': len(snapshot['completed_new_runs']),
                          'TAB_range_complete_runs': len(ranges), 'historical_runs': len(snapshot['legacy_mtsad_records']),
                          'author_pipeline_registered_jobs': len(author_jobs),
                          'author_pipeline_counts': dict(Counter(j['status'] for j in author_jobs)),
                          'historical_protocol_rows': len(history), 'imputation': imputation['summary']},
              'strict_summary': sorted(summary, key=lambda r: (-r['completed_seeds'], r['algorithm_config'], r['dataset'])),
              'strict_runs': strict_runs, 'tab_ratio_diagnostic': diagnostic, 'historical_runs': history,
              'imputation': imputation, 'tsad_jobs': snapshot['new_jobs'], 'sources': sources,
              'author_pipeline_tracks': author_tracks, 'author_pipeline_jobs': author_jobs,
              'paper_scope': snapshot['original_paper_scope'], 'remaining_obligations': snapshot['other_remaining_obligations'],
              'boundary': 'Complete registered-source snapshot, not proof all papers/ablations executed or all paper tables transcribed. Strict point F1, PA F1, affiliation F, VUS and imputation errors are separate. Nonoverlapping training differs from stride=1 author budgets.'}
    (target / 'complete_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    exports = {'strict_summary': report['strict_summary'], 'strict_per_seed': strict_runs,
               'strict_per_entity': entity_runs, 'tab_ratio_diagnostic': diagnostic,
               'all_tsad_numeric_metrics': leaves, 'historical_protocol_results': history,
               'imputation_summary': imputation['formal_results'], 'imputation_per_run': imputation['per_run_metrics'],
               'registered_tsad_jobs': snapshot['new_jobs'], 'registered_imputation_jobs': imputation['jobs'],
               'paper_claims': imputation['author_claims'], 'original_dataset_scope': imputation['original_dataset_scope']}
    exports['author_pipeline_jobs'] = author_jobs
    exports['author_pipeline_numeric_metrics'] = author_leaves
    for name, rows in exports.items():
        flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
        headers = list(dict.fromkeys(k for r in flat for k in r))
        if name == 'author_pipeline_numeric_metrics' and not headers:
            headers = ['algorithm', 'dataset', 'author_recipe', 'seed', 'run_id', 'metric_path', 'value', 'result_path', 'result_sha256']
        write_csv(target / (name + '.csv'), flat, headers)
    # The exhaustive repeated provenance table is large as text; preserve the
    # local CSV and commit its deterministic, byte-identical compressed copy.
    numeric_csv = target / 'all_tsad_numeric_metrics.csv'
    (target / 'all_tsad_numeric_metrics.csv.gz').write_bytes(gzip.compress(numeric_csv.read_bytes(), mtime=0))
    def fmt(value):
        return '—' if value is None else f'{value:.6f}'
    lines = ['# 算法—数据集完整实验指标', '', f"快照：{report['captured_utc']}。异常检测源快照：{snapshot['captured_utc']}。",
             '', '这里完整列出已登记任务和已有指标，未完成项留空。全论文、全消融实验尚未全部完成。', '',
             'strict_summary.csv 覆盖所有已登记算法配置—数据集组合；strict_per_seed.csv 保留三档验证阈值、P/R/F1、AUROC、AP、事件召回、延迟、误报率、置信区间、参数量、耗时和显存。',
             'TAB 的 VUS/Affiliation 单列，已保存全实体及未定义计数。all_tsad_numeric_metrics.csv 保留每个数值及原字段路径。历史与论文报告值分别存储。', '',
             '**比较约束：**严格主表只在验证段按 1% 分位数定阈值，不做 PA。TAB 比例诊断合并训练/测试分数校准，最优比例依赖测试标签，不能充当严格泛化成绩。非重叠训练比部分作者 stride=1 少很多梯度更新，不能据此判定原论文无效。', '',
             '| 算法配置 | 数据集 | 完成/预定种子 | Precision | Recall | 点级 F1 | AUROC | AP | VUS ROC（种子数） | VUS PR | Affiliation F（种子数） |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['strict_summary']:
        if not r['completed_seeds']:
            continue
        lines.append(f"| {r['algorithm_config']} | {r['dataset']} | {r['completed_seeds']}/{r['required_seeds']} | {fmt(r['precision_mean'])} | {fmt(r['recall_mean'])} | {fmt(r['f1_mean'])} | {fmt(r['auroc_mean'])} | {fmt(r['average_precision_mean'])} | {fmt(r['VUS_ROC_mean'])} ({r['VUS_ROC_n'] or 0}) | {fmt(r['VUS_PR_mean'])} | {fmt(r['affiliation_f_mean'])} ({r['affiliation_f_n'] or 0}) |")
    lines += ['', '各指标可能由不同数量的已完成种子产生，范围指标种子数必须一起读取；不混合缺失值为零。独立 CFM 与单阶段 Rectified 在这里 sigma=0 时等价。迭代 reflow 独立列行，其完整队列尚未全部完成。', '',
              '## 插补实验已有结果', '', '| 算法 | 数据集 | 缺失率 | 掩码/协议 | 指标 | 均值 | SE | 完成/预定折或种子 | 论文值 | 比较状态 |',
              '|---|---|---:|---|---|---:|---:|---:|---:|---|']
    for r in imputation['formal_results']:
        if r['local_mean'] is not None:
            lines.append(f"| {r['algorithm']} | {r['dataset_label']} | {r['missing_ratio']} | {r['pattern']} / {r['track']} | {r['metric']} | {fmt(r['local_mean'])} | {fmt(r['local_se'])} | {r['completed_repeats']}/{r['required_repeats']} | {fmt(r['paper_mean'])} | {r['verdict']} |")
    lines += ['', '数值兼容筛查不等于统计等价证明；not_comparable、incomplete_repeats 等原因详见 imputation_summary.csv。插补误差不能代替异常检测成绩。', '',
              '## 文件', '', '[完整 Excel](complete_experiment_metrics.xlsx)、[所有组合与状态](strict_summary.csv)、[逐种子严格指标](strict_per_seed.csv)、[历史协议结果](historical_protocol_results.csv)、[插补汇总](imputation_summary.csv)、[论文报告值](paper_claims.csv)、[完整机器可读结果](complete_results.json)。', '',
              '原文数据范围、未运行任务及每篇复现缺口均保留在 CSV/JSON 与 Excel 中。']
    (target / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (target / 'README.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## 作者完整流程\n\n')
        stream.write(f"独立作者流程登记 {len(author_jobs)} 项，状态 {dict(Counter(j['status'] for j in author_jobs))}。")
        stream.write('逐项配方、数据集、种子和阶段状态见 author_pipeline_jobs.csv；尚无完整流程指标时保留空值，不计入严格成绩。\n')
    validation = {'captured_utc': report['captured_utc'], 'checks': {
        'tsad_job_accounting': sum(snapshot['new_job_counts'].values()) == len(snapshot['new_jobs']),
        'all_registered_tsad_groups_present': sum(r['required_seeds'] for r in summary) == len(snapshot['new_jobs']),
        'all_complete_runs_have_three_strict_rows': len(strict_runs) == 3 * len(snapshot['completed_new_runs']),
        'all_complete_runs_have_ten_ratio_diagnostics': len(diagnostic) == 10 * len(snapshot['completed_new_runs']),
        'missing_groups_never_zero_filled': all(r['f1_mean'] is None for r in summary if not r['completed_seeds']),
        'source_result_and_range_hashes_verified': True,
        'author_pipeline_jobs_all_accounted_for': len(author_jobs) == sum(t['registered_jobs'] for t in author_tracks),
        'author_pipeline_not_promoted_to_strict_results': all(not j['strict_TAB_result'] for j in author_jobs),
        'imputation_jobs_all_accounted_for': len(imputation['jobs']) == 540,
        'paper_claims_not_promoted_to_local_results': all(r['local_value'] is None for r in imputation['author_claims'])},
        'csv_row_counts': {name: len(rows) for name, rows in exports.items()},
        'source_count': len(sources), 'outputs': {p.name: sha(p) for p in target.iterdir() if p.is_file() and p.name != 'validation.json'}}
    if not all(validation['checks'].values()):
        raise ValueError(validation['checks'])
    (target / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
