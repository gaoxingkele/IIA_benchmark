"""Publish a complete, immutable listing from audited captures, using stdlib only."""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def recipe(value):
    data = ast.literal_eval(value) if isinstance(value, str) else value
    return int(data['source_samples']), int(data['flow_evaluations'])


def update_rows(rows, cfm, native, cfm_source, native_source, native_capture):
    """Preserve measured native slots; add no repeat for exact recovery executions."""
    index = {}
    for row in rows:
        if row['task'] == 'continuous_time_dynamics':
            key = ('CFM', row['algorithm'], row['dataset'], row['protocol'],
                   int(row['epochs']), row['metric'])
        elif row.get('score_recipe') and row['algorithm'] in {r['algorithm'].split('/')[-1] for r in native}:
            key = ('GRASP', row['algorithm'], row['dataset'].lower(), row['protocol'],
                   recipe(row['score_recipe']), row['metric'])
        else:
            continue
        if key in index:
            raise ValueError('Duplicate canonical metric')
        index[key] = row
    changed = set()
    for group in cfm['seed_aggregates']:
        for metric in ('MSE', 'MAE', 'RMSE'):
            key = ('CFM', 'CFM-TS/' + group['variant'], group['dataset'],
                   group['track'], int(group['epochs']), metric)
            row = index[key]
            n = group[metric + '_n']
            if key in changed or int(row['n']) > n or int(row['required_repeats']) != group['required_seeds']:
                raise ValueError('Changed repeat budget or lost seed')
            changed.add(key)
            row.update(n=n, source=cfm_source, captured_utc=cfm['captured_utc'], comparison=group['comparison'])
            for field in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'):
                row[field] = group[metric + '_' + field]
    added_native = 0
    seen = set()
    for result in native:
        key = ('GRASP', result['algorithm'].split('/')[-1], result['dataset'].lower(), result['protocol'],
               (int(result['source_samples']), int(result['flow_evaluations'])), result['metric'])
        if key in seen:
            raise ValueError('Duplicate native update')
        seen.add(key)
        row = index[key]
        if int(result['n']) != 1 or int(row['required_repeats']) != int(result['required_seeds']):
            raise ValueError('Native repeat identity mismatch')
        if int(row['n']):
            if (int(row['n']) != 1 or row['execution_id'] != result['execution_id']
                or row['original_canonical_id'] != result['original_canonical_id']
                or float(row['mean']) != float(result['mean'])):
                raise ValueError('Refuse replacement of a different measured native slot')
        else:
            added_native += 1
        row.update(n=1, mean=float(result['mean']), std=None, se=None, ci95_low=None, ci95_high=None,
                   source=native_source, captured_utc=native_capture, paper_mean=result['paper_mean'],
                   original_canonical_id=result['original_canonical_id'], execution_id=result['execution_id'],
                   comparison='one_seed_numerical_comparison_only')
        changed.add(key)
    return len(changed), added_native


def fmt(value):
    return '—' if value is None or value == '' else f'{float(value):.6f}'


def table(headers, rows):
    def text(value):
        return str(value).replace('|', '\\|').replace('\n', ' ') if value is not None else '—'
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(text(v) for v in row) + ' |' for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read(config_path)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Published captures are immutable')
    base = ROOT / config['base_catalog']
    receipt = read(base / 'publication_validation.json')
    for entry in receipt['source_receipts']:
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Published source changed')
    for name, expected in receipt['outputs'].items():
        if sha(base / name) != expected:
            raise ValueError('Published output changed')
    cfm_dir, native_dir = ROOT / config['cfm_capture'], ROOT / config['native_capture']
    cfm, native_audit = read(cfm_dir / 'execution_audit.json'), read(native_dir / 'independent_array_audit.json')
    audited = native_audit['audited_full_results']
    if len(audited) != 2 or not all(r['metrics_independently_recomputed'] and r['full_row_and_label_identity_checked'] and r['saved_checkpoint_binding_checked'] for r in audited):
        raise ValueError('Full native independent audit absent')
    for entry in audited:
        if sha(ROOT / entry['result_path']) != entry['result_sha256']:
            raise ValueError('Audited native result changed')
    rows = csv_rows(base / 'all_algorithm_dataset_metrics.csv')
    native = csv_rows(native_dir / 'algorithm_dataset_metrics.csv')
    if len(native) != 66 or {r['original_canonical_id'] for r in native} != set(native_audit['canonical_original_slots']):
        raise ValueError('Audited native slot coverage changed')
    if {r['execution_id'] for r in native} != {r['id'] for r in audited}:
        raise ValueError('Native metric execution identity changed')
    count, native_added = update_rows(rows, cfm, native,
        (cfm_dir / 'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix(),
        (native_dir / 'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix(), native_audit['captured_utc'])
    if len(rows) != 6923 or count != 237 or native_added != 6:
        raise ValueError('Catalog coverage changed')
    for row in rows:
        n = int(row['n'])
        missing = row['mean'] is None or row['mean'] == ''
        if missing != (n == 0) or n > int(row['required_repeats']):
            raise ValueError('Invalid repeat count or missingness')
        if not missing and not math.isfinite(float(row['mean'])):
            raise ValueError('Nonfinite metric')
    complete = ROOT / config['complete_capture']
    strict = csv_rows(complete / 'strict_summary.csv')
    target.mkdir(parents=True)
    fields = list(rows[0])
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    counts = Counter(r['task'] for r in rows)
    summary = ['# 完整算法—数据集实验技术指标', '',
        '本清单列出全部已登记组合及缺失值。不同任务保留各自核验时间。全论文、全方法、全消融尚未全部完成。', '',
        '[全部6923条汇总指标 CSV](all_algorithm_dataset_metrics.csv)；[全部6923条可阅读指标](all_metrics.md)。', '',
        f"严格异常检测：{len(strict)}个配置—数据集组合，{sum(r['status']=='completed' for r in strict)}个完成预定重复，"
        f"{sum(r['status']=='partial' for r in strict)}个仅完成部分重复，{sum(r['status']=='no_complete_result' for r in strict)}个暂无完整运行。快照为2026-10-09 10:24:49（北京时间），核验958次完整运行。", '',
        f"CFM-TS：{cfm['job_counts']}；核验时间{cfm['captured_utc']}（UTC）。285原始种子槽位、57组比较；精确重跑不增加种子。", '',
        f"GRASP：本次保留4个已完成原生组，并纳入main和Transformer，共6个全实体完成种子组。更新核验时间{native_audit['captured_utc']}（UTC）。main/Transformer在SWAN均只完成1/10种子，不能生成STD或置信区间。", '',
        '概率指标按0–1列出，误差保留原量纲。—表示缺失，0为实测零值。n为该指标实际重复数，范围指标可能少于点级指标。', '',
        '严格主表用验证分数第99百分位阈值，无PA；作者PA、测试标签最优阈值、发布权重评价及历史协议均单列。窗口与更新预算尚未全部对齐，未认证作者等价。PRONTO正常段选择使用标签，SKAB校准含原生异常，TEP为经典单工况整运行。', '',
        '完整逐种子字段还包含事件召回、漏检、检测延迟、误报/千正常点、参数量、训练/评分秒数和CUDA峰值。延迟来自离线窗口分数。确定性统计方法重复拟合不作随机训练置信区间。', '',
        '| 任务 | 汇总指标行数（含缺失） |', '|---|---:|']
    summary += [f'| {task} | {n} |' for task, n in counts.items()]
    older = '../' + complete.name
    summary += ['', '完整明细入口：', '',
        f'- [全部730个异常检测组合的核心指标](strict_algorithm_dataset.md)',
        f'- [按数据集的F1矩阵]({older}/overview.md)',
        f'- [16工作表Excel（严格与插补快照，CFM/GRASP最新更新见本清单）]({older}/complete_experiment_metrics.xlsx)',
        f'- [2874行逐种子严格指标和资源]({older}/strict_per_seed.csv)',
        f'- [42576行逐实体严格指标]({older}/strict_per_entity.csv)',
        f'- [作者完整流程的314行数值字段]({older}/author_pipeline_numeric_metrics.csv)',
        f'- [548行历史协议指标]({older}/historical_protocol_results.csv)',
        f'- [9580行TAB比例诊断与PA对照]({older}/tab_ratio_diagnostic.csv)',
        f'- [173行插补逐次指标]({older}/imputation_per_run.csv)',
        f'- [已核验的71条论文报告值]({older}/paper_claims.csv)',
        f'- [45篇论文的273条原始数据/任务范围及未完成义务]({older}/original_dataset_scope.csv)',
        f'- [CFM-TS全部57组、论文值及差异](../{cfm_dir.name}/README.md)',
        f'- [最新GRASP完整数组独立审计](../{native_dir.name}/independent_array_audit.json)', '',
        '这里的6923行是指标行数，并非6923次训练。作者流程与历史记录未混入严格主表。登记范围以外的原始任务缺口在原文范围清单中明示，不假设已有成绩。', '']
    (target / 'README.md').write_text('\n'.join(summary), encoding='utf-8')
    headers = ['算法配置', '数据集', '完成/预定', '状态', 'P', 'R', 'F1', 'AUROC', 'AP', 'VUS-ROC(n)', 'VUS-PR(n)', 'Affiliation F(n)', 'F1 STD', 'F1 95%CI']
    values = [[r['algorithm_config'], r['dataset'], r['completed_seeds']+'/'+r['required_seeds'], r['status'],
        *[fmt(r[k+'_mean']) for k in ('precision','recall','f1','auroc','average_precision')],
        *[fmt(r[k+'_mean'])+' ('+r[k+'_n']+')' for k in ('VUS_ROC','VUS_PR','affiliation_f')],
        fmt(r['f1_std']), fmt(r['f1_ci95_low'])+' / '+fmt(r['f1_ci95_high'])] for r in strict]
    (target / 'strict_algorithm_dataset.md').write_text('# 全部730个严格异常检测组合\n\n验证分数第99百分位阈值，无PA。概率指标0–1；缺失为—。核验快照2026-10-09 10:24:49（北京时间）。\n\n' + '\n'.join(table(headers, values))+'\n', encoding='utf-8')
    headers = ['任务','算法','数据集','协议','轮数','掩码/缺失率','推断配方','指标','n/预定','均值','STD','SE','95%下界','95%上界','论文值','核验时间UTC']
    values = [[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('epochs',''),
        str(r.get('pattern',''))+'/'+str(r.get('missing_ratio','')),r.get('score_recipe',''),r['metric'],
        str(r['n'])+'/'+str(r['required_repeats']),*[fmt(r.get(k)) for k in ('mean','std','se','ci95_low','ci95_high','paper_mean')],r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text('# 全部汇总技术指标（含全部空缺）\n\n各协议独立解释；数值0–1或原误差量纲。n=0为无完整结果。来源文件和审计哈希见CSV与publication_validation.json。\n\n'+'\n'.join(table(headers,values))+'\n', encoding='utf-8')
    sources = [config_path,base/'publication_validation.json',base/'all_algorithm_dataset_metrics.csv',
        complete/'validation.json',complete/'strict_summary.csv',cfm_dir/'execution_audit.json',
        cfm_dir/'algorithm_dataset_metrics.csv',native_dir/'independent_array_audit.json',native_dir/'algorithm_dataset_metrics.csv']
    validation = dict(published_utc=datetime.now(timezone.utc).isoformat(),aggregate_metric_rows=len(rows),
        measured_metric_rows=sum(int(r['n'])>0 for r in rows), task_rows=dict(counts),
        updated_canonical_metric_rows=count, newly_measured_native_metric_rows=native_added,
        cfm_completed_canonical_slots=cfm['job_counts']['completed'], grasp_completed_native_cohorts=6,
        strict_groups=len(strict), strict_complete_groups=sum(r['status']=='completed' for r in strict),
        strict_partial_groups=sum(r['status']=='partial' for r in strict),all_original_experiments_complete=False,
        checks=dict(no_extra_seed_slots=True,existing_measured_native_results_identical=True,
                    missing_not_zero=True,source_capture_times_preserved=True,protocols_separate=True),
        source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources],
        outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()})
    (target / 'publication_validation.json').write_text(json.dumps(validation,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in validation.items() if k not in ('source_receipts','outputs')},ensure_ascii=False))


if __name__ == '__main__':
    main()
