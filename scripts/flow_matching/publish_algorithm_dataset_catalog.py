"""Combine immutable result captures, retaining protocols, missingness and times."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ('precision', 'recall', 'f1', 'auroc', 'average_precision',
           'VUS_ROC', 'VUS_PR', 'affiliation_f')


def strict_metric_rows(records, source, captured):
    rows = []
    for record in records:
        for metric in METRICS:
            count = record.get(metric + '_n', record['completed_seeds'])
            if count is None:
                count = 0
            rows.append(dict(algorithm=record['algorithm_config'], dataset=record['dataset'],
                             task='anomaly_detection', protocol=record['protocol'],
                             metric=metric, n=count, required_repeats=record['required_seeds'],
                             **{key: record.get(metric + '_' + key)
                                for key in ('mean', 'std', 'se', 'ci95_low', 'ci95_high')},
                             paper_mean=None, author_equivalence_certified=False,
                             captured_utc=captured, source=source))
    return rows


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value):
    return '—' if value is None or value == '' else f'{float(value):.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    target = ROOT / settings['output_directory']
    if target.exists():
        raise FileExistsError('Keep previous published captures unchanged')
    base = ROOT / settings['complete_capture']
    dynamic = ROOT / settings['continuous_time_capture']
    report = json.loads((base / 'complete_results.json').read_text(encoding='utf-8'))
    cfm = json.loads((dynamic / 'execution_audit.json').read_text(encoding='utf-8'))
    grasp = json.loads((dynamic / 'grasp_execution_audit.json').read_text(encoding='utf-8'))
    rows = strict_metric_rows(report['strict_summary'],
                              (base / 'strict_summary.csv').relative_to(ROOT).as_posix(),
                              report['tsad_capture_utc'])
    for name, records in [('imputation', report['imputation']['formal_results']),
                          ('native_statistics', report['native_summary']),
                          ('GRASP', read_csv(dynamic / 'grasp_algorithm_dataset_metrics.csv'))]:
        for record in records:
            imp = name == 'imputation'
            row = dict(algorithm=record.get('recipe', record['algorithm']), dataset=record['dataset'],
                       task='imputation' if imp else 'anomaly_detection',
                       protocol=record.get('track', '') if imp else record['protocol'],
                       pattern=record.get('pattern'), missing_ratio=record.get('missing_ratio'),
                       score_recipe=record.get('score_recipe'), metric=record['metric'],
                       n=record['completed_repeats'] if imp else record['n'],
                       required_repeats=record['required_repeats'] if imp else record['required_seeds'],
                       mean=record.get('local_mean') if imp else record['mean'],
                       std=record.get('local_std') if imp else record['std'],
                       se=record.get('local_se') if imp else record['se'],
                       ci95_low=record.get('ci95_low'), ci95_high=record.get('ci95_high'),
                       paper_mean=record.get('paper_mean'),
                       comparison=record.get('verdict'), author_equivalence_certified=False,
                       captured_utc=grasp['capture']['captured_utc'] if name == 'GRASP' else report['captured_utc'],
                       source=(dynamic / 'grasp_algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix()
                       if name == 'GRASP' else (base / ('imputation_summary.csv' if imp else 'native_summary.csv')).relative_to(ROOT).as_posix())
            rows.append(row)
    for record in cfm['seed_aggregates']:
        for metric in ('MSE', 'MAE', 'RMSE'):
            rows.append(dict(algorithm='CFM-TS/' + record['variant'], dataset=record['dataset'],
                             task='continuous_time_dynamics', protocol=record['track'], epochs=record['epochs'],
                             metric=metric, n=record[metric+'_n'], required_repeats=record['required_seeds'],
                             **{key: record[metric+'_'+key] for key in ('mean','std','se','ci95_low','ci95_high')},
                             paper_mean=record['paper_MSE_mean'] if metric == 'MSE' else None,
                             paper_std=record['paper_MSE_std'] if metric == 'MSE' else None,
                             comparison=record['comparison'], author_equivalence_certified=False,
                             captured_utc=cfm['captured_utc'], source=(dynamic/'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix()))
    generation = None
    if settings.get('generation_capture'):
        generation_path=ROOT/settings['generation_capture']/'execution_audit.json'
        generation=json.loads(generation_path.read_text(encoding='utf-8'))
        for record in generation['summaries']:
            rows.append(dict(algorithm='Spectral-author/'+record['model'],dataset=record['dataset'],
                task='unconditional_time_series_generation',protocol=record['data_track'],metric=record['metric'],
                n=record['n'],required_repeats=record['required_seeds'],
                **{key:record[key] for key in ('mean','std','se','ci95_low','ci95_high')},
                paper_mean=record['paper_mean'],paper_spread=record['paper_spread'],
                comparison='unavailable' if not record['n'] else 'numerical_comparison_only',author_equivalence_certified=False,
                captured_utc=generation['captured_utc'],source=(generation_path.parent/'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix()))
    target.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (target/'all_algorithm_dataset_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    old = '../' + base.name
    new = '../' + dynamic.name
    count=sum(r['completed_seeds'] > 0 for r in report['strict_summary'])
    lines=['# 全部算法—数据集实验技术指标', '',
           f"严格异常检测快照：{report['tsad_capture_utc']}（UTC）；连续时间快照：{cfm['captured_utc']}（UTC）；GRASP快照：{grasp['capture']['captured_utc']}（UTC）。", '',
           f"异常检测登记730个组合，{count}个已有指标；完整运行{report['summary']['complete_tsad_runs']}次。插补登记540项，完成{report['summary']['imputation']['completed_jobs']}项。CFM-TS登记285项，快照状态{cfm['job_counts']}。GRASP登记9120个实体任务，完成{grasp['capture']['completed_cohorts']}个全实体种子组。", '',
           '所有已登记组合和空缺均列出；全论文、全方法及全消融尚未全部完成。—为缺失，实测零值保留为0。表内概率指标为0–1，误差保留原量纲。各轨道独立保留协议和采集时间。', '',
           f"[全部汇总指标CSV](all_algorithm_dataset_metrics.csv)；[完整16工作表Excel（严格轨同一快照）]({old}/complete_experiment_metrics.xlsx)；[逐种子指标、事件/延迟/误报、区间和资源]({old}/strict_per_seed.csv)；[完整作者流程指标]({old}/author_pipeline_numeric_metrics.csv)；[完整历史记录]({old}/historical_protocol_results.csv)。", '',
           '严格主表使用验证分数第99百分位阈值，无PA；TAB比例诊断、作者PA、测试最优阈值、统计方法测试特征拟合、发布权重评价分别解释。训练窗口与更新预算尚未与所有作者配方对齐，不能据低分直接否定论文。', '',
           '## 严格异常检测：全部730个算法配置—数据集组合', '',
           '| 算法配置 | 数据集 | 完成/预定 | P | R | F1 | AUROC | AP | VUS-ROC (n) | VUS-PR (n) | Affiliation F (n) | F1 STD | F1 95% CI |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in report['strict_summary']:
        values=[fmt(r[k+'_mean']) for k in METRICS[:5]]
        values += [fmt(r[k+'_mean'])+f" ({r[k+'_n'] or 0})" for k in METRICS[5:]]
        lines.append(f"| {r['algorithm_config']} | {r['dataset']} | {r['completed_seeds']}/{r['required_seeds']} | "+' | '.join(values)+f" | {fmt(r['f1_std'])} | {fmt(r['f1_ci95_low'])} – {fmt(r['f1_ci95_high'])} |")
    lines += ['', '## 插补、原生统计方法及作者流程', '',
              f"全部插补组合及论文比较见[328条指标表]({old}/imputation_summary.csv)，包括MAE、RMSE、CRPS、MRE、缺失率、掩码、种子/折、STD/SE/区间及比较原因。", '',
              f"[HBOS/COPOD全部实测指标]({old}/native_summary.csv)；[全部实体指标]({old}/native_entity_metrics.csv)；[作者流程1429项状态]({old}/author_pipeline_jobs.csv)。", '',
              '## GRASP：全预算训练及消融', '',
              f"[全部504条算法—数据集—推断配方—协议指标及空缺]({new}/grasp_algorithm_dataset_metrics.csv)；[检查点、完整数组和独立指标复算审计]({new}/grasp_execution_audit.json)。", '',
              '| 消融 | 数据集 | 推断配方 | 协议 | 指标 | 均值 | STD | 完成/预定种子 |',
              '|---|---|---|---|---|---:|---:|---:|']
    for r in read_csv(dynamic/'grasp_algorithm_dataset_metrics.csv'):
        if int(r['n']):
            lines.append(f"| {r['recipe']} | {r['dataset']} | {r['score_recipe']} | {r['protocol']} | {r['metric']} | {fmt(r['mean'])} | {fmt(r['std'])} | {r['completed_seeds']}/{r['required_seeds']} |")
    lines += ['', '## CFM-TS：全部57组原始连续时间实验', '',
              f"[完整指标、训练耗时、参数量及论文差值]({new}/algorithm_dataset_metrics.csv)；[285项逐次任务及状态]({new}/registered_jobs.csv)；[完整数组、轨道和实现边界审计]({new}/execution_audit.json)。", '',
              '| 协议 | 数据集 | 方法/公式 | 轮数 | 完成/预定 | MSE | STD | MAE | RMSE | 论文MSE |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in cfm['seed_aggregates']:
        lines.append(f"| {r['track']} | {r['dataset']} | {r['variant']} | {r['epochs']} | {r['completed_seeds']}/{r['required_seeds']} | {fmt(r['MSE_mean'])} | {fmt(r['MSE_std'])} | {fmt(r['MAE_mean'])} | {fmt(r['RMSE_mean'])} | {fmt(r['paper_MSE_mean'])} |")
    lines += ['', '五种子数值相近不等于等价证明。CFM-TS原作者代码与原始随机轨迹未公开，正文/附录计数冲突、网络解释、ODE容差等本地选择已冻结。Pendulum只有一条物理轨迹，结果对应新时间点评价。', '',
              f"[45篇原始实验范围和缺口]({old}/original_dataset_scope.csv)；[已核验论文声称值]({old}/paper_claims.csv)。未执行的预测、生成、图像、单细胞、表格等原始实验不能从异常检测迁移结果推定成绩。", '']
    if generation is not None:
        relative='../'+Path(settings['generation_capture']).name
        lines += ['', '## Spectral Mean Flow / Diffusion-TS：原始生成任务', '',
            f"登记14组、70次完整训练；捕获时间{generation['captured_utc']}；状态{generation['job_counts']}。四个原始评价器：Context-FID、Correlational、Discriminative、Predictive。预检不计成绩。", '',
            f"[全部56条指标及论文值]({relative}/algorithm_dataset_metrics.csv)；[70项逐次状态]({relative}/registered_jobs.csv)；[环境、协议及剩余原论文实验]({relative}/README.md)。", '']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[base/'complete_results.json',base/'strict_summary.csv',dynamic/'execution_audit.json',dynamic/'grasp_execution_audit.json',dynamic/'grasp_algorithm_dataset_metrics.csv']
    if generation is not None:paths.append(generation_path)
    receipt=dict(published_utc=datetime.now(timezone.utc).isoformat(), aggregate_metric_rows=len(rows),
                 strict_groups=len(report['strict_summary']), cfm_cohorts=len(cfm['seed_aggregates']),
                 grasp_complete_cohorts=grasp['capture']['completed_cohorts'],
                 source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in paths],
                 checks=dict(missing_values_kept=True,protocols_separate=True,source_capture_times_retained=True,
                             full_strict_combination_count=len(report['strict_summary'])==730,
                             full_cfm_cohort_count=len(cfm['seed_aggregates'])==57),
                 outputs={p.name:digest(p) for p in target.iterdir() if p.is_file()})
    (target/'publication_validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['aggregate_metric_rows','strict_groups','cfm_cohorts','grasp_complete_cohorts']}))


if __name__ == '__main__':
    main()
