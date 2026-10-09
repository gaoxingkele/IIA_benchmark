"""Readable algorithm-by-dataset matrix for one immutable metrics capture."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    target = ROOT / args.source
    data = json.loads((target / 'complete_results.json').read_text(encoding='utf-8'))
    rows = [r for r in data['strict_summary'] if r['completed_seeds']]
    models = sorted({r['algorithm_config'] for r in rows})
    datasets = sorted({r['dataset'] for r in rows})
    index = {(r['algorithm_config'], r['dataset']): r for r in rows}
    full = sum(r['status'] == 'completed' for r in rows)
    lines = ['# 算法—数据集实验结果总览', '',
             f"异常检测源快照：{data['tsad_capture_utc']}（UTC）；导出完成：{data['captured_utc']}（UTC）。", '',
             f"登记 {len(data['strict_summary'])} 个异常检测组合，{len(rows)} 个已有数值，其中 {full} 个完成预定重复，{len(rows)-full} 个仅部分重复，{len(data['strict_summary'])-len(rows)} 个暂无完整结果。核验 {data['summary']['complete_tsad_runs']} 次完整运行，其中 {data['summary']['TAB_range_complete_runs']} 次有 TAB 范围指标。", '',
             '[完整 Excel](complete_experiment_metrics.xlsx)；[所有已有指标的 P/R/F1、AUROC、AP、VUS、Affiliation](README.md)；[全部组合含空缺](strict_summary.csv)；[逐种子指标、区间和资源](strict_per_seed.csv)。', '',
             '以下点级 F1 为百分数，验证分数第99百分位定阈值，不做 PA。数值来自已完成种子的均值；n=x/y 表示部分重复，MOMENT 1/1 是固定权重的一次评价。— 表示暂无完整运行结果。名称按冻结配置保留，未合并不同预算或不同消融。']
    common = ['PSM', 'SMAP', 'MSL', 'SMD', 'SWAT', 'SWAN', 'CICIDS']
    for title, columns in [('常用异常检测数据集', [d for d in common if d in datasets]),
                           ('工业与过程数据集', [d for d in datasets if d not in common])]:
        lines += ['', '## ' + title, '', '| 算法配置 | ' + ' | '.join(columns) + ' |',
                  '|---|' + '---:|' * len(columns)]
        for model in models:
            values = []
            for dataset in columns:
                row = index.get((model, dataset))
                value = '—' if row is None else f"{100*row['f1_mean']:.2f}"
                if row and (row['completed_seeds'] != row['required_seeds'] or row['required_seeds'] == 1):
                    value += f" [n={row['completed_seeds']}/{row['required_seeds']}]"
                values.append(value)
            lines.append('| ' + model + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 原生统计协议', '',
              '测试特征用于拟合，测试标签选最优 F1，无 PA。实体宏平均。十次确定性拟合不报告随机训练置信区间。不能与上表严格阈值成绩混排。', '',
              '| 方法 | 数据集 | 完成/预定 | 最优 F1 (%) | AUROC | AP |', '|---|---|---:|---:|---:|---:|']
    native = {(r['recipe'],r['dataset'],r['metric']):r for r in data.get('native_summary', [])
              if r['algorithm']=='grasp_mtsbench_statistical_replay'}
    for method in ['HBOS','COPOD']:
        for dataset in ['swan','smd','smap','cicids']:
            items = [native.get((method,dataset,k)) for k in ['Best_F1','ROC','PRC']]
            f = items[0]
            if not f:
                continue
            values = ['—' if r['mean'] is None else f"{r['mean']*(100 if i==0 else 1):.6f}"
                      for i,r in enumerate(items)]
            lines.append(f"| {method} | {dataset.upper()} | {f['n']}/{f['required_seeds']} | " + ' | '.join(values) + ' |')
    imp = data['imputation']['summary']
    grasp = data.get('grasp_full_protocol') or {}
    lines += ['', '## 其他结果与未完成范围', '',
              f"- 插补完成 {imp['completed_jobs']}/{imp['registered_jobs']} 项，全部缺失率、掩码、MAE/RMSE/CRPS/MRE、种子和论文差值见[插补表](imputation_summary.csv)。",
              f"- 作者完整流程状态：{data['summary']['author_pipeline_counts']}；[指标](author_pipeline_numeric_metrics.csv)和[任务状态](author_pipeline_jobs.csv)单列。",
              f"- 附加原生队列状态：{dict(Counter(r['status'] for r in data['additional_native_paper_jobs']))}；[全部任务](additional_native_paper_jobs.csv)。",
              f"- GRASP 完整训练登记 {grasp.get('registered_entity_jobs',0)} 个实体任务、{grasp.get('registered_cohorts',0)} 个全实体组，完成 {grasp.get('completed_cohorts',0)} 组。全部结构消融与推断配方、指标及空缺见[GRASP 完整表](grasp_full_summary.csv)。",
              '- MaelNet、Pi-Transformer、DT-LA、SHCL、KGL、CrossAD 等尚未在本次严格主表形成完整结果，论文值不能替代本地成绩。',
              '- [论文报告值](paper_claims.csv)和[45篇论文原文数据范围/缺口](original_dataset_scope.csv)单列。清单覆盖已登记实验与已保存指标，尚非全部论文、全部消融均已完成。',
              '- 训练窗口、梯度更新预算与部分原作者 stride=1 尚不一致，GRASP 作者代码与局部实现选择尚未等价认证。Independent CFM 与单阶段 Rectified 在本地 sigma=0 配置下等价；SF2M flow-only 是局部消融。',
              '- 工业迁移：TEP为经典单工况整运行，SKAB校准包含异常，PRONTO正常段选择使用标签且整日轮换相关。范围指标与点级 F1 不可相互替代。', '']
    (target / 'overview.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'numeric_groups':len(rows),'full_groups':full,'configurations':len(models),'datasets':len(datasets)}))


if __name__ == '__main__':
    main()
