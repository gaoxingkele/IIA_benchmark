"""Publish the complete frozen metric listing without mixing evaluation protocols."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


def fmt(value):
    return '—' if value is None else f'{value:.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    directory = Path(args.source)
    report = json.loads((directory / 'complete_results.json').read_text(encoding='utf-8'))
    rows = report['strict_summary']
    numeric = sum(r['completed_seeds'] > 0 for r in rows)
    full = sum(r['status'] == 'completed' for r in rows)
    stamp = datetime.fromisoformat(report['tsad_capture_utc']).astimezone(timezone(timedelta(hours=8)))
    summary = report['summary']
    lines = ['# 算法与数据集完整实验结果清单', '',
             f'严格异常检测快照：{stamp:%Y-%m-%d %H:%M:%S}（北京时间）。其他轨道的捕获时间见 complete_results.json。', '',
             f"异常检测登记 **{len(rows)} 个配置—数据集组合**：**{numeric} 个已有指标**，其中 **{full} 个完成预定重复**、**{numeric-full} 个完成部分重复**；**{len(rows)-numeric} 个暂无完整运行**。共核验 **{summary['complete_tsad_runs']} 次完整运行**，其中 **{summary['TAB_range_complete_runs']} 次已核验 TAB 范围指标**。", '',
             '[完整 Excel](complete_experiment_metrics.xlsx)；[全部组合及状态](strict_summary.csv)；[逐种子指标与资源](strict_per_seed.csv)；[按数据集的 F1 矩阵](overview.md)。', '',
             '数值按 0–1 列出，插补保留原量纲。— 表示缺失，不表示实测零值。不同协议、训练预算或数据身份的值不能直接排名。', '',
             '技术字段包括 Precision、Recall、点级 F1、AUROC、AP、VUS-ROC、VUS-PR、Affiliation、事件召回、漏检、检测延迟、每千正常点误报、种子 STD/SE/95% 区间、参数量、训练/评分秒数和 CUDA 峰值。逐种子表保留三档验证分位数，插补另列 MAE/RMSE/CRPS/MRE、缺失率和掩码。', '',
             '主表为验证分数第 99 百分位阈值，无 PA。本地非重叠训练尚未与作者 stride=1 的更新次数对齐，作者等价未认证。确定性 HBOS/COPOD 的十次重新拟合不作随机训练置信区间。', '',
             '下列所有已有实测值来自冻结结果，论文声称值单列于 [论文报告值](paper_claims.csv)。45 篇论文的原始任务、数据范围和未运行义务见 [原文数据范围](original_dataset_scope.csv)。全论文、全方法和全消融实验尚未全部完成。', '',
             '---', '', (directory / 'README.md').read_text(encoding='utf-8').strip()]
    lines += ['', '## 已完成作者流程的核心指标', '',
              '以下结果不混入严格窗口主表。Pi 为修正轴语义后的历史镜像、单种子完整训练，尚未认证对应期刊版；CrossAD 为发布权重评价，未重新训练。全部字段与来源见 [作者流程数值](author_pipeline_numeric_metrics.csv)。', '',
              '| 算法/配方 | 数据集 | 种子 | 协议 | Precision | Recall | F1 | AUROC | AP | VUS-ROC | VUS-PR | Affiliation F |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def append(job, protocol, metrics, ranges=None):
        ranges = ranges or {}
        values = [metrics.get(k) for k in ['precision', 'recall', 'f1', 'auroc', 'average_precision']]
        values += [ranges.get(k) for k in ['VUS_ROC', 'VUS_PR', 'Affiliation_F1']]
        lines.append(f"| {job['algorithm']} / {job['author_recipe']} | {job['dataset']} | {job['seed']} | {protocol} | " + ' | '.join(map(fmt, values)) + ' |')
    for job in report['author_pipeline_jobs']:
        if job['status'] != 'completed':
            continue
        metrics = job['metrics']
        if job['algorithm'] == 'pi_transformer_historical_mirror':
            append(job, '作者阈值 + PA', metrics)
            append(job, '相同作者阈值，无 PA', metrics['pointwise_no_PA_same_author_threshold'])
        elif job['algorithm'] == 'crossad_complete':
            original = metrics['original_author']
            best, auc = original['_best_f1'][0], original['_auc'][0]
            converted = {'precision': best['P_best'], 'recall': best['R_best'], 'f1': best['F1_best'],
                         'auroc': auc['AUC_ROC'], 'average_precision': auc['AUC_PR']}
            ranges = dict(original['_vus'][0], **original['_affiliation'][0])
            append(job, '测试标签最优点级 F1；范围/关联指标为作者原流程', converted, ranges)
            append(job, '验证分数第 99 百分位，无 PA', metrics['native_validation_control']['strict']['1']['micro'])
    if not any('| 模型/消融 | 数据集 | 推断配方 | 协议 | 指标 |' in s for s in lines):
        lines += ['', '## GRASP 完整训练已完成值', '',
                  '以下全实体完整运行仍只完成部分种子；未完成实体不产生数据集指标。各消融及推断配方的完整空缺表见 [GRASP 汇总](grasp_full_summary.csv)。', '',
                  '| 模型/消融 | 数据集 | 推断配方 | 协议 | 指标 | 均值 | STD | 完成/预定种子 |',
                  '|---|---|---|---|---|---:|---:|---:|']
        for row in report['grasp_full_summary']:
            if row['n']:
                lines.append(f"| {row['recipe']} | {row['dataset']} | {row['score_recipe']} | {row['protocol']} | {row['metric']} | {fmt(row['mean'])} | {fmt(row['std'])} | {row['completed_seeds']}/{row['required_seeds']} |")
    (directory / 'algorithm_dataset_metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'groups': len(rows), 'numeric': numeric, 'full': full, 'partial': numeric-full,
                      'no_complete_result': len(rows)-numeric}, ensure_ascii=False))


if __name__ == '__main__':
    main()
