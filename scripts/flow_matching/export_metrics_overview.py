"""Readable, exhaustive F1 matrices from an immutable complete metrics export."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

NAMES = {
    'fm_objective_independent': 'Independent CFM',
    'fm_objective_ot': 'OT-CFM',
    'fm_objective_rectified': '单阶段 Rectified',
    'fm_objective_target': 'Target CFM',
    'fm_objective_vp': 'VP-CFM',
    'fm_objective_sb_stable_v2': 'SB 稳定版 v2',
    'fm_objective_sf2m_flow_only_stable_v2': 'SF2M 仅流消融 v2',
    'fm_objective_sf2m_stable_v2': 'SF2M 稳定版 v2',
    'fm_rectified_40epoch_budget_control': 'Rectified 40轮对照',
    'fm_rectified_60epoch_budget_control': 'Rectified 60轮对照',
    'fm_reflow_2stage': 'Reflow 两阶段',
    'fm_reflow_3stage': 'Reflow 三阶段',
    'fm_strict_anomaly_transformer__psm__registered': 'Anomaly Transformer（PSM配方）',
    'fm_strict_anomaly_transformer__smap__registered': 'Anomaly Transformer（SMAP配方）',
    'fm_strict_dcdetector__psm__registered': 'DCdetector（PSM配方）',
    'fm_strict_dcdetector__smap__registered': 'DCdetector（SMAP配方）',
    'fm_strict_itransformer__psm__registered': 'iTransformer（PSM配方）',
    'fm_strict_timesnet__psm__registered': 'TimesNet（PSM配方）',
    'fm_strict_timesnet__smap__registered': 'TimesNet（SMAP配方）',
    'fm_strict_tranad__psm__registered': 'TranAD（PSM配方）',
    'fm_strict_usad__psm__registered': 'USAD（登记损失）',
    'fm_strict_usad__psm__signed_paper_loss': 'USAD（论文有符号损失）',
    'fm_moment_pretrained_small__release512': 'MOMENT-small（预训练512）',
    'fm_moment_pretrained_small__TAB100': 'MOMENT-small（TAB窗口100）',
    'fm_strict_anomaly_transformer__msl__registered': 'Anomaly Transformer（MSL配方）',
    'fm_strict_dcdetector__msl__registered': 'DCdetector（MSL配方）',
    'fm_strict_itransformer__smap__registered': 'iTransformer（SMAP配方）',
    'fm_strict_timesnet__msl__registered': 'TimesNet（MSL配方）',
    'fm_strict_tranad__msl__registered': 'TranAD（MSL配方）',
    'fm_strict_usad__msl__registered': 'USAD（MSL配方）',
    'fm_strict_usad__smap__registered': 'USAD（SMAP配方）',
    'fm_strict_usad__smap__signed_paper_loss': 'USAD（SMAP论文有符号损失）',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    target = Path(args.source)
    report = json.loads((target / 'complete_results.json').read_text(encoding='utf-8'))
    rows = report['strict_summary']
    numeric = [r for r in rows if r['completed_seeds']]
    models = list(dict.fromkeys([a for a in NAMES if any(r['algorithm_config'] == a for r in numeric)]
                               + sorted({r['algorithm_config'] for r in numeric} - NAMES.keys())))
    index = {(r['algorithm_config'], r['dataset']): r for r in numeric}
    full = sum(r['status'] == 'completed' for r in rows)
    stamp = datetime.fromisoformat(report['tsad_capture_utc']).astimezone(timezone(timedelta(hours=8)))
    stats = report['summary']
    lines = ['# 算法—数据集实验指标总览', '', f"异常检测快照：{stamp:%Y-%m-%d %H:%M:%S}（北京时间）。", '',
             f"{len(rows)}个登记组合：{len(numeric)}个已有数值，{full}个完成预定重复，{len(numeric)-full}个完成部分重复，{len(rows)-len(numeric)}个暂无完整结果。核验{stats['complete_tsad_runs']}次完整运行，{stats['TAB_range_complete_runs']}次已有TAB范围指标。", '',
             '[完整 Excel](complete_experiment_metrics.xlsx)；[全部技术指标长表](README.md)；[所有组合及状态](strict_summary.csv)；[逐种子技术指标](strict_per_seed.csv)。', '',
             '下表为点级F1百分数（0–100）：验证分数第99百分位阈值，无PA。[n=数值]表示部分重复或固定预训练权重单次评价。—表示暂无完整运行结果；0.00是实测零值。', '',
             '本地非重叠训练与作者stride=1预算尚未等效认证；作者阈值、PA-F1、Affiliation F和VUS分别解释。']
    displayed = set()
    sections = [('常用与扩展数据集', ['PSM', 'SMAP', 'MSL', 'SMD', 'SWAT', 'SWAN', 'CICIDS']),
                ('工业与过程数据集', ['SKAB', 'TEP_CLASSIC', 'PRONTO_DAY2', 'PRONTO_DAY3', 'PRONTO_DAY4'])]
    extra = sorted({r['dataset'] for r in numeric} - {d for _, cols in sections for d in cols})
    if extra:
        sections.append(('其他数据集', extra))
    for title, columns in sections:
        lines += ['', '## ' + title, '', '| 算法配置 | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
        for model in models:
            cells = []
            for dataset in columns:
                row = index.get((model, dataset))
                value = '—' if row is None else f"{100 * row['f1_mean']:.2f}"
                if row:
                    displayed.add((model, dataset))
                    if row['completed_seeds'] < row['required_seeds'] or row['required_seeds'] == 1:
                        value += f" [n={row['completed_seeds']}]"
                cells.append(value)
            lines.append('| ' + NAMES.get(model, model) + ' | ' + ' | '.join(cells) + ' |')
    assert displayed == set(index), 'Numerical combination omitted from overview'
    lines += ['', 'TEP是经典单工况整运行；SKAB校准含原生异常；PRONTO正常段选择使用标签且整日轮换相关。基线PSM配方迁移尚未目标调优。', '',
              '## 其他实验及完整状态', '',
              f"- 作者流程共{stats['author_pipeline_registered_jobs']}项，状态{stats['author_pipeline_counts']}；[作者实际技术指标](author_pipeline_numeric_metrics.csv)与[逐项状态](author_pipeline_jobs.csv)单列。发布权重评价不等于重新训练复现。",
              f"- 插补共540项，已有{len(report['imputation']['per_run_metrics'])}条逐次技术指标；MAE/RMSE/CRPS/MRE、重复数、论文差值见[插补汇总](imputation_summary.csv)。",
              f"- 历史{stats['historical_runs']}次运行、{stats['historical_protocol_rows']}条协议记录见[历史结果](historical_protocol_results.csv)。",
              '- DT-LA、SHCL、KGL等无当前严格主表完整成绩的组合保留缺口，论文值不能替代本地实测。',
              '- [已核验论文报告值](paper_claims.csv)和[45篇原始实验范围](original_dataset_scope.csv)保留原文证据及复现缺口。', '',
              '## 配置名称对应', '', '| 表内名称 | 模型配置名称 |', '|---|---|']
    if report.get('additional_native_paper_jobs'):
        native = report['additional_native_paper_jobs']
        done = sum(r['status'] == 'completed' for r in native)
        lines += [f'| 原生GiFlow作者镜像/已审修正 | 另登记{len(native)}项主方法及消融，{done}项完整结果；见additional_native_paper_jobs.csv |']
    lines += ['| ' + NAMES.get(a, a) + ' | ' + a + ' |' for a in models]
    (target / 'overview.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'numerical_combinations': len(numeric), 'complete_repeats': full,
                      'partial_repeats': len(numeric)-full, 'no_complete_result': len(rows)-len(numeric),
                      'covered_by_matrices': len(displayed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
