"""Publish current complete GRASP cohorts while preserving all other captures."""
import argparse
import copy
from datetime import datetime, timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT, legacy = previous.ROOT, previous.legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT/args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Pinned result publication input changed')
    base, capture, target = (ROOT/config[k] for k in ('base_capture','audited_capture','output_directory'))
    if target.exists():
        raise FileExistsError('Keep every publication immutable')
    receipt = previous.verified_capture(base,'publication_validation.json')
    proof = previous.verified_capture(capture,'validation.json')
    original = previous.load_rows(base/'all_algorithm_dataset_metrics.csv')
    updates = previous.load_rows(capture/'algorithm_dataset_metrics.csv')
    if len(updates) != config['expected_replaced_metric_rows'] or any(not r['algorithm'].startswith('GRASP/') for r in updates):
        raise ValueError('Only the registered original GRASP replacement scope is allowed')
    rows = previous.merge_audited_rows(original,updates)
    if any(a!=b for a,b in zip(original,rows) if not a['algorithm'].startswith('GRASP/')):
        raise ValueError('All non-GRASP captures must remain exactly unchanged')
    target.mkdir(parents=True)
    legacy.csv_write(target/'all_algorithm_dataset_metrics.csv',rows)
    for name in ('strict_algorithm_dataset.md','overview.md','original_generation_jobs.csv'):
        shutil.copyfile(base/name,target/name)
    headers = ['任务','算法／消融','数据集','协议','评分配方','指标','完成／预定','均值','STD','SE','95%下界','95%上界','论文值','核验时间UTC']
    values = [[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('score_recipe',''),r['metric'],
               f"{r['n']}/{r['required_repeats']}",*[legacy.fmt(r.get(k)) for k in
               ('mean','std','se','ci95_low','ci95_high','paper_mean')],r['captured_utc']] for r in rows]
    (target/'all_metrics.md').write_text('# 全部算法—数据集指标及缺口\n\n各协议分别列出；—为缺失，0为实测零。完整来源、预算和掩码等字段见CSV。\n\n'+legacy.table(headers,values)+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    measured = sum(r['mean'] is not None for r in rows)
    now = datetime.now(timezone.utc).isoformat()
    (target/'README.md').write_text(
        '# 完整算法—数据集实验指标\n\n'+f'发布：{now}。本版只更新GRASP原始完整训练结果；其他轨道保留原快照时间。\n\n'
        +f'**{len(rows)}条指标记录：{measured}条有实测值，{len(rows)-measured}条缺结果。指标行数不等于训练次数。全部原论文实验仍未完成。**\n\n'
        +'[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整可读表](all_metrics.md)；[全部严格异常检测F1矩阵](overview.md)。\n\n'
        +f"GRASP现在有{proof['completed_cohorts']}/480个全实体种子组完成。新完成tau0.25/SWAN/seed1103：1500轮、93000次更新、3935验证点、72000测试点。\n\n"
        +'作者回顾性无PA Best-F1为53.0506%；验证第99百分位阈值、无PA的对照F1为52.9858%，Precision为38.7339%、Recall为83.8308%；原生PRC（源码字段AP）为0.520136、AUROC为0.721708。这里只完成1/10种子，未给单种子STD或置信区间。\n\n'
        +'保存分数对应的训练预算、模型和数据绑定已核验；该新训练模型的完整独立重推断正在执行。作者回顾性测试标签阈值与验证阈值对照分开，未取得作者等价或严格TAB认证。\n\n'
        +'[最新GRASP规范槽位](../grasp_canonical_listing_v6/grasp_execution_audit.json)；[USHCN完整训练接入和检查点状态](../ls4_native_ushcn_training_execution_v1/README.md)；[之前的完整协议说明](../all_algorithm_dataset_metrics_v10/README.md)。\n\n'
        +'USHCN新接入的100次完整500轮任务和20个完整GPU容量案例尚无500轮结果，不以注册、数据审计或容量诊断填性能分数。此前Pi和GiFlow结果沿用已核验快照；各行来源与时间均保留。\n',encoding='utf-8')
    validation = copy.deepcopy(receipt)
    validation.update(published_utc=now,base_publication_utc=receipt['published_utc'],
        aggregate_metric_rows=len(rows),measured_metric_rows=measured,missing_metric_rows=len(rows)-measured,
        replaced_metric_rows=len(updates),newly_measured_metric_rows=measured-receipt['measured_metric_rows'],
        grasp_complete_cohorts=proof['completed_cohorts'],other_tracks_recaptured=False,
        newly_completed_tau025_model_reinference_complete=False,all_original_experiments_complete=False)
    validation['source_receipts'] += config['source_receipts']+[dict(path=args.config,sha256=legacy.sha(ROOT/args.config))]
    validation['outputs'] = {p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',validation)
    print({k:validation[k] for k in ('aggregate_metric_rows','measured_metric_rows','missing_metric_rows','newly_measured_metric_rows','grasp_complete_cohorts')})


if __name__ == '__main__':
    main()
