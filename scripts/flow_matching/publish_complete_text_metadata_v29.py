"""Preserve all v28 numeric results and expose every metric field in text."""
import argparse
import csv
from datetime import datetime,timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT,legacy=previous.ROOT,previous.legacy


def complete_text(fields,rows):
    for name in previous.KEY_FIELDS:
        if name not in fields:
            raise ValueError('Every scientific experiment identity column required')
    labels=dict(task='任务',algorithm='算法/消融',dataset='数据集',protocol='协议',metric='指标',
        n='完成训练种子',required_repeats='计划训练种子',mean='均值',std='训练种子STD',se='训练种子SE',
        ci95_low='训练种子95%下界',ci95_high='训练种子95%上界',paper_mean='论文值',
        epochs='轮数',pattern='掩码类型',missing_ratio='缺失率',score_recipe='推断配方',
        evaluator_repeats='评价重复数',evaluator_std='评价STD',evaluator_se='评价SE',evaluator_ci95='评价95%区间',
        source='证据来源',captured_utc='采集时间UTC',metric_unit='指标单位',selection_boundary='选择/协议边界',
        uncertainty_scope='不确定性口径',author_equivalence_certified='作者等价确认',strict_TAB_result='完整TAB确认')
    values=[[r[name] if r[name]!='' else '—' for name in fields] for r in rows]
    return ('# 完整算法—数据集实验技术指标 v29\n\n'
        '完整保留CSV全部字段，包括轮数、掩码类型、缺失率、推断配方、指标单位、'
        '评价不确定性及证据来源。数值和每行时间戳与v28一致；v28新增的四项指标仍仅为'
        '一个完整Diffusion-TS/Stocks生成器训练种子（1/5），五次评价不是五次生成器训练。'
        'SMAP修正重跑与其余登记实验未全部完成，—表示缺结果。\n\n'+legacy.table([labels.get(n,n) for n in fields],values)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=legacy.read(ROOT/args.config)
    for pin in config['source_receipts']:
        if legacy.sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full metadata publication input changed')
    base=ROOT/config['base_publication']
    receipt=previous.verified_capture(base,'publication_validation.json')
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        fields,rows=reader.fieldnames,list(reader)
    text=complete_text(fields,rows)
    target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable full metric inventories')
    target.mkdir(parents=True)
    for name in receipt['outputs']:
        if name not in ['all_metrics.md','README.rst']:
            shutil.copyfile(base/name,target/name)
    (target/'all_metrics.md').write_text(text,encoding='utf-8')
    readme=(base/'README.rst').read_text(encoding='utf-8').replace('完整算法—数据集指标 v28','完整算法—数据集指标 v29',1)
    readme+=('\n\n本版数值、HTML、原始生成任务、异常检测及能量任务导出与v28逐字节一致。'
        '完整文字表增加CSV所有字段：轮数、掩码、缺失率、推断配方、单位、不确定性和证据来源，'
        '避免原文字视图将不同实验配置显示为同一组合。保留v28全部派生工件；未新增指标或训练种子。\n')
    (target/'README.rst').write_text(readme,encoding='utf-8')
    updated=dict(receipt)
    updated.update(published_utc=datetime.now(timezone.utc).isoformat(),base_publication=config['base_publication'],
        base_publication_utc=receipt['published_utc'],base_receipt_sha256=legacy.sha(base/'publication_validation.json'),
        changed_numeric_metric_rows=0,replaced_metric_rows=0,newly_measured_metric_rows=0,
        newly_completed_canonical_training_slots=0,changed_row_positions=[],changed_generation_job_rows=0,
        cumulative_native_Stocks_metric_increment_since_v27=4,
        all_numeric_tables_byte_identical_to_base=True,base_numeric_csv_prefix_byte_identical=True,
        performance_rows_changed=False,all_CSV_fields_in_complete_text=True,
        source_receipts=receipt['source_receipts']+config['source_receipts']+
            [dict(path=args.config,sha256=legacy.sha(ROOT/args.config))])
    updated['outputs']={p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',updated)
    print(dict(metric_rows=len(rows),all_CSV_fields=len(fields),numeric_values_changed=0,
        complete_text_configuration_identity_preserved=True))


if __name__=='__main__':
    main()
