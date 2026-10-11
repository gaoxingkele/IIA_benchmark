"""Independently verify range-only changes and freeze their complete publication."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import shutil

from scripts.flow_matching.publish_full_range_increment_v25 import ROOT, METRICS, PROTOCOL, SCOPE, legacy, stream_sha
from scripts.flow_matching.publish_result_listing_audited_v9 import verified_capture, key


def verify_delta(before, after):
    if len(before) != len(after) or [key(r) for r in before] != [key(r) for r in after]:
        raise ValueError('Original canonical metric scope and order must remain unchanged')
    changed = []
    fields = {'n','mean','std','se','ci95_low','ci95_high','source','captured_utc','author_equivalence_certified','strict_TAB_result'}
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (new['task'] != 'anomaly_detection' or new['protocol'] != PROTOCOL or new['metric'] not in METRICS
                or new['required_repeats'] != old['required_repeats'] or not int(old['n']) < int(new['n']) <= int(old['required_repeats'])
                or new['author_equivalence_certified'].lower() != 'false' or new['strict_TAB_result'].lower() != 'false'
                or any(old.get(field) != new.get(field) for field in set(old)|set(new) if field not in fields)):
            raise ValueError('Only original range-repeat increments with unchanged protocols and budgets may change')
        changed.append(index)
    legacy.validate_rows(after)
    return dict(changed_metric_rows=len(changed),newly_measured_metric_rows=sum(before[i]['mean']=='' for i in changed),
                unchanged_rows=len(before)-len(changed), changed_indices=changed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=legacy.read(ROOT/args.config)
    if config['original_scope']!=SCOPE:raise ValueError('Preserve all original experiment obligations')
    for r in config['source_receipts']:
        if stream_sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Frozen complete publication source changed')
    before=ROOT/config['pre_increment_publication'];base=ROOT/config['increment_publication']
    verified_capture(before,'publication_validation.json');old=verified_capture(base,'publication_validation.json')
    def rows(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    original,current=rows(before/'all_algorithm_dataset_metrics.csv'),rows(base/'all_algorithm_dataset_metrics.csv')
    delta=verify_delta(original,current)
    if delta['changed_metric_rows']!=18 or delta['newly_measured_metric_rows']!=12:
        raise ValueError('Exactly the independently audited eighteen range increments required')
    measured=sum(bool(r['mean']) for r in current)
    if len(current)!=7283 or measured!=1805:raise ValueError('Keep full original registered and measured metric scope')
    proof=legacy.read(ROOT/config['increment_proof'])
    if proof['full_saved_score_evaluations_verified']!=18 or proof['additional_training_seed_slots']!=0:
        raise ValueError('Eighteen original full saved-score evaluations without added training seeds required')
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Preserve every previous publication')
    target.mkdir(parents=True)
    for name in old['outputs']:
        if name not in ('README.rst','README.md'):
            shutil.copyfile(base/name,target/name)
            assert stream_sha(base/name)==stream_sha(target/name)
    captured=datetime.now(timezone.utc).isoformat()
    review=dict(captured_utc=captured,original_scope=SCOPE,comparison_from=config['pre_increment_publication'],
                comparison_to=config['increment_publication'],**delta,aggregate_metric_rows=len(current),measured_metric_rows=measured,
                all_other_rows_and_capture_times_identical=True,all_point_and_author_metrics_unchanged=True,
                v25_inherited_prefix_flag_not_valid_for_v24=True,v26_all_numeric_exports_byte_identical_to_v25=True,
                additional_training_seed_slots=0,all_original_experiments_complete=False)
    review_path=ROOT/config['independent_review'];review_path.parent.mkdir(parents=True,exist_ok=True)
    if review_path.exists():raise FileExistsError('Preserve previous independent publication review')
    legacy.write(review_path,review)
    (target/'README.rst').write_text('完整算法—数据集指标 v26\n==========================\n\n'
        '7,283项指标；1,805项有实测值；5,478项缺结果。\n\n'
        '相对v24，本轮18项完整保存分数评价更新18项VUS／验证阈值Affiliation指标，其中12项由缺失变为实测；其余7,265条指标及采集时间逐字段一致。'
        '点级P/R/F1、AUROC、AP、所有其他任务和作者协议均未更新；原种子预算不变，无新增训练种子。\n\n'
        'v26所有数值导出及逐行时间逐字节保留v25。v25继承了历史前缀一致旗标，该旗标不适用于v25与v24的比较；'
        'v26重新核验两者完整差异，当前base指v25，因此本版本的数值一致旗标适用于v26与v25。保留v25及失败登记作为历史证据。\n\n'
        '18项评价覆盖全部原始实体与完整测试时间点，核验原模型/分数、原始与预处理数据、250个阈值及每一个整数缓冲区、验证校准阈值和宏平均。'
        '测试标签比例诊断未混入本次更新；单种子没有不确定性，多种子报告STD、SE和Student-t95%区间。\n\n'
        '完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md、strict_algorithm_dataset.md。'
        '证据见publication_validation.json、range_increment_proof及independent_range_publication_review。\n\n'
        '全部45篇论文、681项方法/基线/消融、273项原始数据任务及采用基线仍须完成。作者等价及完整TAB训练框架未认证。\n',encoding='utf-8')
    receipt=copy.deepcopy(old)
    receipt.update(published_utc=captured,base_publication_utc=old['published_utc'],replaced_metric_rows=0,changed_numeric_metric_rows=0,
        newly_measured_metric_rows=0,performance_rows_changed=False,all_numeric_tables_byte_identical_to_base=True,
        base_numeric_csv_prefix_byte_identical=True,range_increment_changed_metric_rows=18,range_increment_newly_measured_metric_rows=12,
        range_increment_comparison_base=config['pre_increment_publication'],independent_range_publication_review=config['independent_review'],
        additional_training_seed_slots=0,newly_completed_canonical_training_slots=0,all_original_experiments_complete=False)
    receipt['source_receipts']+=config['source_receipts']+[dict(path=args.config,sha256=stream_sha(ROOT/args.config)),dict(path=config['independent_review'],sha256=stream_sha(review_path))]
    receipt['outputs']={p.name:stream_sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',receipt)
    print(dict(rows=len(current),measured=measured,range_rows_updated_since_v24=18,new_measured_range_rows_since_v24=12,all_original_experiments_complete=False))


if __name__=='__main__':main()
