"""Fill thirty existing pending controls from two complete native conditions."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as previous
from scripts.flow_matching import publish_native_condition_catalog_v55 as p

ROOT,read,sha=p.ROOT,p.read,p.sha
MODULE='scripts.flow_matching.publish_crossad_two_native_increments_v63'
CONFIG='configs/reproducibility/fm_crossad_two_native_increments.2026-10-16_v63.json'
TEST='tests/test_fm_crossad_two_native_increments_v63.py'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v62'
CASES={
    'crossad_ablation__row1__gecco__seed1105':'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1105_full18_v1',
    'crossad_ablation__row2__gecco__seed1105':'projects/flow_matching_research/results/2026-10-16/CrossAD_row2_seed1105_full19_v1',
}


def registration():
    sources=[MODULE.replace('.','/')+'.py',TEST,previous.MODULE.replace('.','/')+'.py',p.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json','configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json']
    sources+=[folder+'/packet_integrity.json' for folder in CASES.values()]
    return dict(base_catalog=BASE,updates=CASES,metric_rows=11925,previous_numeric_rows=3126,expected_updated_rows=30,
        expected_numeric_rows=3156,output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v63',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in sources],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(x['mean']) for x in before)!=3126:raise ValueError('Exact full v62 base required')
    if len(updates)!=30:raise ValueError('Both complete conditions require all thirty controls')
    ids=set()
    for row in updates:
        try:identity=json.loads(row['score_recipe'])['canonical_id']
        except (ValueError,KeyError,TypeError):raise ValueError('Actual case binding required') from None
        if identity not in CASES or row['task']!='anomaly_detection':raise ValueError('Only the two verified CrossAD cases are eligible')
        ids.add(identity)
    if ids!=set(CASES):raise ValueError('Neither actual completed condition may be omitted')
    after=previous.merge_pending(before,updates)
    if sum(bool(x['mean']) for x in after)!=3156:raise ValueError('Only thirty pending values may be filled')
    return after


def actual_updates(headers,captured):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    original,q=cold.verify(read(ROOT/cold.CONFIG));mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
    rows=[]
    for identity,folder in CASES.items():
        packet=ROOT/folder;previous.packet(packet)
        job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
        report=read(packet/'independent_metric_report.json')
        if metrics.audit(mc,q,upstream,job)!=report or groups.condition(original,q,job)!=read(packet/'full_checkpoint_array_proof.json'):
            raise ValueError('Complete guarded checkpoint/coordinate/full-array/independent metrics required')
        rows+=p.cross_rows(headers,[job],{identity:report},captured,dict(cross_independent_artifact_root=mc['artifact_root']))
    return rows


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Frozen complete-scope source receipts required')
    with (ROOT/BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    captured=datetime.now(timezone.utc).isoformat();after=merge(before,actual_updates(headers,captured))
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(after)
    (out/'results.html').write_text(p.interactive(after,headers).replace('实验指标 v55','实验指标 v63').replace('算法—数据集—指标 v55','算法—数据集—指标 v63'),encoding='utf-8')
    for filename in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/filename,out/filename)
    status=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3156,
        untouched_rows_all_fields_and_capture_times_preserved=11895,previously_pending_controls_completed=30,
        CrossAD_complete_conditions=19,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=8,generation_required_conditions=70,generation_complete_cohorts=1,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v63

共11925行，3156行有数值。仅将CrossAD row1/seed1105和row2/seed1105两个完整条件的30项原待完成控制填入；其他11895行全部字段和捕获时间保持。
每个条件覆盖3种冻结验证集阈值和precision/recall/F1/AUROC/AP五项，均无PA、不使用测试标签选择阈值，完整数组与独立指标复核通过。
CrossAD19/139；row1、row2各3/5训练种子，其余消融各2/5；完整主要五种子组仍1/22，不计算部分五种子均值或区间。

原70项生成重跑范围、8个完整条件及1/14完整五模型组、原SPOT25独立案例/30比较位置/130控制、原TAB804macro及878案例/2057标签视图全部保持。
已有24项SPOT控制完成，8项完整macro有定义，16项校准未定义，106项待执行。完整但未定义的值保持空白，不补零或静默剔除。
固定检查点、五次评估器内部重复、五/十训练种子汇总分别标注；这些行不认证历史作者实现或完整TAB训练协议。
TimesNet与GRASP等近期独立完整结果包仍单独提供，尚未逐轨合并；其他算法保留各行原捕获时间，表格不是当前全部进程的实时状态。
原45篇/681方法记录/273数据任务/702目标及原论文/TAB/新数据完整范围保持，整体未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return status


if __name__=='__main__':print(json.dumps(publish()))
