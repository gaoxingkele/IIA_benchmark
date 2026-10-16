"""Fill only fifteen actual complete CrossAD controls in the full v66 catalog."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as old
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows

ROOT,read,sha=rows.ROOT,rows.read,rows.sha
MODULE='scripts.flow_matching.publish_crossad_row5_native_increment_v67'
TEST='tests/test_fm_crossad_row5_native_increment_v67.py'
CONFIG='configs/reproducibility/fm_crossad_row5_native_increment.2026-10-16_v67.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v66'
PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1105_full22_v1'
IDENTITY='crossad_ablation__row5__gecco__seed1105'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,old.MODULE.replace('.','/')+'.py',rows.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',PACKET+'/packet_integrity.json',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json','configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json']
    return dict(base_catalog=BASE,completed_packet=PACKET,completed_case=IDENTITY,metric_rows=11925,previous_numeric_rows=3190,
        completed_new_controls=15,numeric_rows=3205,output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v67',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in paths],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(x['mean']) for x in before)!=3190:raise ValueError('Entire v66 catalog must be preserved')
    if len(updates)!=15:raise ValueError('All three fixed thresholds and five metrics required')
    for x in updates:
        if (json.loads(x['score_recipe']).get('canonical_id')!=IDENTITY or x['task']!='anomaly_detection'
                or x['algorithm']!='CrossAD/reconstructed_row5'):
            raise ValueError('Only the actual completed reconstructed row5 condition is eligible')
    result=old.merge_pending(before,updates)
    if sum(bool(x['mean']) for x in result)!=3205:raise ValueError('Only fifteen previously pending values may be filled')
    return result


def actual_updates(headers,captured):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as full
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    packet=ROOT/PACKET;old.packet(packet)
    parent,q=cold.verify(read(ROOT/cold.CONFIG));mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==IDENTITY)
    report=read(packet/'independent_metric_report.json')
    if (metrics.audit(mc,q,upstream,job)!=report or full.condition(parent,q,job)!=read(packet/'full_checkpoint_array_proof.json')):
        raise ValueError('Actual guarded full checkpoint,all points and independent metrics required')
    return rows.cross_rows(headers,[job],{IDENTITY:report},captured,dict(cross_independent_artifact_root=mc['artifact_root']))


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Frozen complete publication provenance required')
    with (ROOT/BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    captured=datetime.now(timezone.utc).isoformat();after=merge(before,actual_updates(headers,captured))
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(after)
    (out/'results.html').write_text(rows.interactive(after,headers).replace('实验指标 v55','实验指标 v67').replace('算法—数据集—指标 v55','算法—数据集—指标 v67'),encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/name,out/name)
    state=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3205,previously_pending_controls_completed=15,
        untouched_rows_all_fields_and_capture_times_preserved=11910,CrossAD_complete_conditions=22,CrossAD_required_conditions=139,
        CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,generation_complete_reexecution_conditions=9,generation_required_conditions=70,
        generation_complete_cohorts=1,generation_required_cohorts=14,SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v67

全部11925行保留，3205行有数值。仅填入CrossAD row5/seed1105完整条件的15项原待执行控制，其他11910行全部字段和捕获时间不变。
三种冻结验证阈值、无PA的precision/recall/F1/AUROC/AP均从完整数组独立复核；固定模型的n=1表示一个完整条件，不是完整训练种子组。
CrossAD22/139；row1..row5均各3/5主要训练种子；完整主要组仍1/22，不计算部分组均值或区间。

既有70生成重跑、9完整条件/1完整五模型组、SPOT25独立案例/30比较位置/130控制、原TAB804macro和878案例/2057标签视图全部保持。
已有24项SPOT控制完成，其中8项完整macro有定义、16项因实体校准未定义，106项待执行。空白不补零，不改变分母。
固定检查点、五次评估器内部重复和五/十训练种子汇总分开标注，完整但未定义的指标保留原状态。
近期MOMENT22案例、工业337数组预处理、466个原条件独立指标及FM25同分数PA对照、GRASP51模型等证据包保持独立轨道；其他方法保留各行原捕获时间，本表不是全部进程的实时状态。
本表不认证历史作者、完整TAB训练或全部论文成绩已重现；原45篇/681方法记录/273数据任务/702指标及原论文/TAB/新数据全部范围保持，整体未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return state


if __name__=='__main__':print(json.dumps(publish()))
