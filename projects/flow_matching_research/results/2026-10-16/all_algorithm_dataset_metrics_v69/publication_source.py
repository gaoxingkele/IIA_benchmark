"""Fill only fifteen actual complete CrossAD controls in the full v68 catalog."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as old
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows

ROOT,read,sha=rows.ROOT,rows.read,rows.sha
MODULE='scripts.flow_matching.publish_crossad_row2_native_increment_v69'
TEST='tests/test_fm_crossad_row2_native_increment_v69.py'
CONFIG='configs/reproducibility/fm_crossad_row2_native_increment.2026-10-16_v69.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v68'
PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row2_seed1106_full24_v1'
IDENTITY='crossad_ablation__row2__gecco__seed1106'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,old.MODULE.replace('.','/')+'.py',rows.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',PACKET+'/packet_integrity.json',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json','configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json']
    return dict(base_catalog=BASE,completed_packet=PACKET,completed_case=IDENTITY,metric_rows=11925,previous_numeric_rows=3220,
        completed_new_controls=15,numeric_rows=3235,output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v69',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in paths],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(x['mean']) for x in before)!=3220:raise ValueError('Entire v68 catalog must be preserved')
    if len(updates)!=15:raise ValueError('All three fixed thresholds and five metrics required')
    for x in updates:
        if (json.loads(x['score_recipe']).get('canonical_id')!=IDENTITY or x['task']!='anomaly_detection'
                or x['algorithm']!='CrossAD/reconstructed_row2'):
            raise ValueError('Only the actual completed reconstructed row2 condition is eligible')
    result=old.merge_pending(before,updates)
    if sum(bool(x['mean']) for x in result)!=3235:raise ValueError('Only fifteen previously pending values may be filled')
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
    assert sum(a!=b for a,b in zip(before,after))==15
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(after)
    html=rows.interactive(after,headers).replace('<title>实验指标 v55</title>','<title>实验指标 v69</title>')
    html=html.replace('<h1>算法—数据集—指标 v55</h1>','<h1>算法—数据集—指标 v69</h1>')
    (out/'results.html').write_text(html,encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/name,out/name)
    state=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3235,
        previously_pending_controls_completed=15,untouched_rows_all_fields_and_capture_times_preserved=11910,
        CrossAD_complete_conditions=24,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=9,generation_required_conditions=70,generation_complete_cohorts=1,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v69

11925行全部保留，3235行有数值。本次只填入CrossAD row2／GECCO／seed1106的15项原待执行指标：三个固定验证阈值下的precision、recall、F1、AUROC、AP。其他11910行的全部字段及捕获时间不变。固定模型行n=1表示一个完整条件，不是完整训练种子组。

CrossAD24/139；row1和row2各4/5，其他消融各3/5，完整主要种子组仍1/22。不计算部分组均值、区间或成对差异。新增条件已完成20轮、8640个训练批次，完整检查点重放和独立指标均通过；消融仍属paper_reconstructed_author_harness，不认证作者精确等价。

原TAB878案例/2057标签视图及804个macro、共享SMD参考、SPOT25独立案例/30比较槽位/130控制及此前全部已完成值保持。生成重执行仍采用既有9/70、1/14完整五模型组快照。训练种子、固定模型推理与评估器内部重复分别列出，未定义值不补零。其他轨道保留各行原捕获时间，本表不是所有进程的实时状态。

工业466项完整点级指标、全部PA/校准对照及最新完整原生区间指标、GRASP52模型、MOMENT22案例等证据包仍单独列出。总表没有借此新增或替换其他轨道成绩。原45篇/681方法记录/273原数据任务/702论文指标及原论文/TAB/新数据范围不变；完整任务尚未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return state


if __name__=='__main__':print(json.dumps(publish()))
