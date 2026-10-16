"""Fill only fifteen actual complete CrossAD controls in the full v72 catalog."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as old
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows

ROOT,read,sha=rows.ROOT,rows.read,rows.sha
MODULE='scripts.flow_matching.publish_crossad_row1_native_increment_v73'
TEST='tests/test_fm_crossad_row1_native_increment_v73.py'
CONFIG='configs/reproducibility/fm_crossad_row1_native_increment.2026-10-16_v73.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v72'
PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1107_full28_v1'
IDENTITY='crossad_ablation__row1__gecco__seed1107'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,old.MODULE.replace('.','/')+'.py',rows.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',PACKET+'/packet_integrity.json',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json','configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json']
    return dict(base_catalog=BASE,completed_packet=PACKET,completed_case=IDENTITY,metric_rows=11925,previous_numeric_rows=3284,
        completed_new_controls=15,numeric_rows=3299,output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v73',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in paths],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(x['mean']) for x in before)!=3284:raise ValueError('Entire v72 catalog must be preserved')
    if len(updates)!=15:raise ValueError('All three fixed thresholds and five metrics required')
    for x in updates:
        if (json.loads(x['score_recipe']).get('canonical_id')!=IDENTITY or x['task']!='anomaly_detection'
                or x['algorithm']!='CrossAD/reconstructed_row1'):
            raise ValueError('Only the actual completed reconstructed row1 condition is eligible')
    result=old.merge_pending(before,updates)
    if sum(bool(x['mean']) for x in result)!=3299:raise ValueError('Only fifteen previously pending values may be filled')
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
    if sum(a!=b for a,b in zip(before,after))!=15:raise ValueError('Exactly fifteen actual completed controls required')
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(after)
    html=rows.interactive(after,headers).replace('<title>实验指标 v55</title>','<title>实验指标 v73</title>')
    html=html.replace('<h1>算法—数据集—指标 v55</h1>','<h1>算法—数据集—指标 v73</h1>')
    (out/'results.html').write_text(html,encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/name,out/name)
    state=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3299,
        previously_pending_controls_completed=15,untouched_rows_all_fields_and_capture_times_preserved=11910,
        CrossAD_complete_conditions=28,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=2,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=10,generation_required_conditions=70,generation_complete_cohorts=2,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v73

保留11925行，3299行有数值。本次仅填入CrossAD row1／GECCO／seed1107的15项原待完成指标：三个固定验证阈值的precision、recall、F1、AUROC和AP。其余11910行全部字段及捕获时间保持。固定条件n=1不代表完整训练种子组。

CrossAD累计28/139、主要完整组2/22：主模型及重建row1均完成原五个种子1103至1107，可进行同种子成对差值分析。全部逐种子向量、组均值、种子SD与t95区间另见CrossAD_row1_seed1107_full28_v1证据包。其余四项消融各4/5，不计算部分组统计。固定划分上的训练种子区间不代表未知实体泛化或多重比较显著性。作者精确消融源码未发布，保持paper_reconstructed_author_harness，不认证作者精确等价。

生成重执行保留10/70、2/14完整组。原TAB878案例/2057标签视图/804macro、共享SMD引用、SPOT25独立案例/30槽位/130控制及既有完成值保持。评估器内部重复与训练种子统计分开，未定义数值不补零。

单细胞、工业、GRASP、MOMENT、DSB等证据包保持独立轨道。本表各轨道保留原捕获时间，并非所有进程实时状态。全部45论文/681方法记录/273原数据任务/702论文指标及原论文/TAB/新数据范围保持，整体未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return state


if __name__=='__main__':print(json.dumps(publish()))
