"""Fill only nineteen newly verified condition cells in the full v69 catalog."""
import csv
from datetime import datetime,timezone
import json
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as old
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows

ROOT,read,sha=rows.ROOT,rows.read,rows.sha
MODULE='scripts.flow_matching.publish_CrossAD25_Spectral10_native_increment_v70'
TEST='tests/test_fm_CrossAD25_Spectral10_native_increment_v70.py'
CONFIG='configs/reproducibility/fm_CrossAD25_Spectral10_native_increment.2026-10-16_v70.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v69'
CROSS_PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row3_seed1106_full25_v1'
GEN_PACKET='projects/flow_matching_research/results/2026-10-16/Spectral_Energy46_full10_reexecution_v1'
CROSS_ID='crossad_ablation__row3__gecco__seed1106'
GEN_ID='spectral_original__diffusion_ts__energy__released_author_data__seed46'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,old.MODULE.replace('.','/')+'.py',rows.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',CROSS_PACKET+'/packet_integrity.json',GEN_PACKET+'/packet_integrity.json',
        'configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json',
        'configs/reproducibility/fm_spectral_full70_generation_evaluator_reexecution.v1.json']
    return dict(base_catalog=BASE,completed_cases=[CROSS_ID,GEN_ID],metric_rows=11925,previous_numeric_rows=3235,
        completed_new_controls=19,numeric_rows=3254,output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v70',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(r['mean']) for r in before)!=3235:raise ValueError('Preserve entire v69 catalog')
    cross=[r for r in updates if json.loads(r['score_recipe']).get('canonical_id')==CROSS_ID]
    gen=[r for r in updates if json.loads(r['score_recipe']).get('canonical_id')==GEN_ID]
    if len(updates)!=19 or len(cross)!=15 or len(gen)!=4:raise ValueError('Only all15CrossAD and all4generation controls required')
    if any(r['task']!='anomaly_detection' or r['algorithm']!='CrossAD/reconstructed_row3' for r in cross):
        raise ValueError('Only actual reconstructed row3 condition may enter')
    if (any(r['task']!='generation' or r['algorithm']!='Spectral_original/diffusion_ts' or r['evaluator_repeats']!='5' for r in gen)
            or {r['metric'] for r in gen}!={'context_fid','correlational','discriminative','predictive'}):
        raise ValueError('All four original full evaluators and five repeats required')
    result=old.merge_pending(before,updates)
    if sum(bool(r['mean']) for r in result)!=3254:raise ValueError('Only nineteen pending controls may be filled')
    return result


def actual_updates(headers,captured):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as full
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as generation
    old.packet(ROOT/CROSS_PACKET);old.packet(ROOT/GEN_PACKET)
    parent,q=cold.verify(read(ROOT/cold.CONFIG));mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==CROSS_ID)
    report=read(ROOT/CROSS_PACKET/'independent_metric_report.json')
    if metrics.audit(mc,q,upstream,job)!=report or full.condition(parent,q,job)!=read(ROOT/CROSS_PACKET/'full_checkpoint_array_proof.json'):
        raise ValueError('Full original checkpoints,all native arrays and independent metrics required')
    updates=rows.cross_rows(headers,[job],{CROSS_ID:report},captured,dict(cross_independent_artifact_root=mc['artifact_root']))
    c=read(ROOT/generation.CONFIG);parent,cases=generation.verify(c)
    binding=read(ROOT/GEN_PACKET/'all10_complete_generator_reexecution_catalog.json')['records'][GEN_ID]
    report=read(binding['report_path']);guard=read(binding['guard_path'])
    _,proof=generation.ready_proof(parent,cases[GEN_ID],GEN_ID)
    if (sha(binding['report_path'])!=binding['report_sha256'] or sha(binding['guard_path'])!=binding['guard_sha256']
            or not report['passed'] or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['report_sha256']!=binding['report_sha256'] or report['original_artifact_proof_sha256']!=sha(proof)
            or report['full_original_sampling_calls']!=2 or report['original_evaluation_repeats']!=5
            or not report['every_model_EMA_optimizer_state_unchanged'] or report['actual_new_generator_training_updates']!=0
            or not report['complete_RNG_restoration_and_equal_consumption'] or not report['all_generated_payload_bytes_replayed']
            or not report['all_four_original_evaluators_reexecuted']):
        raise ValueError('Actual full generation,RNG,checkpoint and evaluator guards required')
    for entry in report['artifacts']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Complete generated/reference/RNG artifact changed')
    updates+=rows.spectral_rows(headers,{GEN_ID:cases[GEN_ID]},{GEN_ID:report},captured,dict(spectral_artifact_root=c['artifact_root']))
    return updates


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Frozen complete publication provenance required')
    with (ROOT/BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    captured=datetime.now(timezone.utc).isoformat();after=merge(before,actual_updates(headers,captured))
    assert sum(a!=b for a,b in zip(before,after))==19
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(after)
    html=rows.interactive(after,headers).replace('<title>实验指标 v55</title>','<title>实验指标 v70</title>')
    html=html.replace('<h1>算法—数据集—指标 v55</h1>','<h1>算法—数据集—指标 v70</h1>')
    (out/'results.html').write_text(html,encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/name,out/name)
    state=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3254,
        previously_pending_controls_completed=19,untouched_rows_all_fields_and_capture_times_preserved=11906,
        CrossAD_complete_conditions=25,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=10,generation_required_conditions=70,generation_complete_cohorts=2,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v70

11925行全部保留，3254行有数值。仅填入CrossAD row3／GECCO／seed1106的15项指标及Diffusion-TS／Energy／seed46完整生成重执行的4项指标，其他11906行的全部字段和捕获时间不变。

CrossAD25/139，row1..3各4/5，row4..5各3/5，完整主要组仍1/22；不计算部分组统计。新增条件完成原20轮/8640批次，全检查点和所有原生点重放、独立三阈值指标通过；消融仍是重建状态，不认证作者精确等价。

生成重执行10/70，完整五训练模型组2/14：Stocks和Energy。Energy42..46已全5/5，完整五模型均值、SD、SE、不截断df4t区间及与原生成的成对差异见独立Energy结果包。此表固定条件行n=1表示一个完整生成器条件，内部五次评估器重复的波动只填evaluator字段；不把它们当作五个训练模型。历史采样RNG未保存，新采样两次逐字节一致不认证历史数组或论文性能等价。

原TAB878案例/2057标签视图及804macro、共享SMD参考、SPOT25独立案例/30比较槽位/130控制及此前全部已完成值保持。未定义值不补零。工业466项完整指标及550检查点队列、DSB首个400040更新源训练（完整原生复核仍待完成）、GRASP52、MOMENT22等证据仍分别列出，没有借此提前填入本表。其他轨道沿用各行原捕获时间，本表不是所有进程实时状态。

原45篇/681方法记录/273原数据任务/702论文指标及原论文/TAB/新数据全部范围不变，完整任务尚未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return state


if __name__=='__main__':print(json.dumps(publish()))
