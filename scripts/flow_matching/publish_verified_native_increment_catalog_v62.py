"""Fill only nineteen previously pending, fully guarded native controls."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
from scripts.flow_matching import publish_native_condition_catalog_v55 as p
ROOT,read,sha=p.ROOT,p.read,p.sha
MODULE='scripts.flow_matching.publish_verified_native_increment_catalog_v62'
TEST='tests/test_fm_verified_native_increment_catalog_v62.py'
CONFIG='configs/reproducibility/fm_verified_native_increment_catalog.2026-10-16_v62.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v61'
CROSS_PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1104_full17_v1'
GEN_PACKET='projects/flow_matching_research/results/2026-10-16/Spectral_Energy44_full8_reexecution_v1'
CROSS_ID='crossad_ablation__row5__gecco__seed1104'
GEN_ID='spectral_original__diffusion_ts__energy__released_author_data__seed44'


def registration():
    files=[MODULE.replace('.','/')+'.py',TEST,p.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',
        CROSS_PACKET+'/packet_integrity.json',GEN_PACKET+'/packet_integrity.json',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json',
        'configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json',
        'configs/reproducibility/fm_spectral_full70_generation_evaluator_reexecution.v1.json']
    return dict(base_catalog=BASE,CrossAD_packet=CROSS_PACKET,generation_packet=GEN_PACKET,
        output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v62',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        metric_rows=11925,expected_numeric_rows=3126,updated_cross_case=CROSS_ID,updated_generation_case=GEN_ID,
        expected_updated_rows=19,source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in files],
        all_original_experiments_complete=False)


def merge_pending(before,updates):
    mapping={p.key(r):r for r in updates}
    if len(mapping)!=len(updates) or not mapping:raise ValueError('Distinct actual complete updates required')
    seen=set();after=[]
    for row in before:
        key=p.key(row)
        if key not in mapping:after.append(dict(row));continue
        if key in seen or row['n']!='0' or row['mean']:raise ValueError('Only unique genuinely pending rows may change')
        new=mapping[key]
        if (set(new)!=set(row) or new['n']!='1' or new['required_repeats']!='1'
                or not new['mean'] or not math.isfinite(float(new['mean']))
                or new['author_equivalence_certified']!='False' or new['strict_TAB_result']!='False'
                or any(new[k] for k in ['std','se','ci95_low','ci95_high'])):
            raise ValueError('One complete fixed case,finite value and honest uncertainty/boundaries required')
        if new['task']=='generation' and new['evaluator_repeats']!='5':raise ValueError('All five evaluator repetitions required')
        after.append(dict(new));seen.add(key)
    if seen!=set(mapping):raise ValueError('Every requested update must map to an existing pending row')
    return after


def packet(path):
    for entry in read(path/'packet_integrity.json')['files']:
        if sha(path/entry['path'])!=entry['sha256']:raise ValueError('Immutable source evidence packet changed')


def updates(headers,c,captured):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as generation
    cp,gp=ROOT/c['CrossAD_packet'],ROOT/c['generation_packet'];packet(cp);packet(gp)
    original,q=cold.verify(read(ROOT/cold.CONFIG));mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==CROSS_ID)
    report=read(cp/'independent_metric_report.json')
    if (metrics.audit(mc,q,upstream,job)!=report
            or groups.condition(original,q,job)!=read(cp/'full_checkpoint_array_proof.json')):
        raise ValueError('Actual complete new CrossAD checkpoint/array/metric guards required')
    rows=p.cross_rows(headers,[job],{CROSS_ID:report},captured,dict(cross_independent_artifact_root=mc['artifact_root']))
    gc=read(ROOT/generation.CONFIG);parent,cases=generation.verify(gc)
    binding=read(gp/'all8_complete_generator_reexecution_catalog.json')['records'][GEN_ID]
    full=read(binding['report_path']);guard=read(binding['guard_path'])
    proof,proof_path=generation.ready_proof(parent,cases[GEN_ID],GEN_ID)
    if (sha(binding['report_path'])!=binding['report_sha256'] or sha(binding['guard_path'])!=binding['guard_sha256']
            or not full['passed'] or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['report_sha256']!=binding['report_sha256'] or full['original_artifact_proof_sha256']!=sha(proof_path)
            or full['full_original_sampling_calls']!=2 or full['original_evaluation_repeats']!=5
            or not full['every_model_EMA_optimizer_state_unchanged'] or full['actual_new_generator_training_updates']!=0
            or not full['complete_RNG_restoration_and_equal_consumption']
            or not full['all_generated_payload_bytes_replayed'] or not full['all_four_original_evaluators_reexecuted']):
        raise ValueError('Complete exact two-sampler/all20-evaluator guard required')
    for item in full['artifacts']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Full generated/reference/RNG evidence changed')
    rows+=p.spectral_rows(headers,{GEN_ID:cases[GEN_ID]},{GEN_ID:full},captured,dict(spectral_artifact_root=gc['artifact_root']))
    if len(rows)!=19:raise ValueError('All fifteen CrossAD plus four generation metric controls required')
    return rows


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Full original-scope frozen catalog configuration required')
    with (ROOT/c['base_catalog']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    if len(before)!=11925 or sum(bool(r['mean']) for r in before)!=3107:raise ValueError('Exact complete v61 base required')
    captured=datetime.now(timezone.utc).isoformat();new=updates(headers,c,captured);after=merge_pending(before,new)
    if sum(bool(r['mean']) for r in after)!=3126:raise ValueError('Preserve every existing value and exactly nineteen new values')
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(after)
    (out/'results.html').write_text(p.interactive(after,headers).replace('实验指标 v55','实验指标 v62').replace('算法—数据集—指标 v55','算法—数据集—指标 v62'),encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/c['base_catalog']/name,out/name)
    status=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3126,
        untouched_rows_all_fields_and_capture_times_preserved=11906,previously_pending_controls_completed=19,
        CrossAD_complete_conditions=17,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=8,generation_required_conditions=70,generation_complete_cohorts=1,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v62

共11925行，3126行有数值。保留全部既有范围与列；仅填入CrossAD row5/seed1104的15项验证阈值控制，以及Diffusion-TS/Energy/seed44完整生成重跑的4项指标，其他11906行全部字段和捕获时间保持。

CrossAD17/139，完整主要五种子组仍1/22；五种重建消融各2/5种子。Energy完整新采样与评估重跑现为3/5模型，生成重跑总体8/70、完整五模型组1/14；部分组不计算完整五种子统计。新生成指标的五次评估器SD/SE/t95只在evaluator_*列，训练种子不确定性列留空。两次完整新采样一致不代表历史采样字节一致。

原SPOT25独立案例/30比较位置/130控制保持，3案例完成对应24控制；8个完整macro有定义，16个已执行但校准未定义，106项待执行。原TAB804macro、878案例/2057标签视图范围、共享SMD引用表及所有既有协议均保留，不补零或改变分母。

本表数值来自实际完整守卫、全数组和独立指标证据，不认证历史作者实现或完整TAB训练策略。TimesNet1106等近期独立报告仍单独提供，尚未逐项并入本表。全45篇/681方法记录/273数据任务/702目标及原论文/TAB/新数据全部任务保持，整体未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return status


if __name__=='__main__':print(json.dumps(publish()))
