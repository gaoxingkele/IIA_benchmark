"""Fill four genuinely pending full-generation controls; preserve the v65 scope."""
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as old
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows

ROOT,read,sha=rows.ROOT,rows.read,rows.sha
MODULE='scripts.flow_matching.publish_spectral_energy45_native_increment_v66'
TEST='tests/test_fm_spectral_energy45_native_increment_v66.py'
CONFIG='configs/reproducibility/fm_spectral_energy45_native_increment.2026-10-16_v66.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v65'
PACKET='projects/flow_matching_research/results/2026-10-16/Spectral_Energy45_full9_reexecution_v1'
IDENTITY='spectral_original__diffusion_ts__energy__released_author_data__seed45'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,old.MODULE.replace('.','/')+'.py',rows.MODULE.replace('.','/')+'.py',
        BASE+'/all_algorithm_dataset_metrics.csv',BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        BASE+'/five_shared_original_SMD_comparison_references.csv',PACKET+'/packet_integrity.json',
        'configs/reproducibility/fm_spectral_full70_generation_evaluator_reexecution.v1.json']
    return dict(base_catalog=BASE,completed_packet=PACKET,completed_case=IDENTITY,metric_rows=11925,
        previous_numeric_rows=3186,completed_new_controls=4,numeric_rows=3190,
        output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v66',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],all_original_experiments_complete=False)


def merge(before,updates):
    if len(before)!=11925 or sum(bool(r['mean']) for r in before)!=3186:
        raise ValueError('Preserve the complete v65 catalog')
    if len(updates)!=4 or {r['metric'] for r in updates}!={'context_fid','correlational','discriminative','predictive'}:
        raise ValueError('All four original evaluator metrics required')
    for r in updates:
        if (json.loads(r['score_recipe']).get('canonical_id')!=IDENTITY or r['task']!='generation'
                or r['algorithm']!='Spectral_original/diffusion_ts' or r['evaluator_repeats']!='5'):
            raise ValueError('Only this complete generator and five repeats per evaluator are eligible')
    result=old.merge_pending(before,updates)
    if sum(bool(r['mean']) for r in result)!=3190:raise ValueError('Only four previously pending controls may change')
    return result


def actual_updates(headers,captured):
    from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as generation
    packet=ROOT/PACKET;old.packet(packet)
    c=read(ROOT/generation.CONFIG);parent,cases=generation.verify(c)
    binding=read(packet/'all9_complete_generator_reexecution_catalog.json')['records'][IDENTITY]
    report=read(binding['report_path']);guard=read(binding['guard_path'])
    _,proof_path=generation.ready_proof(parent,cases[IDENTITY],IDENTITY)
    if (sha(binding['report_path'])!=binding['report_sha256'] or sha(binding['guard_path'])!=binding['guard_sha256']
            or not report['passed'] or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['report_sha256']!=binding['report_sha256'] or report['original_artifact_proof_sha256']!=sha(proof_path)
            or report['full_original_sampling_calls']!=2 or report['original_evaluation_repeats']!=5
            or not report['every_model_EMA_optimizer_state_unchanged'] or report['actual_new_generator_training_updates']!=0
            or not report['complete_RNG_restoration_and_equal_consumption'] or not report['all_generated_payload_bytes_replayed']
            or not report['all_four_original_evaluators_reexecuted']):
        raise ValueError('All full-generation/RNG/checkpoint/evaluator guards required')
    for entry in report['artifacts']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Complete generated/reference/RNG artifact changed')
    return rows.spectral_rows(headers,{IDENTITY:cases[IDENTITY]},{IDENTITY:report},captured,dict(spectral_artifact_root=c['artifact_root']))


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Frozen complete-catalog provenance changed')
    with (ROOT/BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    captured=datetime.now(timezone.utc).isoformat();after=merge(before,actual_updates(headers,captured))
    out=ROOT/c['output_directory'];out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
    with (out/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(after)
    (out/'results.html').write_text(rows.interactive(after,headers).replace('实验指标 v55','实验指标 v66').replace('算法—数据集—指标 v55','算法—数据集—指标 v66'),encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
        shutil.copyfile(ROOT/BASE/name,out/name)
    state=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=11925,numeric_rows=3190,
        previously_pending_controls_completed=4,untouched_rows_all_fields_and_capture_times_preserved=11921,
        CrossAD_complete_conditions=21,CrossAD_required_conditions=139,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
        generation_complete_reexecution_conditions=9,generation_required_conditions=70,generation_complete_cohorts=1,generation_required_cohorts=14,
        SPOT_complete_cases=3,SPOT_required_unique_cases=25,SPOT_comparison_slots=30,
        new_training_or_inference_by_publication=False,all_original_experiments_complete=False)
    (out/'publication_validation.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# 算法—数据集—指标 v66

全部11925行保留，3190行有数值。仅填入Diffusion-TS/Energy/seed45完整生成重跑的4项原待完成指标，其他11921行全部字段和捕获时间不变。
完整两次采样、全部20次原始评估器、原始模型/EMA/优化器和完整随机状态恢复守卫均已验收。
每项n=1表示一个固定训练模型的完整条件；五次评估器SD/SE/t95仅放evaluator_*列，训练种子不确定性列留空。
Energy42..45为4/5模型，不计算部分组均值、STD或CI；Stocks完整五模型组保持，总计9/70条件、1/14完整组。
历史采样RNG未保存，新采样逐位重现不证明历史生成字节一致；新减原差异同时涉及采样及评估器随机性。

CrossAD21/139、完整主要组1/22，原SPOT25独立案例/30位置/130控制、TAB804macro和878案例/2057标签视图保持。
源训练产物34/70是另一层证据，不与生成重跑数相加；16项已执行但校准未定义控制保持空白，不补零。
MOMENT22、GRASP51及工业TEP95等独立证据包保留各自轨道，本表保留原捕获时间，不是所有进程实时状态。
不认证历史作者或完整TAB等价；45论文/681方法记录/273数据任务/702标量及原论文/TAB/新数据范围保持，整体未完成。
''',encoding='utf-8',newline='\n')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),out/'publication_source.py')
    return state


if __name__=='__main__':print(json.dumps(publish()))
