"""Identify equal saved outputs without claiming extra independent algorithms."""
from pathlib import Path
import json
import shutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_TEP_all95_source_saved_score_independent_metrics_v1'
bindings=read(packet/'all95_full_array_metric_and_guard_bindings.json')['records']
q=read(ROOT/'configs/experiments/fm_industrial_tsad_cpu_queue.v1.json')
jobs={j['id']:j for j in q['jobs']}
pairs=[]
for left,right,meaning in [
    ('fm_objective_independent','fm_objective_rectified','Independent sigma=0 and first-stage rectified share the same straight path; this is not multi-stage reflow.'),
    ('fm_objective_sb__stable_numerics_v2','fm_objective_sf2m_flow_only__stable_numerics_v2','SB and SF2M with the score loss/head disabled share the same bridge/transport and velocity objective in this adapter.')]:
    for seed in range(1103,1108):
        a=left+f'__industrial_v1__tep_classic__seed{seed}'
        b=right+f'__industrial_v1__tep_classic__seed{seed}'
        pa,pb=read(bindings[a]['independent_report_path']),read(bindings[b]['independent_report_path'])
        assert sha(bindings[a]['independent_report_path'])==bindings[a]['independent_report_sha256']
        assert sha(bindings[b]['independent_report_path'])==bindings[b]['independent_report_sha256']
        assert pa['all_array_bindings']==pb['all_array_bindings'] and len(pa['all_array_bindings'])==92
        assert pa['independent_strict_metrics']==pb['independent_strict_metrics']
        sa=read(ROOT/jobs[a]['output_directory']/'result.json');sb=read(ROOT/jobs[b]['output_directory']/'result.json')
        assert sa['training_losses']==sb['training_losses'] and len(sa['training_losses'])==20
        ca=read(ROOT/jobs[a]['model_config']);cb=read(ROOT/jobs[b]['model_config'])
        changes={k:[ca['parameters'].get(k),cb['parameters'].get(k)] for k in set(ca['parameters'])|set(cb['parameters']) if ca['parameters'].get(k)!=cb['parameters'].get(k)}
        if left=='fm_objective_independent':
            assert changes=={'objective':['independent','rectified']} and ca['parameters']['sigma']==cb['parameters']['sigma']==0.
        else:
            assert changes=={'objective':['sb','sf2m'],'score_weight':[1.0,0.0]}
        pairs.append(dict(left=a,right=b,seed=seed,identical_full_score_and_metadata_arrays=92,
            all_three_independent_strict_controls_identical=True,all20_source_losses_identical=True,
            configuration_parameter_differences=changes,interpretation=meaning,
            arbitrary_dataset_or_historical_author_equivalence_claim=False,checkpoint_weights_independently_compared=False))
assert len(pairs)==10
write(packet/'ten_actual_paired_score_identity_controls.json',dict(records=pairs,
    configured_method_records_retained=19,these_are_configurations_not_nineteen_independent_objectives=True,
    multi_stage_reflow_not_equated_with_first_stage_rectified=True,
    actual_evidence_scope='TEP full scores, all3 strict budgets and20 recorded source losses, five matched seeds per pair',
    full_original_research_scope_unchanged=True))
note='''
相同分数的结构核对
----------------
另核对10组真实配对：5种子下independent(sigma=0)与单阶段rectified，以及SB与关闭score分支的SF2M-flow-only。
每对全部92个评分/元数据数组内容哈希、三种严格阈值指标和20个已记录训练损失均完全一致。
参数差异分别仅objective名称，以及objective与不生效的score_weight；本地适配器中对应直线路径或同一桥路径/速度目标。
这是预期的目标退化，19个配置记录不能说成19个独立数学目标。多阶段reflow仍是不同配置，未合并或删去原记录。
这里没有重新比较检查点权重，也不把TEP五种子一致扩大为任意任务或历史作者实现等价。
'''
with (packet/'README.md').open('a',encoding='utf-8',newline='\n') as f:f.write(note)
shutil.copyfile(__file__,packet/'score_identity_audit_source.py')
integrity=packet/'packet_integrity.json'
write(integrity,dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p!=integrity]))
manifest=ROOT/'tmp/industrial_TEP95_independent_metrics_publication_manifest_20261016_v1.json'
m=read(manifest)
m['paths']=sorted(set(m['paths'])|{(packet/name).relative_to(ROOT).as_posix() for name in ['ten_actual_paired_score_identity_controls.json','score_identity_audit_source.py']})
write(manifest,m)
print(json.dumps(dict(actual_complete_pair_controls=len(pairs),source_losses_per_case=20,exact_score_metadata_arrays_per_case=92)))
