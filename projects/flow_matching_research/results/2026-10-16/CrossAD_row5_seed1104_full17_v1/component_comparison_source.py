"""Archive all ten completed reconstructions as paired single-seed controls."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1104_full17_v1'
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==17
cc=read(ROOT/cold.CONFIG);_,q=cold.verify(cc)
with (packet/'all17_conditions_51_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:values=list(csv.DictReader(f))
lookup={(r['canonical_id'],r['validation_alarm_percent']):r for r in values}
rows=[];bindings={}
for seed in [1103,1104]:
    main=f'crossad_author__fresh_train__gecco__seed{seed}'
    for row in range(1,6):
        identity=f'crossad_ablation__row{row}__gecco__seed{seed}'
        budgets={}
        for cid in [identity,main]:
            job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==cid)
            path=ROOT/job['output_directory']/'result.json';result=read(path)
            budget=training_budget(job,result['details'],read(path.parent/'training_trace.json'))
            assert budget['original_maximum_epochs']==20 and budget['original_patience']==3
            budgets[cid]=budget
            model_path=job.get('model_config','configs/models/fm_crossad_author_pipeline.json')
            frozen={p['path']:p['sha256'] for p in job['frozen_files']}
            assert frozen[model_path]==sha(ROOT/model_path)
            bindings[cid]=dict(result_path=str(path),result_sha256=sha(path),trace_sha256=sha(path.parent/'training_trace.json'),
                actual_original_budget=budget,model_configuration=model_path,model_configuration_sha256=sha(ROOT/model_path),
                full_original_model_parameters=job['model_parameters'],full_original_train_parameters=job['train_parameters'])
        model=read(ROOT/next(j['model_config'] for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity))
        assert model['implementation']=='paper_reconstructed_author_harness' and model['parameters']['row']==row
        for ratio in ['0.1','1','5']:
            a,b=lookup[(identity,ratio)],lookup[(main,ratio)]
            af,bf=float(a['f1']),float(b['f1'])
            tp,fp,fn=(int(a[k]) for k in ['true_positives','false_positives','false_negatives'])
            mtp,mfp,mfn=(int(b[k]) for k in ['true_positives','false_positives','false_negatives'])
            assert tp+fn==mtp+mfn==730 and tp+fp+fn<=69261 and mtp+mfp+mfn<=69261
            assert abs(af-2*tp/(2*tp+fp+fn))<1e-14 and abs(bf-2*mtp/(2*mtp+mfp+mfn))<1e-14
            rows.append(dict(training_seed=seed,ablation_row=row,components=json.dumps(model['parameters']['components']),
                validation_alarm_percent=ratio,complete_test_points=69261,test_positive_points=730,test_negative_points=68531,
                ablation_actual_epochs=budgets[identity]['actual_completed_epochs'],ablation_actual_updates=budgets[identity]['full_training_forward_calls'],
                full_model_actual_epochs=budgets[main]['actual_completed_epochs'],full_model_actual_updates=budgets[main]['full_training_forward_calls'],
                ablation_F1=af,full_model_F1=bf,ablation_minus_full_F1=af-bf,
                ablation_AUROC=float(a['auroc']),full_model_AUROC=float(b['auroc']),
                ablation_AP=float(a['average_precision']),full_model_AP=float(b['average_precision']),
                ablation_threshold=float(a['threshold']),full_model_threshold=float(b['threshold']),
                ablation_TP=tp,ablation_FP=fp,ablation_FN=fn,ablation_TN=68531-fp,
                full_model_TP=mtp,full_model_FP=mfp,full_model_FN=mfn,full_model_TN=68531-mfp,
                ablation_actual_test_normal_FPR=fp/68531,full_model_actual_test_normal_FPR=mfp/68531,
                complete_five_training_seed_comparison=False,historical_author_ablation_equivalence_certified=False))
assert len(rows)==30 and len(bindings)==12
with (packet/'all5_reconstructed_rows_two_seeds_three_validation_budgets.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(packet/'all12_paired_source_training_budget_bindings.json',dict(records=bindings,
    original_stopping_rule_same_but_realized_training_updates_may_differ=True,
    partial_training_seed_means_or_intervals_computed=False,paper_component_causality_certified=False))
table=['| 重建消融 | 原组件布尔值 | 实际轮数 | 验证1%无PA F1 | 同种子完整模型F1 | 差值 |',
    '|---|---|---:|---:|---:|---:|']
for r in rows:
    if r['training_seed']==1104 and r['validation_alarm_percent']=='1':
        table.append(f"| row{r['ablation_row']} | {r['components']} | {r['ablation_actual_epochs']} | {r['ablation_F1']:.6f} | {r['full_model_F1']:.6f} | {r['ablation_minus_full_F1']:.6f} |")
with (packet/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n现在另外汇总全部5种重建消融×2个已完成训练种子×3个固定验证阈值，共30组同种子点值比较。每行列明两侧实际早停轮数和训练更新、测试TP/FP/FN/TN、测试正常点实际误报率、F1、AUROC和AP。全部12个参与模型的实际预算再次核对；相同原20轮上限和patience3允许实际更新量不同，该差异明确保留，不能把这些点值差解释成完全匹配计算量的纯组件因果效应。\n\n'+ '\n'.join(table)+'\n\n布尔顺序为multi-scale、cross-scale、subseries、global-context。上述是种子1104的固定数据与阈值点值，完整CSV还含1103；不汇总部分种子均值、排名或置信区间。不同变体的排序能力和固定阈值F1变化分别列出，原论文历史消融等价性仍未证明。没有新训练、推理或重新校准。\n')
shutil.copyfile(__file__,packet/'component_comparison_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
ara='projects/flow_matching_research/ara/execution/CrossAD_row5_seed1104_full17.2026-10-16_v1.json'
paths=[ara,'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/CrossAD17_component_controls_manifest_20261016_v1.json';assert not manifest.exists()
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(full_conditions=17,paired_controls=30,ablation_models=10,participating_full_models=2,files=len(paths))))
