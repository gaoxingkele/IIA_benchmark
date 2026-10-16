"""Close one actual full Lin model, preserving the complete fifty-model ledger."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as g
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

runtime=read(ROOT/'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
originals,executions,_,_=g.verify(runtime)
identity='grasp_protocol_v3__lin__swan__swan_sf__seed1107'
ref=g.completed_pipeline(runtime,identity)
pp=ROOT/ref['completed_pipeline_configuration'];pipeline=read(pp);producer=binding(pipeline)
c,native,receipt=g.current_native(pipeline['producer_configuration'])
assert len(native['records'])==1
record=native['records'][0];assert record['canonical_id']==identity
result_path=ROOT/record['result_path'];result=read(result_path)
g.check_complete_result(originals[identity],executions[identity],result)
assert record['epochs_completed']==1500 and record['optimizer_updates']==93000
assert record['parameter_count']==result['parameter_count']==69244
assert sha(result_path)==record['result_sha256'] and sha(record['checkpoint_path'])==record['checkpoint_sha256']
for item in result['artifacts']:assert sha(ROOT/item['path'])==item['sha256']
assert len(record['recipes'])==1 and record['recipes'][0]['recipe']==dict(source_samples=5,flow_evaluations=10)
recipe=record['recipes'][0]
assert recipe['full_recipe_reinference_passed'] and recipe['complete_validation_points']==3935 and recipe['complete_test_points']==72000
for role in ['original','replay']:assert sha(recipe[role+'_score_path'])==recipe[role+'_score_sha256']
review_path=ROOT/pipeline['review_output_directory']/'independent_full_replay_verification.json'
review=read(review_path)
assert review['passed'] and review['actual_completed_recipe_count']==1 and review['terminal_resource_receipt']==receipt
assert review['all_original_forward_counts_and_recipe_budgets_verified'] and review['all_metrics_independently_recomputed_from_full_replay']
assert receipt['exit_code']==0 and not receipt['resource_guard_aborted']
prior_path=ROOT/'projects/flow_matching_research/results/2026-10-16/GRASP_ER_SwanSF_seed1107_full52_v1/actual_full52_model_increment_closure.json'
prior=read(prior_path);assert len(set(prior['current_complete_original_ids']))==52
for label,digest in prior['current_source_proof_hashes'].items():assert sha(prior_path.parent/(label+'.json'))==digest
state_path=ROOT/runtime['state_root']/'status.json';state=read(state_path)
assert set(state['verified_original_ids'])==set(prior['current_complete_original_ids'])|{identity}
assert not state['failed_preserved']
p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
seeds=sorted(originals[x]['seed'] for x in state['verified_original_ids'] if x.startswith('grasp_protocol_v3__lin__swan__swan_sf__seed'))
assert seeds==list(range(1103,1108))
main_path=ROOT/'projects/flow_matching_research/results/2026-10-16/GRASP_main_SwanSF_seed1107_full49_model_increment_v1/native_report.json'
main=read(main_path)['records'][0]
assert main['canonical_id']=='grasp_protocol_v3__main__swan__swan_sf__seed1107'
main_ref=g.completed_pipeline(runtime,main['canonical_id'])
main_pipeline=read(ROOT/main_ref['completed_pipeline_configuration'])
_,main_native,_=g.current_native(main_pipeline['producer_configuration'])
assert main_native==read(main_path)
assert sha(main['checkpoint_path'])==main['checkpoint_sha256'] and sha(ROOT/main['result_path'])==main['result_sha256']
assert main['prepared_data_sha256']==record['prepared_data_sha256']
main_recipe=next(x for x in main['recipes'] if x['recipe']==recipe['recipe'])
assert main_recipe['complete_validation_points']==3935 and main_recipe['complete_test_points']==72000
assert main_recipe['forward_evaluation_counts']==recipe['forward_evaluation_counts']
for role in ['original','replay']:assert sha(main_recipe[role+'_score_path'])==main_recipe[role+'_score_sha256']
comparison=[]
for method,item,parameters in [('main',main_recipe,main['parameter_count']),('Lin',recipe,record['parameter_count'])]:
    v=item['replayed_metrics']['validation_only_control'];r=item['replayed_metrics']['paper_retrospective']
    comparison.append(dict(method=method,dataset='SWAN_SF',training_seed=1107,parameters=parameters,
        source_samples=5,flow_evaluations=10,validation_points=3935,test_points=72000,
        validation_only_noPA_F1=v['f1'],precision=v['precision'],recall=v['recall'],threshold=v['threshold'],
        ROC=r['ROC'],AP=r['AP'],test_label_optimized_BestF1_diagnostic_only=r['Best_F1'],complete_ten_training_seed_cohort=False))
xml=ROOT/'tmp/GRASP_lin1107_full53_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)>50
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/GRASP_Lin_SwanSF_seed1107_full53_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
paths=dict(original_result=result_path,native_configuration=ROOT/pipeline['producer_configuration'],
    native_report=ROOT/producer['output_directory']/'checkpoint_replay_audit.json',native_guarded_receipt=ROOT/producer['state_root']/'receipt.json',
    completed_pipeline_configuration=pp,completed_pipeline_proof=ROOT/ref['completed_full_replay_proof'],
    independent_configuration=ROOT/pipeline['review_configuration'],independent_full_recipe_proof=review_path,
    independent_guarded_receipt=ROOT/pipeline['state_root']/'receipt.json',
    original_training_resource_usage=ROOT/result['job']['output_directory']/'resource_usage.json',
    frozen_current_full9120_completion_state=state_path)
for label,path in paths.items():shutil.copyfile(path,packet/(label+'.json'))
write(packet/'actual_full53_model_increment_closure.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=identity,
    complete_original_training_and_full_native_replay_and_independent_recipe_review=True,
    epochs=1500,optimizer_updates=93000,parameter_count=69244,recipes=record['recipes'],
    actual_current_unique_complete_models=53,current_complete_original_ids=state['verified_original_ids'],
    prior52_model_closure=prior_path.relative_to(ROOT).as_posix(),prior52_closure_sha256=sha(prior_path),
    current_source_proof_hashes={label:sha(path) for label,path in paths.items()},
    required_original_entity_models=9120,required_original_training_cohorts=480,required_original_training_seeds=10,
    complete_Lin_SwanSF_seeds=seeds,pending_Lin_SwanSF_seeds=list(range(1108,1113)),
    full_ten_seed_Lin_SwanSF_cohort_complete=False,score_tolerance=c['score_tolerance'],bitwise_score_identity_claim=False,
    inherited_native_template_prose_discrepancy='Frozen inherited prose mentions tau1/seed1104; structured profile/canonical/seed/weights and complete5x10 recipe prove Lin1107. Original frozen files preserved.',
    additional_model_inference_during_closure=0,additional_optimizer_updates_during_closure=0,
    original_scope=runtime['original_scope'],strict_TAB_result=False,author_equivalence_certified=False,all_original_experiments_complete=False))
with (packet/'same_prepared_data_same_seed_default_recipe_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
write(packet/'same_input_single_seed_comparison_binding.json',dict(prepared_data_sha256=record['prepared_data_sha256'],
    main_complete_native_report_path=main_path.relative_to(ROOT).as_posix(),main_complete_native_report_sha256=sha(main_path),
    main_full_forward_counts=main_recipe['forward_evaluation_counts'],Lin_full_forward_counts=recipe['forward_evaluation_counts'],records=comparison,
    single_seed_difference_not_full_ten_seed_superiority=True))
shutil.copyfile(xml,packet/'relevant_fresh_validation.xml')
v=comparison[1];a=comparison[0]
(packet/'README.md').write_text(f'''# GRASP Lin／SWAN SF／seed1107：第53个完整核验模型

完成原1500轮、93000次优化更新、69244参数，以及3935个完整验证点和72000个完整测试点。检查点恢复、全部5个源样本×10次流评估的评分重放、完整前向计数及独立指标均通过，相关守卫退出0且无资源中止。Lin基线采用原一项完整5×10配方，不缩减主模型的十项完整配方网格。

验证集99%分位数阈值、无PA：F1={v['validation_only_noPA_F1']:.12f}，Precision={v['precision']:.12f}，Recall={v['recall']:.12f}，ROC={v['ROC']:.12f}，AP={v['AP']:.12f}。测试标签优化BestF1={v['test_label_optimized_BestF1_diagnostic_only']:.12f}仅作事后诊断，不作为验证校准成绩。

相同prepared_data SHA256、训练种子1107、完整5×10配方及完全相同前向预算的主模型F1为{a['validation_only_noPA_F1']:.12f}，Lin-main差值为{v['validation_only_noPA_F1']-a['validation_only_noPA_F1']:.12f}。两者参数量分别公开；单种子对照不证明十种子或跨数据集优势。

评分使用原rtol=0.0002、atol=0.001核验，不声称逐字节相同。继承冻结模板的文字仍提到tau1/seed1104，实际结构化profile、canonical、seed、权重及完整配方均证明本条件为Lin/seed1107；保留原冻结文件，差异单列。

累计53/9120个完整训练/重放/独立复核模型，原52个已完成ID和证明哈希保持。Lin本组1103..1107仅5/10，1108..1112仍必需，不计算部分十种子均值、区间或论文行。相关58项验证通过，归档未新增训练或推理。

原480组、十训练种子及45篇/681方法记录/273数据任务/702论文指标全部范围保持。该方法仍是已披露的论文描述实现，不认证原作者环境或完整TAB训练等价，完整研究尚未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/GRASP_Lin1107_full53.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),new_full_original_model=identity,
    full_original_models_closed=53,required_models=9120,required_training_seeds=10,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+=f'\n`GRASP Lin／SWAN SF／seed1107完整核验，累计53 <results/2026-10-16/{packet.name}/README.md>`_\n'
text+=f'原1500轮/93000更新/69244参数、3935验证与72000测试点、完整5×10评分重放和独立核算通过。验证99%无PA F1={v["validation_only_noPA_F1"]:.12f}、ROC={v["ROC"]:.12f}、AP={v["AP"]:.12f}；相同数据/种子/前向预算对照单列，测试BestF1仅作诊断。Lin本组5/10，不计算部分十种子统计；原52证明保持，58项相关验证通过，评分保留原容差而非逐字节断言。原9120条件/480组/十种子及全局范围保持，整体未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.iterdir() if p.is_file()]
write(ROOT/'tmp/GRASP_Lin1107_full53_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(packet=str(packet),complete_models=53,required_models=9120,comparison=comparison))
