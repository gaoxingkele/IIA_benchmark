"""Close all five original Sines training seeds, keeping two data tracks separate."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import fitz
from pypdf import PdfReader
import psutil
from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as s
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

identity='spectral_original__diffusion_ts__sines__paper_sines_distribution__seed46'
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_fourth_complete_GPU_and_Spectral33_full_source_artifacts_v1'
catalog_path=prior/'Spectral_unique33_full_source_artifact_model_catalog.json'
catalog=read(catalog_path);assert len(catalog['records'])==33
c=read(ROOT/s.CONFIG);cases,_=s.verify(c)
state=read(ROOT/c['state_root']/'status.json');owner=psutil.Process(state['pid'])
assert owner.create_time()==state['create_time'] and owner.is_running()
assert identity in state['verified'] and not state['failed_preserved']
entry=state['verified'][identity];proof_path=ROOT/entry['proof'];proof=read(proof_path)
assert proof['passed'] and sha(proof_path)==entry['proof_sha256']
assert proof['canonical_id']==identity and proof['seed']==46
assert proof['original_optimizer_updates']==12000 and proof['original_training_minibatches']==24000
assert proof['actual_model_parameters']==235249 and proof['paper_reported_parameters']==232177
assert proof['complete_reference_shape']==[10000,24,5] and proof['complete_generated_shape']==[10005,24,5]
for field in ['actual_full_original_checkpoint_tensors_restored_bitwise','complete_original_reference_array_bitwise_verified',
    'all_original_generated_arrays_finite','all_four_five_repeat_evaluator_statistics_independently_recomputed']:
    assert proof[field]
assert not proof['cuda_context_initialized'] and not proof['generative_model_reinferred_by_this_audit']
assert not proof['learned_evaluators_independently_retrained']
case=cases[identity];job=case['job'];queue=read(ROOT/case['queue'])
result_path,terminal_path,result=s.source_ready(case)
assert sha(result_path)==proof['original_result_sha256'] and sha(terminal_path)==proof['actual_terminal_receipt_sha256']
guard_path=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json';guard=read(guard_path)
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(proof_path)
validation_path=proof_path.parent/'validation.json';validation=read(validation_path)
assert validation['passed'] and validation['outputs']['independent_full_native_artifact_audit.json']==sha(proof_path)
for pin in validation['source_receipts']:assert sha(ROOT/pin['path'])==pin['sha256']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_PaperSines46_full34_complete5_source_cohort_v2'
assert not packet.exists()
new=dict(canonical_id=identity,model=job['model'],dataset=job['dataset'],seed=46,optimizer_updates=12000,training_minibatches=24000,
    complete_reference_shape=proof['complete_reference_shape'],complete_generated_shape=proof['complete_generated_shape'],
    model_parameters=235249,metrics=proof['metrics'],original_result_sha256=sha(result_path),artifacts=proof['artifacts'],
    full_native_proof=(packet/'new_independent_artifact_proof.json').relative_to(ROOT).as_posix(),full_native_proof_sha256=sha(proof_path),
    original_runtime_proof_path=str(proof_path),independently_retrained_learned_evaluators=False,generative_output_reinferred=False)
records=deepcopy(catalog['records'])+[new]
assert len(records)==len({r['canonical_id'] for r in records})==34 and records[:33]==catalog['records']
groups=defaultdict(list);artifact_bindings=[];actual_by_id={}
for row in records:
    actual=proof if row['canonical_id']==identity else read(ROOT/row['full_native_proof'])
    full=proof_path if row['canonical_id']==identity else ROOT/row['full_native_proof']
    assert sha(full)==row['full_native_proof_sha256'] and actual['passed']
    assert actual['canonical_id']==row['canonical_id'] and actual['metrics']==row['metrics']
    current=cases[row['canonical_id']]['job']
    source_path,source_guard,source=s.source_ready(cases[row['canonical_id']])
    assert sha(source_path)==row['original_result_sha256']
    assert sha(source_guard)==actual['actual_terminal_receipt_sha256']
    assert actual['original_optimizer_updates']==current['training_updates']
    assert actual['original_training_minibatches']==current['training_updates']*current['gradient_accumulate_every']
    for artifact in row['artifacts']:
        assert sha(ROOT/artifact['path'])==artifact['sha256'];artifact_bindings.append(artifact)
    actual_by_id[row['canonical_id']]=actual
    groups[(row['model'],row['dataset'],current['data_track'])].append(dict(actual,model=row['model'],dataset=row['dataset'],data_track=current['data_track']))
summaries=[summarize_cohort(v) for _,v in sorted(groups.items())]
paper_group=next(v for v in summaries if v['dataset']=='sines' and v['data_track']=='paper_sines_distribution')
released_group=next(v for v in summaries if v['dataset']=='sines' and v['data_track']=='released_author_data')
assert paper_group['full_five_seed_cohort_complete'] and released_group['full_five_seed_cohort_complete']
assert paper_group['independently_reviewed_training_seeds']==[42,43,44,45,46]
assert not paper_group['missing_original_training_seeds']
prior_groups=read(prior/'Spectral_all33_cohort_completion_and_metrics.json')['cohorts']
normal=lambda x:json.loads(json.dumps(x))
for old in prior_groups:
    if old['dataset']=='sines' and old['data_track']=='paper_sines_distribution':continue
    now=next(x for x in summaries if (x['model'],x['dataset'],x['data_track'])==(old['model'],old['dataset'],old['data_track']))
    assert normal(now)==old

def interval(values):
    assert len(values)==5 and all(math.isfinite(v) for v in values)
    mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(5);half=2.7764451051977987*se
    return dict(n=5,mean=mean,seed_sample_SD=sd,seed_SE=se,t95_low=mean-half,t95_high=mean+half)

paired=[];recipe_bindings=[]
allowed={'id','data_track','model_config','frozen_samples','data_sha256','reference_path','reference_sha256','output_directory','artifact_directory'}
for seed in [42,43,44,45,46]:
    a=f'spectral_original__diffusion_ts__sines__released_author_data__seed{seed}'
    b=f'spectral_original__diffusion_ts__sines__paper_sines_distribution__seed{seed}'
    ja,jb=cases[a]['job'],cases[b]['job'];ma,mb=read(ROOT/ja['model_config']),read(ROOT/jb['model_config'])
    assert {k:v for k,v in ja.items() if k not in allowed}=={k:v for k,v in jb.items() if k not in allowed}
    assert ma['parameters']==mb['parameters'] and ma['author_yaml_sha256']==mb['author_yaml_sha256']
    assert {k:v for k,v in ma.items() if k not in ['id','data_case']}=={k:v for k,v in mb.items() if k not in ['id','data_case']}
    recipe_bindings.append(dict(seed=seed,released_job=ja,paper_distribution_job=jb,all_non_data_job_fields_identical=True,
        full_model_parameters_and_author_YAML_identical=True))
    for metric in proof['metrics']:
        av,bv=actual_by_id[a]['metrics'][metric]['mean'],actual_by_id[b]['metrics'][metric]['mean']
        paired.append(dict(seed=seed,metric=metric,released_mean=av,paper_distribution_mean=bv,paper_minus_released=bv-av))
paired_summaries={metric:interval([r['paper_minus_released'] for r in paired if r['metric']==metric]) for metric in proof['metrics']}
cohort_rows=[]
for group in [released_group,paper_group]:
    for metric,values in group['metrics'].items():
        checked=interval([r['evaluator_mean'] for r in values['per_training_model']])
        assert math.isclose(checked['mean'],values['full_cohort_mean'],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(checked['seed_sample_SD'],values['between_training_model_sample_STD'],rel_tol=1e-12,abs_tol=1e-12)
        assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip([checked['t95_low'],checked['t95_high']],values['between_training_model_CI95']))
        cohort_rows.append(dict(data_track=group['data_track'],metric=metric,**checked,actual_evaluator_runs=25,
            training_models=5,evaluator_repetitions_per_model=5,uncertainty_unit='five_model_specific_evaluator_means'))
protocol=read(ROOT/'configs/reproducibility/spectral_mean_flow_original_protocol.v2.json')
pdf=ROOT/protocol['source_pdf'];assert sha(pdf)==protocol['source_pdf_sha256']
texts=[fitz.open(pdf)[8].get_text(),PdfReader(str(pdf)).pages[8].extract_text()]
# pypdf concatenates neighboring numeric table cells. Read the actual printed
# three decimal places; a greedy [.\d]+ incorrectly consumes the next column.
pattern=r'Diffusion-TS\s*(\d+\.\d{3})\s*±\s*(\.\d{3})'
extractions=[re.findall(pattern,text) for text in texts]
assert len(extractions[0])==len(extractions[1])==4
assert extractions[0]==extractions[1]
names=['context_fid','correlational','discriminative','predictive']
assert [float(x[0]) for x in extractions[0]]==[.013,.016,.030,.095]
comparisons=[]
for metric,(mean,spread) in zip(names,extractions[0]):
    release=next(x for x in cohort_rows if x['data_track']=='released_author_data' and x['metric']==metric)
    paper=next(x for x in cohort_rows if x['data_track']=='paper_sines_distribution' and x['metric']==metric)
    comparisons.append(dict(metric=metric,paper_Table1_printed_mean=float(mean),paper_printed_spread_literal=spread,
        paper_spread_not_reinterpreted_as_training_seed_SD=True,PDF_page=9,
        released_data_five_model_mean=release['mean'],paper_distribution_five_model_mean=paper['mean'],
        released_minus_published=release['mean']-float(mean),paper_distribution_minus_published=paper['mean']-float(mean),
        historical_author_equivalence_certified=False))
packet.mkdir()
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
for name,path in dict(new_original_result=result_path,new_source_terminal_guard=terminal_path,new_independent_artifact_proof=proof_path,
    new_artifact_review_guard=guard_path,new_review_validation=validation_path,
    new_review_configuration=ROOT/c['configuration_root']/(identity+'.json')).items():shutil.copyfile(path,packet/(name+'.json'))
write(packet/'unique34_full_source_and_artifact_model_catalog.json',dict(records=records,full_source_and_artifact_reviewed_models=34,
    required_original_models=70,previous33_records_unchanged=True,previous_catalog_sha256=sha(catalog_path),
    independent_generative_reexecution_certified=False,learned_evaluator_retraining_certified=False,all_original_experiments_complete=False))
write(packet/'all34_cohort_completion_and_metrics.json',dict(cohorts=summaries,required_original_training_seeds=[42,43,44,45,46],
    complete_original_five_seed_cohorts=sum(x['full_five_seed_cohort_complete'] for x in summaries),required_original_cohorts=14,all_original_experiments_complete=False))
write(packet/'all34_actual_artifact_hash_checks.json',dict(all_original_artifacts_checked=artifact_bindings,source_results_checked=34,independent_proofs_checked=34))
write(packet/'paired_data_protocol_recipe_bindings.json',dict(records=recipe_bindings,paired_model_seed_differences=paired_summaries,
    only_registered_data_protocol_changes=True,frequency_and_phase_both_change=True,exclusive_historical_causal_explanation_proven=False))
write(packet/'paper_Table1_Sines_numeric_source_verification.json',dict(source_pdf=str(pdf),source_pdf_sha256=sha(pdf),PDF_page=9,
    table='Table1',model='Diffusion-TS',column='Sines',PyMuPDF_and_pypdf_same_four_cells=True,
    extracted_cells=extractions[0],paper_spread_semantics_not_certified=True))
write(packet/'prepublication_extraction_correction.json',dict(preserved_v1_helper='tmp/close_Spectral_PaperSines46_full34_five_seed_cohort_20261016_v1.py',
    failed_prepublication_check='Two PDF extractors initially disagreed because pypdf joins adjacent numeric columns',
    correction='Match exact three-decimal printed table cells,then require both extractors to agree exactly',
    numerical_worker_failed_or_retried=False,source_results_or_statistics_changed=False))
for filename,rows in [('both_complete_Sines_five_model_cohorts_8_metrics.csv',cohort_rows),
    ('all5_paired_seeds_20_data_protocol_differences.csv',paired),('paper_Table1_Sines_two_protocol_comparisons.csv',comparisons)]:
    with (packet/filename).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
metric_rows=[]
for row in records:
    for metric,v in row['metrics'].items():
        metric_rows.append(dict(canonical_id=row['canonical_id'],model=row['model'],dataset=row['dataset'],
            data_track=cases[row['canonical_id']]['job']['data_track'],training_seed=row['seed'],metric=metric,
            five_evaluator_mean=v['mean'],within_model_evaluator_sample_STD=v['std'],within_model_evaluator_t95_low=v['ci95_low'],within_model_evaluator_t95_high=v['ci95_high']))
with (packet/'all34_models_136_original_evaluator_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],full_source_and_artifact_models=34,required_source_models=70,
    paper_distribution_Sines_complete_training_seeds=[42,43,44,45,46],complete_source_five_training_model_cohorts=sum(x['full_five_seed_cohort_complete'] for x in summaries),
    required_source_cohorts=14,new_completed_source_condition=1,new_training_seed_slots=0,
    independent_generation_evaluator_reexecution_credit=0,all_original_experiments_complete=False))
lines=['# Diffusion-TS论文分布Sines：五训练种子完整组已核验','',
    'seed46完成12000次原始优化更新、24000训练小批次、完整10000×24×5参考与10005×24×5生成数组，以及四类各5次原始评估。源守卫、完整检查点模型/EMA/优化器及全部数组和评估统计审计均通过。',
    '论文分布Sines现有42—46五个独立训练模型，每模型五次评估，均不再是部分组。以下SD/SE/t区间针对五个模型的评估均值，25次内部评估不当成25个训练模型。发布数据分布的五模型组单独列出，不能混合两个数据轨道。','',
    '| 指标 | 发布数据五模型均值 | 论文分布五模型均值 | 原论文Table1均值 |','|---|---:|---:|---:|']
for row in comparisons:lines.append(f"| {row['metric']} | {row['released_data_five_model_mean']:.9f} | {row['paper_distribution_five_model_mean']:.9f} | {row['paper_Table1_printed_mean']:.3f} |")
lines += ['',
    '对每个同种子配对，逐字段证明除数据及其元数据/输出路径外原数值参数一致，完整模型参数与作者YAML一致；单独给出20个配对差及五模型描述性区间。数据协议同时改变频率和相位分布，不把差异独归因于某一个参数，也未证明历史论文的唯一原因。',
    '原论文PDF第9页Table1的四个Diffusion-TS/Sines目标由PyMuPDF和pypdf独立取值一致；论文印刷±保留原字符串，不冒充训练种子SD。模型实际235249参数，论文列232177，框架/历史环境和采样初始RNG仍有边界，所以不认证原论文成绩已重现。',
    '完整源与产物核验34/70；之前33条记录、各原始产物和其他组汇总保持，所有14个原始五种子组仍要求完成。本次审计不重新生成序列或独立重训评估器，不能抵扣另一个70条件的生成与评估重跑轨道。',
    '现有验证器及完整种子统计代码未改变；沿用已通过的固定源验证规则，并实际重算本次全部源产物哈希、五模型统计、配对差和论文目标。',
    '原45篇/681方法记录/273数据任务/702原始标量及原论文、TAB、新数据的全部未完成范围保留，整体目标未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=x.relative_to(packet).as_posix(),sha256=sha(x)) for x in sorted(packet.rglob('*')) if x.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_PaperSines46_full34_complete5_source_cohort.2026-10-16_v2.json'
assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),full_source_and_artifact_models=34,required_models=70,
    paper_distribution_Sines_complete_training_seeds=[42,43,44,45,46],all_original_experiments_complete=False))
head=lambda x:subprocess.check_output(['git','-C',str(x),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/Spectral_PaperSines46_full34_complete5_manifest_20261016_v2.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),
    paths=sorted([ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']+[x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()])))
print(json.dumps(dict(packet=str(packet),source_models=34,complete_source_cohorts=sum(x['full_five_seed_cohort_complete'] for x in summaries),
    paper_vs_two_data_protocols=comparisons,paired_data_differences=paired_summaries)))
