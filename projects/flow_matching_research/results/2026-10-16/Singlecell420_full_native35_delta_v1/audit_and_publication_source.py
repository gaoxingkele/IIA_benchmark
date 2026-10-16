from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import importlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching import summarize_otcfm_Multiome_full120_paper_rows_v1 as multi
from scripts.flow_matching import summarize_otcfm_EB_CITE_full150_paper_rows_v1 as eb
from scripts.flow_matching import run_otcfm_CITE13176_GEO_native60_v1 as geo
from scripts.flow_matching import recover_otcfm_completed1000_CPU_test_v1 as recovery

specs=[('Multiome120','run_otcfm_singlecell_native_v4','audit_otcfm_singlecell_full_replay_v2','otcfm_singlecell_result_integrity_v2',multi.validate_native),
    ('EB_CITE150','run_otcfm_EB_CITE_native150_v2','audit_otcfm_EB_CITE_full150_replay_v1','otcfm_EB_CITE_result_integrity_v1',eb.validate_native),
    ('EB100_90','run_otcfm_EB100_code_referenced_native90_v1','audit_otcfm_EB100_full90_replay_v1','otcfm_EB100_result_integrity_v1',eb.validate_native)]
rows=[];details={};proofs={};scalars=[];states=[];runtime_counts={};used_configs=set()
recovery_c=read(ROOT/recovery.CONFIG);recovery_state=read(ROOT/recovery_c['state_root']/'status.json')
for family,sname,rname,iname,validator in specs:
    source=importlib.import_module('scripts.flow_matching.'+sname)
    replay=importlib.import_module('scripts.flow_matching.'+rname)
    integrity=importlib.import_module('scripts.flow_matching.'+iname)
    q=read(ROOT/source.QUEUE);source.verify(q)
    rc=read(ROOT/replay.CONFIG);replay.verify(rc)
    used_configs.update([source.QUEUE,replay.CONFIG])
    source_state=read(ROOT/q['state_root']/'status.json');native_state=read(ROOT/rc['state_root']/'status.json')
    states.extend([source_state,native_state])
    runtime_counts[family]=dict(source_completed=len(source_state['completed_ids']),
        native_completed=len(native_state['verified_ids']),required_models=len(q['jobs']))
    for job in q['jobs']:
        key=Path(job['artifact_directory']).name;identity=job['id']
        sgp=ROOT/q['state_root']/'attempts'/key/'guard_receipt.json'
        ngp=ROOT/rc['state_root']/'attempts'/key/'guard_receipt.json'
        rp=Path(rc['artifact_root'])/key/'full_checkpoint_replay.json'
        row=dict(id=identity,family=family,dataset=job['dataset'],method=job['method'],protocol=job['protocol'],
            training_seed=job['seed'],leaveout_timepoint=job['leaveout_timepoint'],
            state='source_not_complete',full_source_complete=False,full_native_complete=False,
            full_cells=job['data']['cells'],original_input_sha256=job['data']['sha256'],
            selected_dimensions=5,full_input_PC_dimensions=job['data'].get('full_original_PC_dimensions'),
            actual_completed_epochs=None,actual_full_training_updates=None,selected_checkpoint_updates=None,
            heldout_W1=None,source_result_sha256=None,native_report_sha256=None,
            known_failure_recovery=None,historical_paper_equivalence_certified=False)
        details[identity]=dict(job=job,source_queue=source.QUEUE,replay_configuration=replay.CONFIG)
        if sgp.exists():
            sg=read(sgp)
            if not sg['passed']:
                row['state']='source_failure_preserved'
                if identity==recovery_state['canonical_id']:
                    row['known_failure_recovery']=recovery.CONFIG
                    row['state']='source_failure_preserved_completed1000_training_recovery_pending'
            else:
                result,provenance=integrity.inspect(q,job)
                row.update(state='full_source_complete_native_pending',full_source_complete=True,
                    actual_completed_epochs=result['actual_completed_epochs'],
                    actual_full_training_updates=result['epoch_history'][-1]['optimizer_updates'],
                    selected_checkpoint_updates=result.get('posttest_selected_checkpoint_optimizer_updates',result['optimizer_updates']),
                    heldout_W1=result['primary_heldout_W1'],source_result_sha256=provenance['result_sha256'])
                details[identity].update(source_result=result,source_provenance=provenance)
                if ngp.exists():
                    ng=read(ngp)
                    if not ng['passed']:row['state']='full_source_complete_native_failure_preserved'
                    else:
                        report=read(rp)
                        validator(job,result,provenance,report,ng,sha(rp),sha(ROOT/replay.CONFIG))
                        for artifact in report['all_full_test_arrays_and_artifacts']:
                            p=Path(artifact['path']);assert p.stat().st_size==artifact['bytes'] and sha(p)==artifact['sha256']
                        row.update(state='full_source_and_native_verified',full_native_complete=True,native_report_sha256=sha(rp))
                        proofs[identity]=dict(source_result_path=provenance['result_path'],source_result_sha256=provenance['result_sha256'],
                            source_guard_path=str(sgp),source_guard_sha256=sha(sgp),native_report_path=str(rp),native_report_sha256=sha(rp),
                            native_guard_path=str(ngp),native_guard_sha256=sha(ngp),
                            independent_full_distribution_LP_solves=report['full_test_population_distances_independently_recomputed'])
                        for metric,value in sorted(result['test_metrics'].items()):
                            scalars.append(dict(id=identity,metric=metric,original_complete_source_value=value,
                                fresh_native_value=report['native_metrics'][metric],
                                absolute_difference=abs(value-report['native_metrics'][metric]),rtol=2e-6,atol=2e-6))
        rows.append(row)
assert len(rows)==360
plan=geo.read_plan();used_configs.add(geo.PLAN)
assert geo.ready_preparation() is None and not (ROOT/geo.QUEUE).exists()
for job in plan['planned_cases']:
    rows.append(dict(id=job['id'],family='GEO_CITE60_deferred',dataset=job['dataset'],method=job['method'],
        protocol=job['protocol'],training_seed=job['seed'],leaveout_timepoint=job['leaveout_timepoint'],
        state='full_raw_QC_PCA_preprocessing_not_complete',full_source_complete=False,full_native_complete=False,
        full_cells=29394,original_input_sha256=None,selected_dimensions=5,full_input_PC_dimensions=None,
        actual_completed_epochs=None,actual_full_training_updates=None,selected_checkpoint_updates=None,
        heldout_W1=None,source_result_sha256=None,native_report_sha256=None,known_failure_recovery=None,
        historical_paper_equivalence_certified=False))
assert len(rows)==len({r['id'] for r in rows})==420
groups=defaultdict(list)
for row in rows:groups[(row['dataset'],row['method'],row['protocol'],row['leaveout_timepoint'])].append(row)
cohorts=[]
for key,items in sorted(groups.items()):
    assert len(items)==5 and {x['training_seed'] for x in items}=={42,43,44,45,46}
    complete=all(x['full_native_complete'] for x in items)
    group=dict(dataset=key[0],method=key[1],protocol=key[2],leaveout_timepoint=key[3],
        expected_training_seeds=[42,43,44,45,46],native_complete_seeds=sorted(x['training_seed'] for x in items if x['full_native_complete']),
        complete=complete,statistics=None)
    if complete:
        assert len({x['original_input_sha256'] for x in items})==1
        values=[x['heldout_W1'] for x in sorted(items,key=lambda x:x['training_seed'])]
        mean=statistics.mean(values);sd=statistics.stdev(values);half=2.7764451051977987*sd/math.sqrt(5)
        group['statistics']=dict(n=5,mean=mean,training_seed_sample_SD=sd,mean_t95_interval=[mean-half,mean+half],
            actual_training_epochs=sorted({x['actual_completed_epochs'] for x in items}),
            actual_full_training_updates=sorted({x['actual_full_training_updates'] for x in items}),
            heldout_W1_by_seed={str(x['training_seed']):x['heldout_W1'] for x in items})
    cohorts.append(group)
assert len(cohorts)==84
selected=[g for g in cohorts if g['method']=='cfm' and g['protocol']=='released_yaml' and
    ((g['dataset']=='released_author_EB' and g['leaveout_timepoint'] in [1,2])
     or (g['dataset']=='trajectorynet_EB100_code_referenced' and g['leaveout_timepoint']==1))]
assert len(selected)==3 and all(g['complete'] for g in selected)
counts=dict(planned_original_cases=420,source_complete=sum(r['full_source_complete'] for r in rows),
    full_native_complete=sum(r['full_native_complete'] for r in rows),
    complete_five_seed_fixed_leaveout_cohorts=sum(g['complete'] for g in cohorts),required_fixed_leaveout_cohorts=84,
    states=dict(Counter(r['state'] for r in rows)),source_runtime_counts=runtime_counts,full_scalar_metric_rows=len(scalars))
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/Singlecell420_full_native30_delta_v4'
old_proofs=read(prior/'all_complete_source_native_case_proof_bindings.json')['records']
assert len(old_proofs)==30 and all(proofs[k]==v for k,v in old_proofs.items())
with (prior/'all420_original_case_execution_matrix.csv').open(encoding='utf-8-sig',newline='') as f:
    old_rows={r['id']:r for r in csv.DictReader(f)}
new_sources=sorted(r['id'] for r in rows if r['full_source_complete'] and old_rows[r['id']]['full_source_complete']!='True')
new_native=sorted(set(proofs)-set(old_proofs))
assert len(new_sources)==6 and len(new_native)==5
assert new_native==sorted(['otcfm_singlecell__released_yaml__cfm__released_author_EB__leave3__seed46']+
    [f'otcfm_singlecell__released_yaml__cfm__trajectorynet_EB100_code_referenced__leave2__seed{s}' for s in [43,44,45,46]])
assert counts['source_complete']==42 and counts['full_native_complete']==35
assert counts['complete_five_seed_fixed_leaveout_cohorts']==6 and len(scalars)==1452
with (prior/'all_complete_native_original_scalar_comparisons.csv').open(encoding='utf-8-sig',newline='') as f:
    old_scalars=list(csv.DictReader(f))
scalar_index={(r['id'],r['metric']):{k:str(v) for k,v in r.items()} for r in scalars}
assert len(old_scalars)==1232 and all(scalar_index[r['id'],r['metric']]==r for r in old_scalars)
old_groups=read(prior/'all84_fixed_leaveout_five_seed_cohort_completion.json')['cohorts']
keys=['dataset','method','protocol','leaveout_timepoint']
new_groups=[]
for group in cohorts:
    previous=next(g for g in old_groups if all(g[k]==group[k] for k in keys))
    if previous['complete']:assert json.loads(json.dumps(group))==previous
    elif group['complete']:new_groups.append(group)
assert len(new_groups)==1
new_group=new_groups[0]
assert new_group['dataset']=='trajectorynet_EB100_code_referenced' and new_group['method']=='cfm' and new_group['leaveout_timepoint']==2
eb3=next(g for g in cohorts if g['dataset']=='released_author_EB' and g['method']=='cfm' and g['protocol']=='released_yaml' and g['leaveout_timepoint']==3)
assert eb3['native_complete_seeds']==[42,43,44,46] and not eb3['complete'] and eb3['statistics'] is None
for family,value in runtime_counts.items():
    assert sum(r['full_source_complete'] for r in rows if r['family']==family)==value['source_completed']
    assert sum(r['full_native_complete'] for r in rows if r['family']==family)==value['native_completed']
xml=ROOT/'tmp/Singlecell420_source_increment_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==53 and not any(int(s.get(k,0)) for s in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Singlecell420_full_native35_delta_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8',newline='\n')
for name,values in [('all420_original_case_execution_matrix.csv',rows),('all_complete_native_original_scalar_comparisons.csv',scalars)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
write(packet/'all84_fixed_leaveout_five_seed_cohort_completion.json',dict(cohorts=cohorts,incomplete_cohorts_have_no_numeric_summary=True,not_paper_averages_over_all_leaveouts=True))
write(packet/'all_complete_source_native_case_proof_bindings.json',dict(records=proofs,every_source_artifact_and_full_replay_array_rehashed=True,extra_model_training_inference_or_LP_execution=False))
for identity in sorted(set(new_native)|set(new_sources)):
    folder=packet/'new_cases'/identity;folder.mkdir(parents=True)
    d=details[identity];provenance=d['source_provenance']
    shutil.copyfile(provenance['result_path'],folder/'source_result.json')
    shutil.copyfile(provenance['guard_path'],folder/'source_guard.json')
    write(folder/'source_provenance.json',provenance)
    if identity in new_native:
        for label,key in [('full_native_report','native_report_path'),('full_native_guard','native_guard_path')]:
            shutil.copyfile(proofs[identity][key],folder/(label+'.json'))
write(packet/'delta_from_prior30_complete_native_cases.json',dict(new_native_ids=new_native,new_source_ids=new_sources,new_full_native_cases=5,new_full_source_cases=6,new_scalar_comparisons=220,prior30_full_native_proofs_unchanged=True,prior1232_scalar_rows_unchanged=True,prior_five_complete_groups_unchanged=True,new_complete_fixed_leaveout_cohort=new_group,publication_training_updates=0))
write(packet/'iteration_status.json',dict(**counts,original_scope=read(ROOT/multi.CONFIG)['original_scope'],original_training_resource_gates_lowered=False,original_numerical_workers_stopped=False,all_original_experiments_complete=False))
live=[]
for state in states:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    live.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),captured_state=state['state']))
write(packet/'actual_live_original_source_and_replay_controllers.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=live))
write(packet/'frozen_configuration_bindings.json',dict(configurations=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(used_configs)]))
v=new_group['statistics'];lo,hi=v['mean_t95_interval']
text=f'''# 单细胞全420条件：35个完整原生核验条件

全量重核360个已物化条件，并保留60个GEO原始预处理条件。源执行42/420，独立完整检查点重放及分布距离核验35/420，完整固定留出五种子组6/84。相对前次30个完整条件新增5个完整重放、6个源执行和220行标量对照；累计1452行标量不是1452次模型训练。

新增完整组为 trajectorynet_EB100_code_referenced 输入上的 I-CFM、released_yaml、固定留出点2，种子42至46。heldout W1 均值 {v['mean']:.12f}，种子样本SD {v['training_seed_sample_SD']:.12f}，均值t95区间 [{lo:.12f}, {hi:.12f}]。完整训练更新数 {v['actual_full_training_updates']}。这是固定留出组，论文整行仍需全部留出点完成。

发布版EB的I-CFM留出点3仅种子42、43、44、46完成独立重放；seed45虽有完整源执行，独立重放仍待完成，不计算部分五种子统计。OT-CFM donor32606留出点1/seed45源执行已完成1000轮、10000次更新，验证选择第320轮/3200更新检查点，原全测试heldout W1为0.769962728023529；独立重放尚未完成，不计入35个完整条件。

每个完整条件的原训练守卫、全部源产物、原生重放守卫、完整测试数组、全部原指标及完整分布LP证明均重新核验。W1保留独立双精度完整成本与LP目标扰动界核验；W2沿用完整原生LP核验。此前30条证明、1232行标量和5组统计保持。53项相关测试通过，归档未额外训练、推理或求解LP。

不同数据版本、PC坐标和白化尺度分轨报告，不声明历史作者输入等价或纯输入版本因果效应。失败源执行及隔离恢复路线保留；未完成指标留空，全部420条件、84组及全研究45论文/681方法记录/273数据任务/702论文标量范围保持，整体未完成。
'''
(packet/'README.md').write_text(text,encoding='utf-8',newline='\n')
shutil.copyfile(xml,packet/'fresh53_source_and_complete_paper_row_validation.xml')
shutil.copyfile(__file__,packet/'audit_and_publication_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Singlecell420_full_native35_delta.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),**counts,additional_training_updates_during_publication=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst'
page.write_text(page.read_text(encoding='utf-8')+f'\n`单细胞全420条件完整重核：源42／原生35，完整五种子组6／84 <results/2026-10-16/{packet.name}/README.md>`_\n新增5个完整重放、6个源条件、220行标量；EB100 I-CFM固定留出点2五种子组完成，其余部分组不聚合。53项检查通过，原证明与统计保持，全研究未完成。\n',encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/Singlecell420_native35_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),counts=counts,new_native_ids=new_native),ensure_ascii=False))
