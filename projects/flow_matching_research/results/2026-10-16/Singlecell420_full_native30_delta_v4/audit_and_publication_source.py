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
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/Singlecell420_native_matrix_EB5_v2'
prior_counts=read(prior/'iteration_status.json')
prior_proofs=read(prior/'all_complete_source_native_case_proof_bindings.json')['records']
assert len(prior_proofs)==28
for identity,entry in prior_proofs.items():assert proofs[identity]==entry
new_native=sorted(set(proofs)-set(prior_proofs))
assert new_native==[f'otcfm_singlecell__released_yaml__cfm__released_author_EB__leave3__seed{i}' for i in [43,44]]
assert counts['source_complete']==prior_counts['source_complete']+2==36
assert counts['full_native_complete']==prior_counts['full_native_complete']+2==30
assert counts['complete_five_seed_fixed_leaveout_cohorts']==5
assert len(scalars)==prior_counts['full_scalar_metric_rows']+88==1232
with (prior/'all_complete_native_original_scalar_comparisons.csv').open(encoding='utf-8-sig',newline='') as f:old_scalars=list(csv.DictReader(f))
new_scalars={(r['id'],r['metric']):{k:str(v) for k,v in r.items()} for r in scalars}
assert len(old_scalars)==1144 and all(new_scalars[(r['id'],r['metric'])]==r for r in old_scalars)
prior_groups=read(prior/'all84_fixed_leaveout_five_seed_cohort_completion.json')['cohorts']
for group in prior_groups:
    if group['complete']:
        current=next(g for g in cohorts if all(g[k]==group[k] for k in ['dataset','method','protocol','leaveout_timepoint']))
        assert json.loads(json.dumps(current))==group
leave3=next(g for g in cohorts if g['dataset']=='released_author_EB' and g['method']=='cfm' and g['protocol']=='released_yaml' and g['leaveout_timepoint']==3)
assert leave3['native_complete_seeds']==[42,43,44] and not leave3['complete'] and leave3['statistics'] is None

for family,value in runtime_counts.items():
    assert sum(r['full_source_complete'] for r in rows if r['family']==family)==value['source_completed']
    assert sum(r['full_native_complete'] for r in rows if r['family']==family)==value['native_completed']
tests=Path('F:/aicoding/IIA_Data/experiments/singlecell420_full_native30_delta_20261016_v3_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==33 and not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Singlecell420_full_native30_delta_v4';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8')
for name,values in [('all420_original_case_execution_matrix.csv',rows),('all_complete_native_original_scalar_comparisons.csv',scalars)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
write(packet/'all84_fixed_leaveout_five_seed_cohort_completion.json',dict(cohorts=cohorts,
    incomplete_cohorts_have_no_numeric_summary=True,not_paper_averages_over_all_leaveouts=True))
write(packet/'all_complete_source_native_case_proof_bindings.json',dict(records=proofs,
    every_source_artifact_and_full_replay_array_rehashed=True,extra_model_training_inference_or_LP_execution=False))
for identity in new_native:
    folder=packet/'new_cases'/identity;folder.mkdir(parents=True)
    for label,key in [('source_result','source_result_path'),('source_guard','source_guard_path'),
        ('full_native_report','native_report_path'),('full_native_guard','native_guard_path')]:
        shutil.copyfile(proofs[identity][key],folder/(label+'.json'))
write(packet/'delta_from_prior28_full_native_cases.json',dict(new_native_ids=new_native,
    new_full_native_cases=2,new_full_source_cases=2,new_scalar_comparisons=88,
    prior28_full_native_proofs_unchanged=True,prior1144_scalar_rows_unchanged=True,
    prior_complete_five_seed_groups_unchanged=True,publication_training_updates=0))
write(packet/'retained_three_complete_EB_fixed_leaveout_cohorts.json',dict(cohorts=selected,
    cells_PC_whitening_coordinate_systems_and_update_counts_differ=True,
    pure_input_version_causal_effect_certified=False,shared_metric_scale_across_input_versions_certified=False,
    complete_five_seed_paper_rows_claimed_from_these_fixed_leaveouts=False))
write(packet/'pending_completed1000_CPU_recovery_snapshot.json',dict(configuration=recovery.CONFIG,actual_state=recovery_state,
    original_failure_preserved=True,full_test_and_native_replay_still_required=True))
live=[]
for state in states:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    live.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),actual_captured_state=state['state']))
write(packet/'actual_live_original_source_and_replay_controllers.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=live))
write(packet/'iteration_status.json',dict(**counts,original_scope=read(ROOT/multi.CONFIG)['original_scope'],
    original_training_resource_gates_lowered=False,original_numerical_workers_stopped=False,all_original_experiments_complete=False))
table=['| 输入轨道 | 固定留出索引 | 五种子平均W1 | 种子SD | 均值t95区间 | 完整训练更新/模型 |',
    '|---|---:|---:|---:|---|---|']
for g in selected:
    v=g['statistics'];lo,hi=v['mean_t95_interval']
    table.append(f'| {g["dataset"]} | {g["leaveout_timepoint"]} | {v["mean"]:.6f} | {v["training_seed_sample_SD"]:.6f} | [{lo:.6f}, {hi:.6f}] | {v["actual_full_training_updates"]} |')
lines=['# 单细胞全420条件：新增两个完整原生核验条件','',
    '重新核对全部360个已物化CPU条件，另保留60个完整GEO原始RNA准备条件。源完整36/420、完整检查点重放与分布距离核验30/420；完整固定留出五种子组仍5/84。',
    '相对上一快照新增2个原生完整条件、2个源训练完整条件。本次归档没有额外训练、推理或LP求解，不把这些核验重复记作新训练。EB100留出3/seed43也完成了原始训练及测试，原生检查点重放仍待完成，不能计入原生完成数。','',
    '| I-CFM / 发布EB / 固定留出3 | 实际训练轮数 | 实际全部更新 | 选中检查点更新 | heldout W1 |',
    '|---|---:|---:|---:|---:|']
for identity in new_native:
    row=next(r for r in rows if r['id']==identity)
    lines.append(f"| seed{row['training_seed']} | {row['actual_completed_epochs']} | {row['actual_full_training_updates']} | {row['selected_checkpoint_updates']} | {row['heldout_W1']:.9f} |")
lines += ['',
    '新条件的原始训练守卫、所有源产物、独立原生重放守卫、全部生成数组和完整测试分布距离逐项重新核验。原完整44指标在既定2e-6条件内重现；全部时间点使用完整W1/W2最优LP。W1另有独立SciPy双精度完整成本与目标扰动界核验，没有用子采样或近似距离替代。',
    '原28个完整条件的证明结构及所有报告/守卫/数组哈希、原1144行标量比较和既有5个完整五种子组的数值均保持。全量标量对照现1232行，新增88行；这些指标格子不是88次模型训练。',
    '发布EB的I-CFM留出3现在仅种子42/43/44完整，45/46仍必需。该3/5组不计算五种子均值、SD、SE、CI，也不充当论文EB完整行。论文行需要3个留出点各5个模型，共15个完整条件；完整论文行总数未因本次部分组变化增加。',
    '保留全部420行与84组；未完成指标留空，不填0。数据输入版本、PC坐标和白化尺度不同的EB轨道仍分别报告，不在此做纯输入版本因果结论或历史作者等价声明。',
    '33项原完整论文行/种子/留出/部分组拒绝检查重新通过；同时执行所有30个实际完整条件的全量产物和守卫核验。旧OT-CFM完成1000轮但测试失败的记录与隔离恢复路线保留，未按训练完成冒充全测试已成功。',
    '全45论文/681方法基线消融/273数据任务/702既有标量目标、原论文/TAB/新数据的未完成项全部保持，整体目标未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(tests,packet/'validation_33_complete_paper_row_checks.xml');shutil.copyfile(__file__,packet/'audit_and_publication_source.py')
write(packet/'frozen_configuration_bindings.json',dict(configurations=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(used_configs)]))
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Singlecell420_full_native30_delta.2026-10-16_v4.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),**counts,new_model_training_updates=0,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/singlecell420_full_native30_delta_manifest_20261016_v4.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),counts=counts,new_native_ids=new_native)))
