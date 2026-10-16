from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import audit_crossad_same_score_protocol_factorial_v1 as s
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as m
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as g
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT/s.CONFIG)
q, upstream = s.verify(c)
mc = read(ROOT/m.CONFIG)
_, full_proof_config = m.verify(mc)
cc = read(ROOT/cold.CONFIG)
original_collector, _ = cold.verify(cc)
expected = {f'crossad_author__fresh_train__gecco__seed{x}' for x in g.SEEDS}|{f'crossad_ablation__row1__gecco__seed{x}' for x in g.SEEDS}
assert len(expected)==10
states = {}
for label, config, key, failures in [('protocol',c,'verified_ids','failed_preserved'),
    ('independent_metrics',mc,'verified_ids','failed_preserved'),
    ('full_array_collector',cc,'verified_original_CUDA_ids','errors_preserved')]:
    state = read(ROOT/config['state_root']/'status.json')
    assert expected.issubset(state[key]) and not state[failures]
    handle = psutil.Process(state['pid'])
    assert handle.is_running() and handle.create_time() == state['create_time']
    states[label] = state

tests = ROOT/'tmp/CrossAD_two_complete_cohorts_factorial_validation_20261016_v1.xml'
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites) == 13
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])

proofs, strict, factorial = {}, {}, {}
source_results, bindings = {}, {}
for identity in sorted(expected):
    job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
    proof_path = Path(original_collector['artifact_root'])/'conditions'/(identity+'.json')
    proofs[identity] = read(proof_path)
    assert g.condition(original_collector,q,job) == proofs[identity]
    mp = Path(mc['artifact_root'])/'conditions'/(identity+'.json')
    strict[identity] = read(mp)
    assert m.audit(mc,q,full_proof_config,job) == strict[identity]
    fp = Path(c['artifact_root'])/'conditions'/(identity+'.json')
    factorial[identity] = read(fp)
    assert s.audit(c,q,upstream,job) == factorial[identity]
    for config,path in [(mc,mp),(c,fp)]:
        guard = read(ROOT/config['state_root']/'guards'/(identity+'.json'))
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
        assert guard['report_sha256']==sha(path)
    source_job = read(ROOT/q['delegated_first_configuration'])['job'] if identity==q['delegated_first_original_condition'] else job
    source_path = ROOT/source_job['output_directory']/'result.json'
    source_results[identity] = read(source_path)
    assert source_results[identity]['details']['native_points']==dict(train=55408,validation=13852,test=69261)
    bindings[identity] = dict(source_result=str(source_path),source_result_sha256=sha(source_path),
        actual_epochs=source_results[identity]['details']['epochs_completed'],
        actual_training_forward_calls=source_results[identity]['details']['training_forward_calls'],
        full_checkpoint_array_proof_sha256=sha(proof_path), independent_metric_report_sha256=sha(mp),
        factorial_report_sha256=sha(fp), native_points=strict[identity]['complete_points'],
        raw_sha256=job['raw_sha256'], complete_payload_sha256=strict[identity]['all_score_label_and_timestamp_payload_sha256'])

cohorts = g.cohorts(q['jobs'],proofs)
factorial_cohorts = s.cohort_summary(q,factorial)
assert cohorts['complete_primary_cohorts']==factorial_cohorts['complete_primary_cohorts']==2
assert cohorts['primary_cohorts']['GECCO/main']['verified_seed_ids']==g.SEEDS
saved_all22=read(Path(c['artifact_root'])/'all22_factorial_cohorts.json')
assert saved_all22['complete_primary_cohorts']==2
for key in ['GECCO/main','GECCO/paper_reconstruction_row1']:
    assert factorial_cohorts['cohorts'][key]==saved_all22['cohorts'][key]
assert len({json.dumps(dict(raw=x['raw_sha256'],points=x['native_points'],
    labels=x['complete_payload_sha256']['labels'],rows=x['complete_payload_sha256']['source_rows']),sort_keys=True)
    for x in bindings.values()})==1

# Every prespecified budget/protocol is retained; the primary cell stays validation_raw/1%.
protocol_rows=[];summary_rows=[];paired_rows=[]
for identity,report in sorted(factorial.items()):
    architecture='row1' if report['ablation_row']==1 else 'main'
    for ratio,cells in report['protocol_factorial'].items():
        for protocol,values in cells['protocols'].items():
            protocol_rows.append(dict(canonical_id=identity,architecture=architecture,training_seed=report['seed'],
                alarm_budget_percent=ratio,protocol=protocol,**values,primary=(ratio=='1' and protocol=='validation_raw')))
for name in ['GECCO/main','GECCO/paper_reconstruction_row1']:
    group=factorial_cohorts['cohorts'][name]
    assert group['complete'] and group['verified_training_seeds']==g.SEEDS
    for ratio,values in group['by_alarm_budget'].items():
        for family,items in values.items():
            for metric,value in items.items():
                summary_rows.append(dict(cohort=name,alarm_budget_percent=ratio,family=family,metric=metric,
                    n=value['n'],mean=value['mean'],seed_sample_SD=value['seed_sample_SD'],
                    mean_t95_low=value['mean_t95_interval'][0],mean_t95_high=value['mean_t95_interval'][1]))
for ratio in ['0.1','1','5']:
    for protocol in s.PROTOCOLS:
        seed_deltas={str(seed):factorial[f'crossad_ablation__row1__gecco__seed{seed}']['protocol_factorial'][ratio]['protocols'][protocol]['f1']-
            factorial[f'crossad_author__fresh_train__gecco__seed{seed}']['protocol_factorial'][ratio]['protocols'][protocol]['f1'] for seed in g.SEEDS}
        v=g.interval(list(seed_deltas.values()))
        paired_rows.append(dict(alarm_budget_percent=ratio,protocol=protocol,contrast='row1_minus_main',n=v['n'],mean=v['mean'],
            seed_sample_SD=v['seed_sample_SD'],mean_t95_low=v['mean_t95_interval'][0],mean_t95_high=v['mean_t95_interval'][1],
            seed_deltas=json.dumps(seed_deltas,sort_keys=True),diagnostic_only=(ratio!='1' or protocol!='validation_raw')))
assert len(protocol_rows)==120 and len(summary_rows)==54 and len(paired_rows)==12
primary=next(r for r in paired_rows if r['alarm_budget_percent']=='1' and r['protocol']=='validation_raw')
closure=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1107_full28_v1'
prior_cohorts=read(closure/'all28_primary_cohort_completion.json')
reference=prior_cohorts['paired_seed_ablation_differences']['GECCO/paper_reconstruction_row1_minus_main']['validation_false_alarm_percent/1/micro/f1']
assert abs(primary['mean']-reference['mean'])<1e-14
prior_main=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_first_complete_five_seed_GECCO_and_same_score_protocol_factorial_v1'
for seed in g.SEEDS:
    identity=f'crossad_author__fresh_train__gecco__seed{seed}'
    assert factorial[identity]==read(prior_main/'all5_complete_conditions'/identity/'protocol_factorial_report.json')

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_two_complete_five_seed_cohorts_same_score_factorial_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8',newline='\n')
for identity in sorted(expected):
    folder=packet/'all10_complete_conditions'/identity;folder.mkdir(parents=True)
    paths=dict(full_checkpoint_array_proof=Path(original_collector['artifact_root'])/'conditions'/(identity+'.json'),
        independent_full_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
        protocol_factorial_report=Path(c['artifact_root'])/'conditions'/(identity+'.json'),
        independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
        protocol_factorial_guard=ROOT/c['state_root']/'guards'/(identity+'.json'))
    for name,path in paths.items():shutil.copyfile(path,folder/(name+'.json'))
for name,values in [('all10_models_120_protocol_cells.csv',protocol_rows),('all54_complete_cohort_protocol_summaries.csv',summary_rows),('all12_same_seed_method_contrasts.csv',paired_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
write(packet/'all10_complete_model_and_data_bindings.json',dict(records=bindings,source_model_count=10,
    per_model_points=138521,same_all_original_source_rows_labels_and_tail=True,original_CPU_completion_credit=0,strict_TAB_result=False))
write(packet/'two_complete_cohort_summaries.json',dict(cohorts={key:factorial_cohorts['cohorts'][key] for key in ['GECCO/main','GECCO/paper_reconstruction_row1']},
    original_required_primary_cohorts=22,other_partial_cohorts_not_aggregated=True))
write(packet/'frozen_all22_captured_cohort_status.json',saved_all22)
write(packet/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),states=states))
write(packet/'frozen_configuration_and_source_bindings.json',dict(paths=[dict(path=p,sha256=sha(ROOT/p)) for p in [s.CONFIG,m.CONFIG,cold.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST]],
    prior_main_five_seed_factorial_unchanged=True,strict_five_seed_closure_path=closure.relative_to(ROOT).as_posix(),strict_five_seed_closure_sha256=sha(closure/'packet_integrity.json')))
main=factorial_cohorts['cohorts']['GECCO/main']['by_alarm_budget']['1']['protocol_F1']
row1=factorial_cohorts['cohorts']['GECCO/paper_reconstruction_row1']['by_alarm_budget']['1']['protocol_F1']
table=['| 1%阈值口径 | 主模型F1均值 | 重建row1 F1均值 | row1减主模型的成对均值t95区间 |',
       '|---|---:|---:|---|']
for protocol in s.PROTOCOLS:
    v=next(r for r in paired_rows if r['alarm_budget_percent']=='1' and r['protocol']==protocol)
    table.append(f'| {protocol} | {main[protocol]["mean"]:.9f} | {row1[protocol]["mean"]:.9f} | [{v["mean_t95_low"]:.9f}, {v["mean_t95_high"]:.9f}] |')
text='''# CrossAD两个完整五种子组：同分数评估口径对照

主模型和论文重建row1均使用完整种子1103至1107。重新逐项核对10个完整模型的检查点/全数组证明、独立严格指标和既定2×2口径诊断；所有预定0.1%、1%、5%阈值、四种口径全部保留。120个条件口径单元、54个完整组汇总和12个同种子方法差值不是新增模型训练。

'''+ '\n'.join(table)+'''

validation_raw/1%仍是主要验证阈值、无PA结果。PA使用测试异常事件标签回填事件；pooled_train_test用训练和测试分数共同定阈值，未用测试标签选择阈值。其余格子均作为固定分数诊断，不能冒充可部署验证校准成绩、原始TAB完整协议或历史作者复现。

1%下主模型的无PA均值高于row1；验证校准加PA后两者均值接近，方法成对区间需同时阅读，不能仅凭均值改变声称优势。混合训练/测试分数校准使主模型F1上升、row1下降，说明该口径在本组并非对所有方法同向提高。全部预算与交互作用见CSV，不事后择优阈值或口径。

区间仅描述同一冻结数据划分上五个训练种子的波动，不是未知实体、数据集、历史作者误差或多重比较显著性。原主模型五种子诊断保持，当前严格成对主结果也保持。重建消融的作者精确层/头选择未获认证。核验没有新增训练、神经推理或原始VUS/affiliation/bootstrap重跑。

全部139条件、22主要组及45论文/681方法记录/273数据任务/702论文标量目标保持；其他四项消融各4/5，不聚合。整体研究未完成。
'''
(packet/'README.md').write_text(text,encoding='utf-8',newline='\n')
shutil.copyfile(tests,packet/'fresh13_full_metric_and_factorial_validation.xml')
shutil.copyfile(__file__,packet/'audit_and_publication_source.py')
write(packet/'iteration_status.json',dict(full_source_models_reviewed=10,complete_primary_cohorts_reviewed=2,required_original_conditions=139,required_original_primary_cohorts=22,
    protocol_cells=120,complete_cohort_summaries=54,paired_method_contrasts=12,new_optimizer_updates=0,new_neural_inference_executed=False,
    original_primary_result_unchanged=True,original_scope=c['original_scope'],all_original_experiments_complete=False))
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_two_complete_cohorts_same_score_factorial.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),complete_model_groups=2,full_source_models_reviewed=10,new_model_training_updates=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst'
page.write_text(page.read_text(encoding='utf-8')+f'\n`CrossAD两个完整五种子组的同分数口径对照 <results/2026-10-16/{packet.name}/README.md>`_\n全部3阈值×4口径×10模型及54组汇总/12个同种子方法差值完整核验；严格主要结果保持，测试分数校准和PA单列诊断，不事后选择口径。全研究未完成。\n',encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/CrossAD_two_complete_cohorts_factorial_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(packet=str(packet),files=len(paths),primary=primary,one_percent_method_contrasts=[r for r in paired_rows if r['alarm_budget_percent']=='1']))
