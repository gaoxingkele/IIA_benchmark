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
expected = {f'crossad_author__fresh_train__gecco__seed{x}' for x in g.SEEDS}
states = {}
for label, config, key, failures in [('protocol',c,'verified_ids','failed_preserved'),
    ('independent_metrics',mc,'verified_ids','failed_preserved'),
    ('full_array_collector',cc,'verified_original_CUDA_ids','errors_preserved')]:
    state = read(ROOT/config['state_root']/'status.json')
    assert expected.issubset(state[key]) and not state[failures]
    handle = psutil.Process(state['pid'])
    assert handle.is_running() and handle.create_time() == state['create_time']
    states[label] = state

tests = Path('F:/aicoding/IIA_Data/experiments/CrossAD_same_score_factorial_20261016_v1_tests.xml')
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
assert cohorts['complete_primary_cohorts']==factorial_cohorts['complete_primary_cohorts']==1
assert cohorts['primary_cohorts']['GECCO/main']['verified_seed_ids']==g.SEEDS
assert factorial_cohorts == read(Path(c['artifact_root'])/'all22_factorial_cohorts.json')
assert len({json.dumps(dict(raw=x['raw_sha256'],points=x['native_points'],
    labels=x['complete_payload_sha256']['labels'],rows=x['complete_payload_sha256']['source_rows']),sort_keys=True)
    for x in bindings.values()})==1

packet = ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_first_complete_five_seed_GECCO_and_same_score_protocol_factorial_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.log -text -whitespace\n*.py -text\n',encoding='utf-8')
for identity in sorted(expected):
    folder = packet/'all5_complete_conditions'/identity
    folder.mkdir(parents=True)
    paths = dict(full_checkpoint_array_proof=Path(original_collector['artifact_root'])/'conditions'/(identity+'.json'),
        independent_full_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
        protocol_factorial_report=Path(c['artifact_root'])/'conditions'/(identity+'.json'),
        independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
        protocol_factorial_guard=ROOT/c['state_root']/'guards'/(identity+'.json'))
    for label,path in paths.items():shutil.copyfile(path,folder/(label+'.json'))
new_identity = 'crossad_author__fresh_train__gecco__seed1107'
new_job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==new_identity)
assert source_results[new_identity]['details']['epochs_completed']==20
assert source_results[new_identity]['details']['training_forward_calls']==8640
folder = packet/'new_complete_seed1107'
folder.mkdir()
paths = dict(source_result=ROOT/new_job['output_directory']/'result.json',
    training_trace=ROOT/new_job['output_directory']/'training_trace.json',
    native_replay=Path(q['artifact_root'])/new_identity/'native_replay/report.json',
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/new_identity/(phase+'.guard.json')
       for phase in ['CUDA_probe','CPU_reference','author','replay']})
for label,path in paths.items():shutil.copyfile(path,folder/(label+'.json'))

rows, strict_rows, summary_rows = [], [], []
for identity in sorted(expected):
    for ratio, group in factorial[identity]['protocol_factorial'].items():
        for protocol,values in group['protocols'].items():
            rows.append(dict(canonical_id=identity,dataset='GECCO',training_seed=factorial[identity]['seed'],
                alarm_budget_percent=ratio,protocol=protocol,**values,full_TAB_certification=False))
    for ratio,values in strict[identity]['independent_strict_metrics'].items():
        strict_rows.append(dict(canonical_id=identity,training_seed=strict[identity]['seed'],
            alarm_budget_percent=ratio,threshold=values['threshold'],**values['micro'],**values['event_metrics']))
for ratio,group in factorial_cohorts['cohorts']['GECCO/main']['by_alarm_budget'].items():
    for family,values in group.items():
        for key,value in values.items():
            summary_rows.append(dict(dataset='GECCO',alarm_budget_percent=ratio,metric_family=family,
                metric=key,training_seeds='1103,1104,1105,1106,1107',n=value['n'],mean=value['mean'],
                seed_sample_SD=value['seed_sample_SD'],mean_t95_low=value['mean_t95_interval'][0],
                mean_t95_high=value['mean_t95_interval'][1]))
for name,values in [('all5_models_60_protocol_rows.csv',rows),('all5_models_15_strict_metric_rows.csv',strict_rows),
                    ('five_seed_27_protocol_and_paired_difference_summaries.csv',summary_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
write(packet/'all5_complete_model_and_data_bindings.json',dict(records=bindings,source_model_count=5,
    per_model_points=138521,original_CPU_completion_credit=0,strict_TAB_result=False))
write(packet/'all22_original_CUDA_cohorts.json',cohorts)
write(packet/'all22_same_score_protocol_cohorts.json',factorial_cohorts)
for label,state in states.items():write(packet/(label+'_actual_state_snapshot.json'),state)
shutil.copyfile(tests,packet/'validation_13_checks.xml')
shutil.copyfile(__file__,packet/'closure_source.py')
shutil.copyfile(ROOT/'tmp/launch_crossad_same_score_factorial_20261016_v1.py',packet/'launch_source.py')
launch = Path('F:/aicoding/IIA_Data/experiments/crossad_same_score_protocol_factorial_v1_launch')
for name in ['launch.json','stdout.log','stderr.log']:shutil.copyfile(launch/name,packet/('actual_launch_'+name))

handles = []
for state in list(states.values())+[read(ROOT/q['state_root']/'status.json')]:
    p = psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])))
for pid,ct in [(19444,1792131765.9271505),(39064,1792133880.0474632),
    (36912,1792097263.1796665),(41992,1791948094.8603492),(37832,1792132776.595039)]:
    p = psutil.Process(pid);assert p.is_running() and p.create_time()==ct
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])))
write(packet/'actual_live_original_worker_and_controller_handles.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))

primary = factorial_cohorts['cohorts']['GECCO/main']['by_alarm_budget']['1']
table = ['| Calibration / PA | Five-seed mean F1 | Seed t95 interval |','|---|---:|---|']
for protocol in s.PROTOCOLS:
    value = primary['protocol_F1'][protocol];low,high=value['mean_t95_interval']
    table.append(f'| {protocol} | {value["mean"]:.6f} | [{low:.6f}, {high:.6f}] |')
strict_group = cohorts['primary_cohorts']['GECCO/main']['metrics']
roc = strict_group['validation_false_alarm_percent/1/micro/auroc']
ap = strict_group['validation_false_alarm_percent/1/micro/average_precision']
combined = primary['paired_F1_differences']['combined']
readme = '''# CrossAD GECCO: first complete five-seed cohort and fixed-score protocol diagnostics

The five prespecified training seeds 1103..1107 completed the frozen original CUDA/FP32 recipe and full fresh checkpoint replay.
The newest seed1107 completed all20 epochs and8640 training forwards. Seed1104 legitimately stopped at19 epochs under the original patience; other four completed20.
Each model covers all55408 train,13852 validation and69261 test points, including the13-point tail. Source and fresh replay score payload bytes match exactly for every split.
All five source file hashes, labels and original row identities match. The independent standard-library audit recalculated full thresholds, confusion counts, F1, ROC-AUC, AP and event diagnostics.
Source and replay resource guards passed with exit0 and no memory/GPU abort. GPU results never replace the separately required original CPU experiments.

The new diagnostic registry retains all139 conditions and22 complete primary cohorts. Each completed model is analyzed at all three prespecified alarm budgets0.1/1/5%, crossing validation-only versus pooled train-plus-test score calibration with raw versus point-adjusted predictions.
All four cells use identical complete model scores and identical float64 linear quantiles/strict greater-than. Pooled calibration uses test scores without test labels; PA uses test event labels to fill hit contiguous anomaly events. No threshold is chosen using test labels.
This calculation adds zero model training updates and zero neural forwards. It is an evaluation protocol diagnostic, not a model architecture ablation or a new training seed.
The source concatenates the dataset timeline; original entity boundaries are not reconstructed. Therefore these rows are not certified as full TAB results and are not a numerical reproduction claim for the historical paper table.

Primary1% alarm budget, F1 scale0..1:

'''+ '\n'.join(table)+f'''

The paired combined change is {combined['mean']:.6f} (five-seed descriptive t95 {combined['mean_t95_interval']}).
The raw strict ROC-AUC mean is {roc['mean']:.6f}; AP mean is {ap['mean']:.6f}. High ranking AUC alone does not imply high F1 at the frozen alert budget.
Intervals describe variation across these five fitted training seeds on this single fixed file. They do not quantify unseen-entity uncertainty, establish algorithm superiority, or imply significance across the full experimental family.
Release checks and seed2025 controls are excluded from these five-seed means. No partial cohort receives a mean or interval.

CSV artifacts contain60 individual protocol rows,15 strict metric rows and27 protocol/paired-difference summaries across all three budgets.
All five complete raw proof reports and guard receipts are retained.13 relevant mathematical, provenance and partial-cohort checks passed.

Current CrossAD CUDA closure is5/139 conditions and1/22 primary cohorts;134 conditions and21 cohorts remain. The live original source controller has advanced to the seed2025 control.
All45 papers,681 method/baseline/ablation records,273 original data tasks,702 scalar targets, and the original/TAB/new-data tracks remain in scope. The overall goal is unfinished.
'''
(packet/'README.md').write_text(readme,encoding='utf-8')
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_verified=5,
    CrossAD_required_conditions=139,complete_primary_cohorts=1,required_primary_cohorts=22,
    new_original_CUDA_models=1,new_model_training_updates_from_diagnostic=0,
    same_score_protocol_conditions_verified=5,protocol_rows=60,strict_metric_rows=15,
    between_training_seed_summary_rows=27,original_CPU_completion_credit=0,
    original_resource_gates_lowered=False,original_numeric_workers_stopped=False,all_original_experiments_complete=False))
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT/'projects/flow_matching_research/ara/execution/CrossAD5_same_score_protocol_factorial.2026-10-16_v1.json'
assert not ara.exists()
write(ara,dict(configuration=s.CONFIG,result_packet=packet.relative_to(ROOT).as_posix(),
    primary_training_seeds=g.SEEDS,CrossAD_full_original_CUDA_models=5,required_original_CUDA_conditions=139,
    complete_primary_cohorts=1,required_primary_cohorts=22,same_score_diagnostic_full_conditions=5,
    new_neural_inference_from_diagnostic=False,strict_TAB_result=False,
    original_scope=q['original_scope'],all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths = [s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
manifest = ROOT/'tmp/CrossAD5_same_score_factorial_publication_manifest_20261016_v1.json'
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),exact_files=len(paths),primary_1_percent=primary,
    strict_ROC=roc,strict_AP=ap),ensure_ascii=False))
