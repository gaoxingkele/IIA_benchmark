"""Archive the first actual complete ten-model row and first original-code EB replay."""
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scipy.stats import t as student_t
from scripts.flow_matching import summarize_otcfm_Multiome_full120_paper_rows_v1 as s
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as eb
from scripts.flow_matching import audit_otcfm_EB100_full90_replay_v1 as er
from scripts.flow_matching.otcfm_EB100_result_integrity_v1 import inspect as inspect_eb


def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def copy(p,t):
    with t.open('xb') as f:f.write(p.read_bytes())
    assert s.sha(p)==s.sha(t)


c=s.read(s.ROOT/s.CONFIG);s.verify(c);q=s.read(s.ROOT/s.native.QUEUE);rc=s.read(s.ROOT/s.replay.CONFIG)
group=next(g for g in c['groups'] if g['id']=='released_author_donor32606__released_yaml__cfm')
rp=Path(c['artifact_root'])/group['id']/'complete_paper_row.json';gp=s.ROOT/c['state_root']/'attempts'/group['id']/'guard_receipt.json'
row,guard=s.read(rp),s.read(gp)
assert row['passed'] and guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert guard['report_sha256']==s.sha(rp) and row['configuration_sha256']==s.sha(s.ROOT/s.CONFIG)
assert row['independently_trained_models']==10 and row['training_seeds']==5 and row['leaveouts_per_seed']==2
assert {r['id'] for r in row['records']}==set(group['required_case_ids'])
actual=s.generic.summarize(row['records'],group)
assert all(row[k]==v for k,v in actual.items())
means=[math.fsum(r['heldout_W1'] for r in row['records'] if r['seed']==seed)/2 for seed in range(42,47)]
mean=math.fsum(means)/5;std=statistics.stdev(means);ci=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5))
assert math.isclose(mean,row['seed_mean_summary']['mean'],rel_tol=1e-12,abs_tol=1e-12)
assert math.isclose(std,row['seed_mean_summary']['std'],rel_tol=1e-12,abs_tol=1e-12)
assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(ci,[row['seed_mean_summary']['ci95_low'],row['seed_mean_summary']['ci95_high']]))
folder=s.ROOT/'projects/flow_matching_research/results/2026-10-16/Multiome_first_full10_row_and_EB100_native1_v1'
assert not folder.exists();archives=[];metric_rows=[];full_cost_entries=0
for identity in group['required_case_ids']:
    job=next(j for j in q['jobs'] if j['id']==identity);source,provenance=s.inspect(q,job)
    sg,ng,np=s.generic.guard_paths(q,rc,job);r,g=s.read(np),s.read(ng)
    s.validate_native(job,source,provenance,r,g,s.sha(np),s.sha(s.ROOT/s.replay.CONFIG))
    assert r['native_metrics']==r['original_metrics']==source['test_metrics'] and len(r['native_metrics'])==36
    assert source['actual_completed_epochs']==1000 and source['epoch_history'][-1]['optimizer_updates']==10000
    assert source['parameter_count']==9093
    for a in r['all_full_test_arrays_and_artifacts']:
        p=Path(a['path']);assert p.stat().st_size==a['bytes'] and s.sha(p)==a['sha256']
    solves=r['full_test_population_distances_independently_recomputed'][0]['solves']
    full_cost_entries+=sum(v['full_cost_shape'][0]*v['full_cost_shape'][1] for v in solves if v['power']==1)
    proof=next(v for v in row['proofs'] if v['id']==identity)
    assert proof['source_provenance']['result_sha256']==provenance['result_sha256']
    assert proof['native_report_sha256']==s.sha(np) and proof['native_guard_sha256']==s.sha(ng)
    archives.append((f"leave{job['leaveout_timepoint']}_seed{job['seed']}",dict(source_result=Path(provenance['result_path']),source_guard=sg,native_report=np,native_guard=ng)))
    for metric,value in source['test_metrics'].items():metric_rows.append([identity,job['seed'],job['leaveout_timepoint'],metric,value,r['native_metrics'][metric],True])
assert full_cost_entries==1554751680 and len(metric_rows)==360

eq=s.read(s.ROOT/eb.QUEUE);ec=s.read(s.ROOT/er.CONFIG);ej=eq['jobs'][0];es,ep=inspect_eb(eq,ej)
esg,eng,enp=s.generic.guard_paths(eq,ec,ej);en,eg=s.read(enp),s.read(eng)
s.generic.validate_native(ej,es,ep,en,eg,s.sha(enp),s.sha(s.ROOT/er.CONFIG))
assert en['native_metrics']==en['original_metrics']==es['test_metrics'] and len(en['native_metrics'])==44
assert es['actual_completed_epochs']==1000 and es['optimizer_updates']==3000 and en['selected_checkpoint_updates']==2610
for a in en['all_full_test_arrays_and_artifacts']:
    p=Path(a['path']);assert p.stat().st_size==a['bytes'] and s.sha(p)==a['sha256']
ebcost=sum(v['full_cost_shape'][0]*v['full_cost_shape'][1] for v in en['full_test_population_distances_independently_recomputed'][0]['solves'] if v['power']==1)
assert ebcost==35572287
folder.mkdir(parents=True)
copy(rp,folder/'complete_ten_model_row.json');copy(gp,folder/'complete_row_guard.json')
for name,paths in archives:
    d=folder/name;d.mkdir()
    for tag,p in paths.items():copy(p,d/(tag+'.json'))
ed=folder/'EB100_seed42_leave1';ed.mkdir()
for tag,p in dict(source_result=Path(ep['result_path']),source_guard=esg,native_report=enp,native_guard=eng).items():copy(p,ed/(tag+'.json'))
with (folder/'all360_Multiome_source_and_native_metrics.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['id','seed','leaveout_timepoint','metric','source','native','exactly_equal']);w.writerows(metric_rows)
with (folder/'all44_EB100_source_and_native_metrics.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['metric','source','native','exactly_equal'])
    for metric,value in es['test_metrics'].items():w.writerow([metric,value,en['native_metrics'][metric],value==en['native_metrics'][metric]])
save(folder/'actual_numeric_comparison_and_fidelity_boundaries.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    Multiome=dict(dataset=group['dataset'],protocol=group['protocol'],method='cfm',cells=33701,
        independently_trained_models=10,original_training_seeds=5,independent_leaveout_models_per_seed=2,
        complete_original_1000epoch_10000update_models=10,full_original_native_metrics_exact=360,
        native_W1_W2_OPTIMAL_solves=40,independent_W1_OPTIMAL_solves=20,full_independent_cost_entries=full_cost_entries,
        seed_level_leaveout_means=means,seed_mean_summary=row['seed_mean_summary'],all10_model_sample_STD=row['all_model_sample_STD'],
        paper_target=row['paper_target'],mean_difference_from_paper=row['mean_difference_from_paper'],
        numeric_mean_close_does_not_certify_historical_reproduction=True,
        historical_donor_mismatch='released author input32606,while paper describes13176',
        paper_SD_aggregation_certified=False),
    EB100=dict(id=ej['id'],cells=16819,raw_PC_dimensions=100,original_epochs=1000,actual_updates=3000,
        selected_checkpoint_updates=2610,selected_epoch=en['selected_checkpoint_epoch'],heldout_W1=es['primary_heldout_W1'],
        all44_native_metrics_exact=True,native_W1_W2_OPTIMAL_solves=6,independent_W1_OPTIMAL_solves=3,
        independent_full_cost_entries=ebcost,full15_model_EB_row_complete=False),
    full_source_and_all_artifact_hashes_rechecked=True,extra_training_or_reinference=False,
    comparison_protocol_definition='Existing full sources/replays aggregated after completion;not a prospective scientific preregistration.',
    original_source_whitening_uses_all_timepoints=True,original_validation_includes_leftout_timepoint=True,
    original_test_compares_full_timepoint_populations=True,strict_inductive_or_TAB_anomaly_detection_claim=False,
    historical_paper_equivalence_certified=False,original_scope=q['original_scope'],all_original_experiments_complete=False))
live={}
for label,root in [('Multiome_source',q['state_root']),('Multiome_native',rc['state_root']),('Multiome_rows',c['state_root']),('EB100_source',eq['state_root']),('EB100_native',ec['state_root'])]:
    state=s.read(s.ROOT/root/'status.json');p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    live[label]=dict(state=state,actual_handle=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
save(folder/'actual_live_remaining_complete_scope_handles.json',live)
copy(s.ROOT/c['state_root']/'launch.json',folder/'complete120_row_collector_launch.json')
new=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG];pins=[]
for i,path in enumerate(new+[s.native.QUEUE,s.replay.CONFIG,eb.QUEUE,er.CONFIG]):
    p=s.ROOT/path;target=folder/f'source{i:02d}{p.suffix}';copy(p,target);pins.append(dict(path=path,sha256=s.sha(p),snapshot=target.name))
save(folder/'source_manifest.json',pins)
xml=Path('F:/aicoding/IIA_Data/experiments/Multiome_full120_complete_paper_rows_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'));assert sum(int(z.get('tests','0')) for z in suites)==44
assert not any(int(z.get(k,'0')) for z in suites for k in ['errors','failures','skipped'])
copy(xml,folder/'validation_44_checks.xml');copy(Path(__file__),folder/'publication_source.py')
(folder/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.csv -text\n',encoding='utf-8')
(folder/'README.md').write_text('First complete ten-model Multiome comparison row;first EB100full native checkpoint replay\n\n'
    'Multiome released-author donor32606 / released-YAML I-CFM completed all10original models:five training seeds42..46 x independently trained leaveouts1/2. '
    'Each completed1000epochs/10000optimizer updates/9093parameters. All360native/source scalars match exactly;40native W1/W2 and20independent W1 LP solves were OPTIMAL,covering1554751680full independent cost entries. '
    'The row report and its successful metadata guard were rechecked against every source/checkpoint/native/array artifact.\n\n'
    'W1='+str(mean)+';sampleSTD across five seed-level two-leaveout means='+str(std)+';df4t95=['+str(ci[0])+','+str(ci[1])+']. '
    'All10model sampleSTD='+str(row['all_model_sample_STD'])+' is a separate spread definition,not ten independent seeds. '
    'Paper Table4 Multi I-CFM reports1.085+/-0.099;mean difference='+str(row['mean_difference_from_paper'])+'. '
    'The numeric mean is close,but donor32606differs from paper-described13176 and the paper SD convention is uncertified. '
    'Both frozen recipes and both input tracks remain separate;the live collector retains all120conditions in12rows and does not select a closest-to-paper recipe. '
    'The other11rows remain required. These summary definitions were registered after existing source results;no prospective preregistration is claimed.\n\n'
    'Original-code-reference EB100seed42/leaveout1completed full1000epochs/3000updates and selected epoch870/2610updates. '
    'Its full44native metrics matched exactly,including heldoutW1=0.8616111874580383;all6native W1/W2 and3independent W1 full LP solves were OPTIMAL over35572287independent cost entries. '
    'This is one of15conditions required for a complete EB paper row,and is separate from released18203-cell/50-PC EB.\n\n'
    'Source-faithful boundaries remain:whitening uses all timepoints,validation includes the leftout timepoint,and test distributions cover full original timepoint populations. '
    'These are single-cell generative interpolation distances,not anomaly-detection F1,strict inductive performance or TAB evidence. '
    '44relevant contract checks passed. All original45papers/681method-baseline-ablation records/273data tasks/702scalar targets and every other unfinished experiment remain.\n',encoding='utf-8')
paths=new+[p.relative_to(s.ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
save(s.ROOT/'tmp/Multiome_full10_EB100_native1_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(s.ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(paths)))
print(json.dumps(dict(complete_Multiome_rows=1,models=10,mean=mean,std=std,CI95=list(ci),EB100_full_native_models=1,files=len(paths))))
