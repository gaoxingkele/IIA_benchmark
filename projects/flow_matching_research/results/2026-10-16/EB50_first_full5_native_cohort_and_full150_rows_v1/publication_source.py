"""Close all five actual EB50-PC models and register every full150 row comparison."""
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scipy.stats import t as student_t
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching import summarize_otcfm_EB_CITE_full150_paper_rows_v1 as s
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats


def save(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def copy(path,target):
    with target.open('xb') as f:f.write(path.read_bytes())
    assert sha(path)==sha(target)


q=read(ROOT/s.native.QUEUE);s.native.verify(q)
rc=read(ROOT/s.replay.CONFIG);s.replay.verify(rc)
c=read(ROOT/s.CONFIG);s.verify(c)
jobs=[j for j in q['jobs'] if (j['dataset'],j['protocol'],j['method'],j['leaveout_timepoint'])
      ==('released_author_EB','released_yaml','cfm',1)]
assert len(jobs)==5 and sorted(j['seed'] for j in jobs)==list(range(42,47))
records=[];archives=[];cost_entries=0
for job in sorted(jobs,key=lambda j:j['seed']):
    source,provenance=s.inspect(q,job)
    sg,ng,rp=s.guard_paths(q,rc,job);r,g=read(rp),read(ng)
    s.validate_native(job,source,provenance,r,g,sha(rp),sha(ROOT/s.replay.CONFIG))
    assert r['native_metrics']==r['original_metrics']==source['test_metrics'] and len(r['native_metrics'])==44
    assert source['actual_completed_epochs']==1000 and source['optimizer_updates']==4000
    assert source['parameter_count']==9093 and r['extra_model_training_updates']==0
    solves=r['full_test_population_distances_independently_recomputed'][0]['solves']
    w1=[v for v in solves if v['power']==1]
    assert [v['full_cost_shape'] for v in w1]==[[2656,4374],[4374,3650],[3650,3873]]
    cost_entries+=sum(v['full_cost_shape'][0]*v['full_cost_shape'][1] for v in w1)
    for artifact in r['all_full_test_arrays_and_artifacts']:
        p=Path(artifact['path']);assert p.stat().st_size==artifact['bytes'] and sha(p)==artifact['sha256']
    records.append(dict(id=job['id'],seed=job['seed'],epochs=1000,actual_training_updates=4000,
        selected_epoch=r['selected_checkpoint_epoch'],selected_updates=r['selected_checkpoint_updates'],
        heldout_W1=source['primary_heldout_W1'],all44_metrics=source['test_metrics'],
        all44_native_scalars_bitwise_equal=True,source_provenance=provenance,
        native_report_sha256=sha(rp),native_guard_sha256=sha(ng),full_independent_W1_solves=w1))
    archives.append((job['seed'],dict(source_result=Path(provenance['result_path']),source_guard=sg,
        complete_native_report=rp,complete_native_guard=ng)))
assert cost_entries==208594470
summary={}
for name in sorted(records[0]['all44_metrics']):
    values=[r['all44_metrics'][name] for r in records];stats=independent_stats(values)
    mean,std=math.fsum(values)/5,statistics.stdev(values)
    ci=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5)) if std else (mean,mean)
    check=dict(mean=mean,std=std,ci95_low=float(ci[0]),ci95_high=float(ci[1]))
    assert all(math.isclose(stats[k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in check.items())
    summary[name]=dict(**stats,degrees_of_freedom=4,per_original_seed_values=values,
        independent_scalar_and_SciPy_t95_verified=True)
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/EB50_first_full5_native_cohort_and_full150_rows_v1'
folder.mkdir(parents=True,exist_ok=False)
for seed,files in archives:
    base=folder/f'seed{seed}';base.mkdir()
    for name,path in files.items():copy(path,base/(name+'.json'))
save(folder/'complete_five_models_all44_scalar_summary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    dataset='released_author_EB',cells=18203,original_PC_dimensions=50,used_PC_dimensions=5,
    protocol='released_yaml',method='cfm',leaveout_timepoint=1,seeds=list(range(42,47)),
    five_original_training_models_complete=True,five_full_native_checkpoint_replays_complete=True,
    all220_native_scalar_metrics_bitwise_equal=True,total_independent_full_cost_entries=cost_entries,
    native_W1_W2_OPTIMAL_solves=30,independent_W1_OPTIMAL_solves=15,
    full_source_and_checkpoint_and_artifact_hashes_rechecked=True,per_model=records,all44_metric_seed_summaries=summary,
    uncertainty_scope='five training seeds on one frozen input/split/recipe/leaveout;not across datasets or leaveouts',
    full_three_leaveout_paper_row_complete=False,historical_paper_identity_certified=False,strict_TAB_result=False,
    original_scope=q['original_scope'],all_original_experiments_complete=False))
with (folder/'all44_full_five_seed_metrics.csv').open('x',encoding='utf-8',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['metric','primary','mean','sample_STD','model_mean_t95_low','model_mean_t95_high',*[f'seed{i}' for i in range(42,47)]])
    for name,r in summary.items():writer.writerow([name,name==q['primary_metric'],r['mean'],r['std'],r['ci95_low'],r['ci95_high'],*r['per_original_seed_values']])
states={}
for name in [q['state_root'],rc['state_root'],c['state_root']]:
    base=ROOT/name;state=read(base/'status.json');p=psutil.Process(state['pid'])
    assert p.is_running() and p.create_time()==state['create_time']
    states[name]=dict(state=state,actual_command=p.cmdline(),actual_cpu_seconds=sum(p.cpu_times()[:2]))
save(folder/'actual_live_source_native_and_all150_row_handles.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),records=states))
copy(ROOT/c['state_root']/'launch.json',folder/'complete150_row_collector_launch.json')
complete_groups=[]
for group in c['groups']:
    ready=[identity for identity in group['required_case_ids']
        if s.availability(q,rc,next(j for j in q['jobs'] if j['id']==identity)) is True]
    complete_groups.append(dict(id=group['id'],dataset=group['dataset'],required_case_ids=group['required_case_ids'],
        ready_complete_native_ids=ready,ready_complete_native_count=len(ready),
        full_row_ready=len(ready)==len(group['required_case_ids'])))
assert len(complete_groups)==12 and sum(len(g['required_case_ids']) for g in complete_groups)==150
assert not any(g['full_row_ready'] for g in complete_groups)
save(folder/'all_twelve_complete_paper_rows_required.json',dict(groups=complete_groups,required_rows=12,
    required_distinct_models=150,complete_paper_rows=0,
    inherited_wait_state_prose='waiting_all15_source_and_native_conditions_per_row is inherited generic text;actual frozen group IDs require15 EB or10 CITE models',
    no_partial_row_promoted=True,historical_input_identity_certified=False,all_original_experiments_complete=False))
new_files=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG]
snapshots=[]
for i,path in enumerate(new_files+[s.native.QUEUE,s.replay.CONFIG]):
    p=ROOT/path;name=f'source{i:02d}{p.suffix}';copy(p,folder/name)
    snapshots.append(dict(path=path,sha256=sha(p),snapshot=name))
save(folder/'source_manifest.json',snapshots)
xml=Path('F:/aicoding/IIA_Data/experiments/EB_CITE_full150_paper_row_contract_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(e.get('tests','0')) for e in suites)==35
assert not any(int(e.get(k,'0')) for e in suites for k in ('errors','failures','skipped'))
copy(xml,folder/'validation_35_checks.xml');copy(Path(__file__),folder/'publication_source.py')
(folder/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.csv -text\n',encoding='utf-8')
primary=summary[q['primary_metric']]
(folder/'README.md').write_text('Released EB50-PC: first full five-seed native cohort and all150 complete-row comparisons\n\n'
    'All five original released-YAML I-CFM leaveout1 models,seeds42..46,completed1000epochs/4000updates/9093parameters each. '
    'Complete original checkpoint restoration,test/auxiliary inference and all44scalar metrics matched exactly in each model. '
    'All30native W1/W2 and15independent full-matrix W1 LP solves were OPTIMAL. '
    'Every one of208594470independent cost entries and its objective perturbation bound was checked by the existing native replay. '
    'This publication rehashes all original source/guard/checkpoint/replay artifacts without extra inference or optimizer updates.\n\n'
    'Heldout W1 mean='+str(primary['mean'])+',sampleSTD='+str(primary['std'])+',five-model Student-t95=['
    +str(primary['ci95_low'])+','+str(primary['ci95_high'])+']. '
    'The interval describes training-seed variation for one frozen dataset/split/leaveout;it is not uncertainty across datasets,timepoints or historical preprocessing. '
    'Full44metric CSV and all five source/native proofs are included.\n\n'
    'Table4 I-CFM EB target0.872+/-0.087 averages heldout-time experiments. These five models cover leaveout1 only. '
    'The full row still requires ten independently trained leaveout2/3 models and their full native proofs. '
    'This released artifact has18203cells/50PCs,whereas the separate original-code-reference asset has16819cells/100PCs. '
    'Input populations,optimizer steps,validation batches and selected weights differ;this is not a causal data-only ablation or historical replication certificate.\n\n'
    'The live metadata collector covers all150original EB/CITE source conditions in12method/recipe/input rows. '
    'Every EB row requires15models,every CITE row10models,with44or36complete metrics and all full LP proofs. '
    'Both recipes stay separate;five seed-level leaveout means and all-model spreads use distinct uncertainty units. '
    'Paper SD aggregation remains uncertified. Released CITE is donor32606,not paper-described13176. '
    'The inherited waiting-state string mentions15generically;the actual required IDs correctly enforce15EB/10CITE. '
    '35relevant checks passed,including mutation rejection on a real full native proof;synthetic fixtures are not benchmark performance. '
    'All other90EB100,120Multiome,deferred60GEO and45paper/681/273/702work remains required.\n',encoding='utf-8')

taskpath=ROOT/'projects/flow_matching_research/tasks.json';before=taskpath.read_bytes();tasks=json.loads(before)
for task in tasks['tasks']:
    if task['id']=='FM-001':task['material_addenda']=list(dict.fromkeys(task.get('material_addenda',[])+['materials/TrajectoryNet_EB100_official_addendum.v1.json']))
    if task['id']=='FM-006':
        task['status']='in_progress'
        task['evidence']=list(dict.fromkeys(task.get('evidence',[])+[
            'results/2026-10-16/EB100_four_sources_full_rows_and_GRASP48_v1',
            'results/2026-10-16/grasp_gnn_seed1107_full48_model_increment_v1',
            folder.relative_to(ROOT/'projects/flow_matching_research').as_posix()]))
assert taskpath.read_bytes()==before
taskpath.write_text(json.dumps(tasks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=dict(previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),
    paths=sorted(new_files+[taskpath.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]))
save(ROOT/'tmp/EB50_full5_and_full150_rows_publication_manifest_20261016_v1.json',manifest)
print(json.dumps(dict(complete_models=5,full_native_scalar_metrics=220,independent_full_cost_entries=cost_entries,
    heldout_W1=primary,paper_rows_complete=0,registered_rows=12,files=len(manifest['paths']))))
