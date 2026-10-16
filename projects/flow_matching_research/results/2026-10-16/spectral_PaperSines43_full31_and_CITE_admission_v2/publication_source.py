"""Close the actual31st full source/artifact model and finite admission change."""
import ast
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha
from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as collector
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort
from scripts.flow_matching import reserve_CITE_headroom_ready_window_v2 as window


def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def copy(path,target):
    with target.open('xb') as f:f.write(path.read_bytes())
    assert sha(path)==sha(target)


def tests(path,count):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)==count
    assert not any(int(s.get(k,'0')) for s in suites for k in ('errors','failures','skipped'))


c=read(ROOT/collector.CONFIG);cases,old=collector.verify(c)
priorpath=ROOT/collector.CATALOG
statepath=ROOT/c['state_root']/'status.json';state=read(statepath)
identity='spectral_original__diffusion_ts__sines__paper_sines_distribution__seed43'
assert len(old['records'])==old['full_native_generators']==30
assert set(state['verified'])=={r['canonical_id'] for r in old['records']}|{identity}
reference=state['verified'][identity];proof_path=Path(reference['proof']);proof=read(proof_path)
assert sha(proof_path)==reference['proof_sha256'] and proof['passed'] and proof['canonical_id']==identity
assert proof['original_optimizer_updates']==12000 and proof['original_training_minibatches']==24000
assert proof['actual_model_parameters']==235249
assert proof['complete_reference_shape']==[10000,24,5] and proof['complete_generated_shape']==[10005,24,5]
assert proof['actual_full_original_checkpoint_tensors_restored_bitwise'] and proof['complete_original_reference_array_bitwise_verified']
assert proof['all_original_generated_arrays_finite'] and proof['all_four_five_repeat_evaluator_statistics_independently_recomputed']
assert not proof['cuda_context_initialized'] and not proof['generative_model_reinferred_by_this_audit']
assert not proof['learned_evaluators_independently_retrained']
guardpath=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json';g=read(guardpath)
assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(proof_path)
case=cases[identity];job=case['job'];sourcequeue=read(ROOT/case['queue'])
resultpath=ROOT/job['output_directory']/'result.json'
terminalpath=ROOT/sourcequeue['state_root']/'receipts'/(identity+'.json')
result,terminal=read(resultpath),read(terminalpath)
assert sha(resultpath)==proof['original_result_sha256'] and sha(terminalpath)==proof['actual_terminal_receipt_sha256']
assert terminal['exitcode']==0 and terminal['full_result_validated'] and not terminal['resource_guard_aborted']
assert result['job']==job and result['optimizer_updates']==12000 and result['training_minibatches']==24000
base=ROOT/'projects/flow_matching_research/results/2026-10-16/spectral_PaperSines43_full31_and_CITE_admission_v2'
base.mkdir(parents=True,exist_ok=False)
sources=dict(new_original_result=resultpath,new_source_terminal_guard=terminalpath,
    new_independent_artifact_proof=proof_path,new_review_guard=guardpath,
    new_review_validation=proof_path.parent/'validation.json',
    new_review_configuration=ROOT/c['configuration_root']/(identity+'.json'))
for name,path in sources.items():copy(path,base/(name+'.json'))
new=dict(canonical_id=identity,model=job['model'],dataset=job['dataset'],seed=job['seed'],
    optimizer_updates=12000,training_minibatches=24000,complete_reference_shape=proof['complete_reference_shape'],
    complete_generated_shape=proof['complete_generated_shape'],model_parameters=235249,metrics=proof['metrics'],
    original_result_sha256=sha(resultpath),artifacts=proof['artifacts'],
    full_native_proof=(base/'new_independent_artifact_proof.json').relative_to(ROOT).as_posix(),
    full_native_proof_sha256=sha(proof_path),original_runtime_proof_path=str(proof_path),
    independently_retrained_learned_evaluators=False,generative_output_reinferred=False)
records=deepcopy(old['records'])+[new]
assert len(records)==len({r['canonical_id'] for r in records})==31 and records[:30]==old['records']
groups=defaultdict(list)
for row in records:
    p=ROOT/row['full_native_proof'];v=read(p)
    assert sha(p)==row['full_native_proof_sha256'] and v['passed'] and v['canonical_id']==row['canonical_id']
    assert v['seed']==row['seed'] and v['metrics']==row['metrics']
    for a in row['artifacts']:assert sha(ROOT/a['path'])==a['sha256']
    source=cases[row['canonical_id']]['job']
    assert sha(ROOT/source['output_directory']/'result.json')==row['original_result_sha256']
    groups[(row['model'],row['dataset'],source['data_track'])].append(dict(v,model=row['model'],dataset=row['dataset'],data_track=source['data_track']))
summaries=[summarize_cohort(v) for _,v in sorted(groups.items())]
paper=next(r for r in summaries if r['dataset']=='sines' and r['data_track']=='paper_sines_distribution')
assert paper['independently_reviewed_training_seeds']==[42,43]
assert paper['missing_original_training_seeds']==[44,45,46] and not paper['full_five_seed_cohort_complete']
assert all(r['full_cohort_mean'] is None and r['between_training_model_CI95'] is None for r in paper['metrics'].values())
save(base/'unique31_full_source_and_artifact_model_catalog.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_source_and_artifact_reviewed_models=31,full_native_generators=31,
    inherited_full_native_generators_label='Legacy field counts full original source execution plus tensor/array/statistic audit;it does not certify generative reinference.',
    original_registered_generation_jobs=70,newly_independently_artifact_reviewed_models=1,reused_previous_full_native_reviews=30,
    previous30_records_unchanged=True,previous_closed_catalog=collector.CATALOG,previous_closed_catalog_sha256=sha(priorpath),
    original_total_optimizer_updates=sum(r['optimizer_updates'] for r in records),
    original_total_training_minibatches=sum(r['training_minibatches'] for r in records),
    original_total_full_evaluator_runs=31*4*5,records=records,full_generative_reinference_certified=False,
    learned_evaluator_retraining_certified=False,author_equivalence_certified=False,strict_TAB_result=False,
    all_original_experiments_complete=False))
save(base/'all_reviewed_cohort_completion_and_metrics.json',dict(cohorts=summaries,
    required_training_seeds=[42,43,44,45,46],all_original_experiments_complete=False))
with (base/'all31_model_evaluator_metrics.csv').open('x',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['canonical_id','model','dataset','data_track','training_seed','metric','five_evaluator_mean','within_model_sample_STD','within_model_t95_low','within_model_t95_high'])
    for row in records:
        for metric,v in row['metrics'].items():writer.writerow([row['canonical_id'],row['model'],row['dataset'],cases[row['canonical_id']]['job']['data_track'],row['seed'],metric,v['mean'],v['std'],v['ci95_low'],v['ci95_high']])

# Record the precise replay limit from the pinned original save/load/sample code.
solver='experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_base/engine/solver.py'
source_entry='scripts/flow_matching/run_spectral_original_job.py'
tree=ast.parse((ROOT/solver).read_text(encoding='utf-8'));trainer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
save_function=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='save')
saved=next(n.value for n in save_function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='data' for t in n.targets))
fields=[k.value for k in saved.keys];assert fields==['step','model','ema','opt']
entry=ast.parse((ROOT/source_entry).read_text(encoding='utf-8'))
seed_lines=sorted(n.lineno for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='seed_everything')
train_line=next(n.lineno for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='train' and isinstance(n.func.value,ast.Name) and n.func.value.id=='trainer')
sample_line=next(n.lineno for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sample' and isinstance(n.func.value,ast.Name) and n.func.value.id=='trainer')
assert len(seed_lines)==2 and seed_lines[0]<train_line<sample_line<seed_lines[1]
save(base/'generative_replay_random_state_evidence_and_remaining_work.json',dict(
    checkpoint_save_fields=fields,source_result_artifact_roles=[a['role'] for a in result['artifacts']],
    checkpoint_contains_generation_start_CPU_CUDA_RNG_state=False,
    seeded_once_before_training_and_again_only_after_generation_for_evaluators=True,
    source_evidence=[dict(path=p,sha256=sha(ROOT/p)) for p in [solver,source_entry]],
    source_seed_call_lines=seed_lines,source_training_call_line=train_line,source_sampling_call_line=sample_line,
    direct_bitwise_regeneration_from_checkpoint_alone_verified=False,
    stochastic_regeneration_and_learned_evaluator_reruns_complete=False,
    required_next_evidence='An independently documented full generation replay RNG policy or proven original RNG reconstruction,and complete original four/five-repeat learned evaluator reruns. Never label finite arrays/checkpoint restoration as generation reinference.',
    original_scope=c['original_scope'],all_original_experiments_complete=False))

wc=read(ROOT/window.CONFIG);window.verify(wc)
wb=ROOT/wc['state_root'];launch=read(wb/'launch.json');ws=read(wb/'status.json')
actor=psutil.Process(launch['pid']);assert actor.is_running() and actor.create_time()==launch['create_time']
prep=read(ROOT/wc['preparation_configuration']);pb=ROOT/prep['state_root'];ps=read(pb/'status.json')
producer=psutil.Process(wc['preparation_handle']['pid']);assert producer.create_time()==wc['preparation_handle']['create_time'] and producer.is_running()
for p,name in [(wb/'launch.json','new_finite_window_launch.json'),(wb/'status.json','new_finite_window_state.json'),
    (pb/'launch.json','same_original_preprocessing_launch.json'),(pb/'status.json','actual_original_preprocessing_state.json'),
    (ROOT/window.PREVIOUS_RECEIPT,'preserved_no_worker_predecessor_window_receipt.json')]:copy(p,base/name)
save(base/'actual_live_window_and_preparation_handles.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    window=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),state=ws),
    same_original_preprocessing=dict(pid=producer.pid,create_time=producer.create_time(),command=producer.cmdline(),state=ps),
    preprocessing_worker_restarted=False,numerical_workers_signalled=False,original_guards_changed=False))
new_files=[window.MODULE.replace('.','/')+'.py',window.TEST,window.CONFIG]
manifest=[]
for i,path in enumerate(new_files+[collector.CONFIG,collector.MODULE.replace('.','/')+'.py',solver,source_entry]):
    p=ROOT/path;target=base/f'source{i:02d}{p.suffix}';copy(p,target);manifest.append(dict(path=path,sha256=sha(p),snapshot=target.name))
save(base/'source_manifest.json',manifest)
for name,count,target in [('CITE_headroom_ready_finite_window_20261016_v2_tests.xml',16,'validation_16_checks.xml'),
    ('Spectral_PaperSines43_full31_closure_20261016_v1_tests.xml',41,'validation_41_checks.xml')]:
    p=Path('F:/aicoding/IIA_Data/experiments')/name;tests(p,count);copy(p,base/target)
copy(Path(__file__),base/'publication_source.py')
(base/'.gitattributes').write_text('*.py -text\n*.json -text\n*.xml -text\n*.csv -text\n',encoding='utf-8')
(base/'README.md').write_text('Spectral full source/artifact models31/70;headroom-ready CITE scheduling entry\n\n'
    'Added Diffusion-TS paper-distribution Sines seed43:12000optimizer updates/24000minibatches/235249parameters, '
    '10000x24x5complete reference and10005x24x5complete generated arrays,all four original five-repeat evaluators. '
    'The successful independent artifact worker restored every model tensor and checked all EMA/optimizer tensors,full arrays and evaluator statistics. '
    'All previous30catalog records are unchanged and all31source/result/checkpoint/generated/reference/proof hashes were rechecked. '
    'The CSV contains all31model metrics with within-model evaluator spread;five evaluator repeats are never five training seeds.\n\n'
    'Paper-distribution Sines currently has2/5training seeds42/43;44..46remain required. No partial-cohort mean or training-seed interval is promoted. '
    'Released-Sines and paper-distribution Sines remain separate. The native architecture235249parameters differs from paper232177. '
    'Generative reexecution and learned evaluator reruns remain uncertified. '
    'Pinned Trainer.save stores step/model/ema/opt without pre-generation RNG;the source seeds before training and only again after sampling. '
    'Checkpoint restoration and finite original arrays do not prove bitwise regeneration. Explicit source-line/hash evidence and remaining work are retained.\n\n'
    'The prior finite metadata window released without admitting a preprocessing worker. '
    'A new isolated entry first measures the unchanged32GiB headroom before attempting both existing family locks. '
    'The same original preprocessing controller and29394QCcells x58676RNAfull worker remain,with6x5second stable admission,all deadlines/guards and budgets unchanged. '
    'No existing numerical process was stopped,changed or repeated. The actual captured state/handles distinguish waiting or active preprocessing from completion. '
    'The automatic60source/native pipeline continues to wait for actual full preprocessing proof. '
    '16window and41Spectral checks passed. All45papers/681methods/273data tasks/702scalar targets and every unfinished original experiment remain in scope.\n',encoding='utf-8')
paths=new_files+[p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file()]
save(ROOT/'tmp/Spectral31_CITE_headroom_window_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(paths)))
print(json.dumps(dict(reviewed_models=31,required_models=70,paper_Sines_training_seeds=[42,43],
    window_state=ws['state'],preprocessing_state=ps['state'],files=len(paths))))
