"""Publish actual full215 recovery and preserve both proven metadata failures."""
from datetime import datetime,timezone
import csv
import hashlib
import json
import marshal
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v4 as s

v3,v2=s.previous,s.previous.previous
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def copy(p,t):
    with t.open('xb') as f:f.write(p.read_bytes())
    assert s.sha(p)==s.sha(t)

c=s.read(s.ROOT/s.CONFIG);source,cases,contract,tasks=s.verify(c);base=s.ROOT/c['state_root']
launch=s.read(base/'launch.json');state=s.read(base/'status.json');actor=psutil.Process(launch['pid'])
assert actor.is_running() and actor.create_time()==launch['create_time']
assert state['pid']==actor.pid and state['create_time']==actor.create_time() and not state['failed_preserved']
assert len(tasks)==215 and len(cases)==500 and v2.no_original_attempts(c)
third=s.ROOT/s.read(s.ROOT/v3.CONFIG)['state_root'];second=s.ROOT/s.read(s.ROOT/v2.CONFIG)['state_root']
h,hi=s.read(third/'handoff.json'),s.read(third/'handoff_identity.json')
assert h['waiting_controller_only_stopped'] and not h['original_native_child_stopped']
assert hi['no_numerical_child_present'] and hi['exact_live_os_pid_time_and_command_checked_twice']
for owner in [c['predecessor_owner'],s.read(third/'launch.json'),s.read(second/'launch.json')]:
    try:
        p=psutil.Process(owner['pid']);assert p.create_time()!=owner['create_time']
    except psutil.NoSuchProcess:pass
assert not (second/'attempts').exists() and not (third/'attempts').exists()
folder=s.ROOT/'projects/flow_matching_research/results/2026-10-16/AT_full215_hot_status_recovery_v4'
folder.mkdir(parents=True,exist_ok=False)
sources=dict(preserved_initial9_preworker_failures=s.ROOT/v2.SNAPSHOT,
    preserved_v2_failed_launch=second/'launch.json',preserved_v3_failed_launch=third/'launch.json',
    actual_previous_idle_status=third/'previous_status_at_verified_handoff.json',
    actual_twice_checked_idle_identity=third/'handoff_identity.json',actual_metadata_only_handoff=third/'handoff.json',
    actual_live_v4_launch=base/'launch.json',actual_live_v4_state=base/'status.json')
for name,p in sources.items():copy(p,folder/(name+'.json'))
copy(second/'controller.stderr.log',folder/'preserved_v2_pre_handoff_TypeError.txt')
copy(third/'controller.stderr.log',folder/'preserved_v3_duplicate_facade_RuntimeError.txt')
save(folder/'actual_complete215_recovery.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_controller=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline()),
    required_models=500,required_tasks=215,required_kinds=dict(cohort=100,paired=95,macro=20),
    preserved_initial_failed_metadata_tasks=c['recovered_initial_preworker_task_ids'],
    new_failed_tasks=state['failed_preserved'],actual_completed_statistics_tasks=state['completed_tasks'],
    all_original_tasks_retained=c['tasks'],original_resource_settings=c['settings'],resource_lock=c['resource_lock'],
    original_numerical_function_code_hashes={name:hashlib.sha256(marshal.dumps(getattr(v2.original,name).__code__)).hexdigest()
        for name in ['worker','five_seed_summary','aggregate_metric_trees','paired_common_support','macro_summary']},
    only_idle_metadata_parent_stopped=True,numerical_child_stopped=False,numerical_attempts_repeated=False,
    completed_metadata_handoff_repeated=False,facade_API_initialized_once=True,
    actual_model_proofs_and_complete_five_seed_cohorts_still_required=True,
    no_benchmark_performance_or_paper_equivalence_claim=True,original_scope=c['original_scope'],
    all_original_experiments_complete=False))
with (folder/'all215_pending_statistics_inventory.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['id','kind','variant','dataset','required_model_conditions','actual_statistics_complete'])
    for t in tasks.values():w.writerow([t['id'],t['kind'],t['variant'],t.get('dataset','all_five'),len(t['cases']),t['id'] in state['completed_tasks']])
up=s.ROOT/source['state_root']/'status.json';us=s.read(up);p=psutil.Process(us['pid'])
assert p.is_running() and p.create_time()==us['create_time']
save(folder/'actual_upstream500_checkpoint_review_observation.json',dict(state=us,actual_handle=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())))
wr=s.ROOT/'experiments/runs/fm_CITE13176_GEO_headroom_ready_data_window_v2/window_receipt.json'
assert s.read(wr)['all_metadata_locks_released'];copy(wr,folder/'CITE_finite_window_v2_actual_terminal_receipt.json')
new=[v2.SNAPSHOT,*[p for m in [v2,v3,s] for p in [m.MODULE.replace('.','/')+'.py',m.TEST,m.CONFIG]]]
pins=[]
for i,name in enumerate(new):
    p=s.ROOT/name;t=folder/f'source{i:02d}{p.suffix}';copy(p,t);pins.append(dict(path=name,sha256=s.sha(p),snapshot=t.name))
save(folder/'source_manifest.json',pins)
for version,count in [(2,31),(3,35),(4,38)]:
    p=Path('F:/aicoding/IIA_Data/experiments')/f'AnomalyTransformer_full215_statistics_metadata_recovery_20261016_v{version}_tests.xml'
    suites=list(ET.parse(p).getroot().iter('testsuite'));assert sum(int(z.get('tests','0')) for z in suites)==count
    assert not any(int(z.get(k,'0')) for z in suites for k in ['errors','failures','skipped'])
    copy(p,folder/f'validation_v{version}_{count}_checks.xml')
copy(Path(__file__),folder/'publication_source.py')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.xml -text\n*.csv -text\n*.txt -text -whitespace\n',encoding='utf-8')
(folder/'README.md').write_text('Full500-model/215-task Anomaly Transformer statistics metadata recovery\n\n'
    'Four cohorts and five dependencies were permanently marked failed after transient PermissionError reading the hot producer status file. '
    'No statistics numerical attempt or artifact had been created. Initial9failures and final predecessor status are preserved. '
    'V2failed before handoff because FunctionType discarded a default poll argument. V3completed the verified idle handoff but failed before statistics startup on duplicate cold-facade initialization. '
    'Both configurations,sources,launches,traces and the actual metadata-only handoff are preserved. '
    'V4uses one real cold API initialization and the original controller code with that same API;the completed handoff is not repeated.\n\n'
    'The live successor retains all500original model bindings,100five-seed dataset cohorts,95paired common-support comparisons and20equal-dataset macro summaries. '
    'All215tasks still require full original source/native/independent proof receipts. None is credited as a completed performance result by metadata recovery. '
    'The numerical worker and mean/uncertainty/paired/macro functions are frozen original functions. '
    'Only temporary sharing,partial JSON or file replacement on the one live status file becomes not-ready;immutable data/checkpoint/guard failures remain fatal. '
    'Original seeds,validation-only noPA F1,diagnostic controls,full-validation calibration,covered support,unclipped df4intervals and32GiBphysical/40GiBcommit/16GiBemergency guards remain.\n\n'
    '38v4checks pass,including a real cold-resource API and actual original controller entry in an explicitly temporary empty-selection fixture. '
    'The fixture is plumbing only;it never becomes a benchmark or a reduced actual215-task selection. '
    'Earlier31/35checks missed the two startup defects and are retained with the failing traces. '
    'The previous CITE terminal receipt confirms its bounded metadata window released both locks without preprocessing. '
    'All45papers/681method-baseline-ablation records/273data tasks/702scalar targets and other unfinished experiments remain required.\n',encoding='utf-8')
paths=new+[p.relative_to(s.ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
save(s.ROOT/'tmp/AT_full215_hot_status_recovery_publication_manifest_20261016_v4.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(s.ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(paths)))
print(json.dumps(dict(actual_successor_pid=actor.pid,required_models=500,retained_tasks=215,preserved_initial_preworker_failures=9,
    actual_statistics_completed=len(state['completed_tasks']),files=len(paths))))
