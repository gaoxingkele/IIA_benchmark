import json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flow_matching import recover_TAB_resource_cells_v2 as s
from scripts.flow_matching import run_cfm_ts_original_job as cfm
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/TAB_resource_cells_recovery_v2_registered'
assert {p.name for p in folder.iterdir()}=={'README.md'}

def copy(source,name):
    target=folder/name;assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);s.verify(c)
final=Path('F:/aicoding/IIA_Data/experiments/TAB_resource_cells_v2_tests_20261016_v3.xml')
xs=list(ET.parse(final).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in xs)==21
assert all(int(x.get(k,'0'))==0 for x in xs for k in ['failures','errors','skipped']);copy(final,'validation21.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.CONFIG,s.TEST]
for i,p in enumerate(paths):copy(ROOT/p,f'original{i:02d}'+Path(p).suffix)
failure=Path('F:/aicoding/IIA_Data/experiments/TAB_resource_cells_v2_tests_20261016_v2.xml')
fs=list(ET.parse(failure).getroot().iter('testsuite'));assert sum(int(x.get('failures','0')) for x in fs)==1
s.write(folder/'preserved_full_worker_test_failure_and_fix.json',dict(path=str(failure),sha256=s.sha(failure),
    failures=1,cause='Windows MAX_PATH on cell_key.json.pending_uuid in deeper complete-worker fixture',
    fix='Same-directory short pending_uuid temporary name;full SHA256 final cell key retained',
    original_numeric_metric_functions_or_data_changed=False,real_scientific_task_retried=False))
fixture=Path('F:/aicoding/IIA_Data/experiments/TAB_resource_cells_v2_unit_fixtures_20261016_v3/test_complete_worker_preserves0/artifacts/cases/5eedb87987df7781d41c/e000')
unit=s.read(fixture/s.REPORT);assert unit['passed'] and unit['entity']=='fixture_only' and unit['metric_calls']==804
assert unit['durable_cell_journal']['fresh_cells']==804 and unit['durable_cell_journal']['checkpoint_cells']==0
s.write(folder/'synthetic_full804_cell_worker_probe_not_benchmark.json',dict(source=str(fixture/s.REPORT),sha256=s.sha(fixture/s.REPORT),
    test_points=unit['test_points'],fresh_cells=804,source_case_field_is_fixture_metadata_only=True,
    actual_resource_guard_emitted=False,real_benchmark_performance_claimed=False))
e=next(e for e in c['entries'] if e['key']=='5eedb87987df7781d41c');evidence=s.resource_abort(c,e);assert evidence is not None
s.write(folder/'actual_original_resource_abort_and_reused6_fragments.json',dict(resource_abort=evidence,
    reused=[dict(path=str(p),sha256=s.sha(p),entity=proof['entity'],metric_calls=proof['metric_calls'])
        for p,proof,kind in [s.checked_fragment(c,e,f'e{i:03d}') for i in range(6)]],
    new_recovered_entities_complete=0,full_case_recovered=False))
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert not state['failed_recoveries_preserved'] and not state['completed_recovery_case_keys'] and state['active_pid'] is None
s.write(folder/'actual_live_recovery_waiting_strengthened_gate.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_handle=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),state=state,
    original_settings=c['original_resource_settings'],strengthened_settings=c['settings'],resource_lock=c['resource_lock'],
    original_metric_functions_or_grids_changed=False,new_recovered_entities_complete=0,all_original_experiments_complete=False))
q=cfm.read(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json')
j=next(j for j in q['jobs'] if j['id']=='cfm_ts_original__appendix_d__node_e10__sine__seed1105')
progress=ROOT/j['output_directory']/'progress.json';receipt=ROOT/q['state_root']/'receipts'/(j['id']+'.json')
r=cfm.read(receipt);v=cfm.read(progress)
assert not r['complete'] and r['exit_code']!=0 and r['resource_observations']['resource_guard_aborted']
assert v['epoch']==1 and v['required_epochs']==10 and v['optimizer_updates']==2 and j['expected_optimizer_updates']==20
assert not (Path(j['artifact_directory'])/'model.pt').exists() and not cfm.complete(j)
copy(receipt,'actual_CFM_NODE1105_resource_abort.json');copy(progress,'actual_CFM_NODE1105_incomplete_progress.json')
st=cfm.read(ROOT/q['state_root']/'status.json');owner=psutil.Process(st['pid'])
assert owner.create_time()==st['create_time'] and 'scripts.flow_matching.run_cached_resource_monitor_controller_v2' in owner.cmdline()
s.write(folder/'actual_CFM_original_owner_not_restarted.json',dict(job=j,queue_sha256=cfm.sha(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json'),
    original_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),state_snapshot=st,
    final_model_checkpoint_exists=False,training_complete=False,original_failed_training_restarted=False,
    incomplete_progress_is_usable_checkpoint=False,all_original_experiments_complete=False))
copy(__file__,'publication_source.py')
s.write(folder/'archive_validation.json',dict(tests_passed=21,original_metric_case_scope=878,label_views=2057,
    old_guarded_fragments_reused=6,new_recovered_entities_complete=0,CFM_NODE1105_resource_failure_preserved=True,
    original_scope=c['original_scope'],all_original_experiments_complete=False,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/TAB_durable_resource_recovery_and_CFM_abort.2026-10-16_v2.json'
s.write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),tests_passed=21,metric_case_scope=878,old_fragments_reused=6,
    new_recovered_entities_complete=0,CFM_NODE1105_incomplete_training_preserved=True,original_scope=c['original_scope'],
    all_original_experiments_complete=False))
paths+=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda root,*a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
s.write(ROOT/'tmp/TAB_resource_cells_recovery_v2_publication_manifest_20261016.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=21,actual_recovery_owner=p.pid,old_fragments_reused=6,new_recovered_entities_complete=0)))
