import json,shutil,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as s
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/TAB878_metric_suite_and_first_actual_entity_v1'
assert folder.is_dir()

def copy(source,name):
    target=folder/name
    if target.exists():assert s.sha(source)==s.sha(target);return
    shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);s.verify(c)
entry=next(e for e in c['entries'] if e['key']=='5eedb87987df7781d41c')
path,proof=s.checked_fragment(c,entry,'e000');attempt,output=s.paths(c,entry,'e000')
g=s.read(attempt/'guard_receipt.json')
assert proof['entity']=='SMAP_A-1' and proof['test_points']==8640 and proof['metric_calls']==804
assert proof['metric_outcome_counts']==dict(finite=772,nonfinite=32) and not proof['all_metric_values_finite']
assert not proof['neural_training_or_inference_performed'] and g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted']
copy(path,'actual_A1_all804_TAB_metric_cells.json');copy(attempt/'guard_receipt.json','actual_A1_original_resource_guard.json')
copy(attempt/'launch.json','actual_A1_metric_worker_handle.json')
validation=Path('F:/aicoding/IIA_Data/experiments/TAB_full878_original_metric_suite_tests_20261016_v2.xml')
xs=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests','0')) for x in xs)==22 and all(int(x.get(k,'0'))==0 for x in xs for k in ['failures','errors','skipped'])
copy(validation,'validation22.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.CONFIG,s.TEST]
for i,p in enumerate(paths):copy(ROOT/p,f'original{i:02d}'+Path(p).suffix)
mapping=[]
tab_files=[ROOT/s.TAB/'ts_benchmark/__init__.py',ROOT/s.TAB/'ts_benchmark/evaluation/__init__.py']
tab_files+=sorted((ROOT/s.TAB/'ts_benchmark/evaluation/metrics').rglob('*.py'))
for i,p in enumerate(tab_files):
    name=f'TAB{i:02d}_'+p.stem[:35]+p.suffix
    # Keep generated partial author-source snapshots locally on F; publish
    # immutable upstream identities/hashes instead of assuming a root license.
    previous=folder/name
    preserved=Path('F:/aicoding/IIA_Data/experiments/TAB878_partial_publication_source_snapshots_v1')
    if previous.exists():
        preserved.mkdir(parents=True,exist_ok=True);target=preserved/name
        assert previous.resolve().is_relative_to(folder.resolve()) and target.resolve().is_relative_to(preserved.resolve())
        assert not target.exists() and s.sha(previous)==s.sha(p)
        shutil.move(str(previous),str(target));assert s.sha(target)==s.sha(p)
    mapping.append(dict(original=p.relative_to(ROOT).as_posix(),sha256=s.sha(p),full_author_source_preserved_locally=True,
        upstream='https://github.com/decisionintelligence/TAB/blob/'+s.COMMIT+'/'+p.relative_to(ROOT/s.TAB).as_posix()))
licenses=[p for p in (ROOT/s.TAB).iterdir() if p.is_file() and p.name.lower() in ['license','license.md','license.txt']]
s.write(folder/'original_TAB_source_mapping.json',dict(commit=s.COMMIT,files=mapping,root_license_files_present=[p.name for p in licenses],
    author_source_full_files_republished=False))
initial=Path('F:/aicoding/IIA_Data/experiments/TAB_full878_original_metric_suite_tests_20261016_v1.xml')
ix=list(ET.parse(initial).getroot().iter('testsuite'));assert sum(int(x.get('failures','0')) for x in ix)==19
wheels=[dict(filename=p.name,path=str(p),sha256=s.sha(p),bytes=p.stat().st_size) for p in sorted(Path(s.WHEELS).glob('*.whl'))]
assert len(wheels)==2
copy(Path(s.WHEELS)/'isolated_install_report.json','isolated_install_report.json')
s.write(folder/'isolated_dependencies_and_preserved_initial_validation.json',dict(
    dependency_directory=s.VENDOR,environment=c['environment'],wheels=wheels,
    initial_failure=dict(path=str(initial),sha256=s.sha(initial),failed_tests=19,
        cause='statsmodels absent from shared Python312;found during initial prelaunch tests'),
    fix='Isolated statsmodels0.14.6+patsy1.0.1 vendor with no shared dependency install or upgrade',
    original_shared_environment_modified=False,scientific_task_retry_count=0))
launch=s.read(ROOT/c['state_root']/'launch.json');state=s.read(ROOT/c['state_root']/'status.json')
owner=psutil.Process(launch['pid']);assert owner.create_time()==launch['create_time']==state['create_time'] and owner.cmdline()==launch['command']
assert not state['failed_cases_preserved']
active=None
if state['active_pid']:
    p=psutil.Process(state['active_pid']);t=p.cpu_times()
    active=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=t.user+t.system)
s.write(folder/'actual_continuing_evaluation_not_completed_case.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),actual_active_worker=active,
    status_snapshot=state,registered_cases=878,registered_label_views=2057,registered_entity_label_views=87798,
    original_policy=dict(settings=c['settings'],resource_lock=c['resource_lock']),
    complete_real_dataset_TAB_case_results=0,all_original_experiments_complete=False))
copy(__file__,'publication_source.py')
s.write(folder/'archive_validation.json',dict(tests_passed=22,actual_original_metric_calls_closed_first_entity=804,
    actual_first_entity_finite_cells=772,actual_first_entity_undefined_cells=32,
    first_entity_source='Actual full54entitySMAP TranADFloat64seed1103;only A1 TAB metric fragment complete',
    original_research_scope=c['original_scope'],all_original_experiments_complete=False,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/TAB878_original21_metrics_first_real_entity.2026-10-16_v1.json'
s.write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_cases=878,label_views=2057,entity_label_views=87798,
    tests_passed=22,actual_first_entity_metric_calls_closed=804,finite_cells=772,undefined_cells=32,
    complete_real_dataset_TAB_case_results=0,original_scope=c['original_scope'],all_original_experiments_complete=False))
paths+=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda root,*a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
s.write(ROOT/'tmp/TAB878_first_actual_entity_publication_manifest_20261016_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=22,actual_first_entity_metric_calls_closed=804,live_owner=owner.pid)))
