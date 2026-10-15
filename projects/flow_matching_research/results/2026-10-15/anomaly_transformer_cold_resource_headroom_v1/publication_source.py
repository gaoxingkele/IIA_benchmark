"""Publish verified idle handoff and actual per-process commit reductions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import ROOT,read,sha,write,verify
relative='configs/runtime/fm_anomaly_transformer_cold_resource_entry.v2.json'
config=read(ROOT/relative)
for role in config['roles']:verify(config,role)
xml=ROOT/'tmp/AnomalyTransformer_cold_resources_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==7
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
receiptpath=ROOT/config['state_root']/'handoff_receipt.json';receipt=read(receiptpath)
assert len(receipt['records'])==4 and receipt['numerical_children_stopped']==0
assert receipt['configuration_sha256']==sha(ROOT/relative)
assert receipt['sum_released_controller_private_commit_bytes']>3*1024**3
current=[]
for r in receipt['records']:
    p=psutil.Process(r['successor_pid']);assert p.create_time()==r['successor_create_time']
    assert 'scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1' in p.cmdline()
    snapshot=r['successor_first_snapshot']
    assert not any(snapshot[k] for k in ['numpy_imported','pandas_imported','torch_imported'])
    assert snapshot['private_bytes']<128*1024**2
    original=read(ROOT/config['roles'][r['role']]['queue']);state=read(ROOT/original['state_root']/'status.json')
    assert state['pid']==p.pid
    current.append(dict(role=r['role'],pid=p.pid,create_time=p.create_time(),command=p.cmdline(),
                        current_private_bytes=p.memory_info().private,state_at_capture=state))
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_cold_resource_headroom_v1'
folder.mkdir(exist_ok=False)
sources={
 'frozen_active_runtime_v2.json':ROOT/relative,
 'preserved_unlaunched_packaging_draft_v1.json':ROOT/'configs/runtime/fm_anomaly_transformer_cold_resource_entry.v1.json',
 'cold_entry_source.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
 'exact_idle_handoff_source.py':ROOT/'scripts/flow_matching/handoff_anomaly_transformer_cold_resources_v2.py',
 'cold_resource_tests.py':ROOT/'tests/test_fm_anomaly_transformer_cold_resource_entry_v1.py',
 'validation.xml':xml,'actual_four_owner_handoff.json':receiptpath}
for role in config['roles']:
    sources[role+'_actual_first_cached_snapshot.json']=ROOT/config['state_root']/role/'actual_first_cached_snapshot.json'
    sources[role+'_actual_handoff_receipt.json']=ROOT/config['state_root']/role/'handoff_receipt.json'
for name,path in sources.items():shutil.copyfile(path,folder/name)
write(folder/'actual_current_live_owners.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    controllers=current,released_private_commit_bytes=receipt['sum_released_controller_private_commit_bytes'],
    release_measurement='Sum of each old process private commit minus successor first actual snapshot private commit, not a physical-RAM claim',
    original_registered_slots=dict(author=25,restored=5,paper=25,sensitivity=400),
    no_numerical_worker_stopped=True,benchmark_performance_claim=False,all_original_experiments_complete=False))
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text('Cold resource imports for four full Anomaly Transformer queues\n\n'
 'The original direct heavy_resources import loaded run_tsad_queue,which imported NumPy/Pandas '
 'through prepare_tsad_execution in each waiting metadata process. The four exact idle owners '
 'each held about802MiBprivate commit. Fresh controllers now expose only the original AST-compiled '
 'exclusive_lock,has_headroom,guarded_wait,terminate_tree and resource_slot through isolated '
 'resource import facades,using the existing cached Windows snapshot type. Original guard bodies '
 'and threshold comparisons are retained,not reimplemented. All numerical children start fresh '
 'original Python commands and receive the frozen CPU-1environment policy.\n\n'
 'Exact OS PID/create-time/command and waiting/no-active-worker/console-only-descendant conditions '
 'were checked before and after suspending each idle metadata owner. No numerical child was '
 'stopped. Actual new main-loop resource snapshots prove NumPy/Pandas/Torch remained unimported '
 'and successor private commit was about14-16MiB. Sum of measured per-controller private-commit '
 'reduction is '+str(receipt['sum_released_controller_private_commit_bytes']/1024**3)+'GiB. '
 'This is private commit,not a claim that the same quantity of physical RAM was freed; host commit '
 'also changes with unrelated live work.\n\n'
 'All complete25author slots(including five unavailable original SWaT-loader slots),five restored '
 'SWaT alternatives,25paper-contract main models and400sensitivity jobs remain. No original '
 'source/config/data/checkpoint/model/budget was edited; all32GiBphysical/40GiBcommit admission '
 'and16GiBemergency guards remain. All four new controllers were actually live at capture. '
 'There was no benchmark completion or performance result added by this scheduler iteration. '
 'The unchanged full45/681/273minimum objective and additional known paper experiments remain.\n\n'
 'Seven checks passed:all four exact entries stay numerically cold,original guard ASTs and exact '
 'admission boundaries are retained,duplicate facade installation is refused,and100real Windows '
 'snapshots do not grow ctypes pointer-type cache. The preserved unlaunched v1packaging draft '
 'pinned an ignored scratch helper; active v2instead pins the tracked side-effect-free handoff '
 'module. That prelaunch packaging change did not launch or alter any numerical run.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=7,
    artifact_files_sha256=files,numerical_workers_stopped=0,benchmark_results_added=0,
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),relative,
 'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
 'scripts/flow_matching/handoff_anomaly_transformer_cold_resources_v2.py',
 'tests/test_fm_anomaly_transformer_cold_resource_entry_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_cold_resource_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),tests_passed=7,
    freed_controller_private_commit_GiB=receipt['sum_released_controller_private_commit_bytes']/1024**3,
    live_successors=[r['pid'] for r in current],numerical_workers_stopped=0)))
