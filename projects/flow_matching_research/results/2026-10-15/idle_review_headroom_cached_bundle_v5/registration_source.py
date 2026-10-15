import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_idle_review_headroom_cached_bundle_v5 as adapter
base=adapter.original
CONFIG='configs/runtime/fm_idle_review_headroom_cached_bundle.2026-10-15_v5.json'
OUTPUT='projects/flow_matching_research/results/2026-10-15/idle_review_headroom_cached_bundle_v5'
ROUTES={
 'giflow140':(2436,1791694031.1243775,'configs/experiments/fm_giflow_native_queue.v5.json',
   'configs/experiments/fm_giflow_native_queue.v5.json','experiments/runs/fm_giflow_native_v5/status.json'),
 'spectral_operator70':(17824,1791696024.8057675,'configs/reproducibility/fm_spectral_full_sampling_derivative_graph.2026-10-11_v2.json',
   'configs/experiments/fm_spectral_mean_flow_original_queue.v2.json','experiments/runs/fm_spectral_full_sampling_derivative_graph_v2/status.json'),
 'cfm_retry285':(26000,1791741181.469874,'configs/runtime/fm_cfm_ts_preserved_seed1103_retry.2026-10-12_v1.json',
   'configs/experiments/fm_cfm_ts_preserved_seed1103_retry_queue.2026-10-12_v1.json','experiments/runs/fm_cfm_ts_preserved_seed1103_retry_v1/status.json'),
 'ls4_physio10':(38132,1791755147.2510471,'configs/runtime/fm_ls4_physio_full_observation_dispatch.v2.json',
   'configs/experiments/fm_ls4_native_physionet_queue.v3.json','experiments/runs/fm_ls4_physio_full_observation_dispatch_v1/status.json'),
}

if '--actual-role-probe' in sys.argv:
 name=sys.argv[sys.argv.index('--actual-role-probe')+1]
 _,_,runtime_path,queue_path,_=ROUTES[name]
 role=dict(original_queue=queue_path)
 observed=adapter.actual_guard_api_probe(name,role)
 module=importlib.import_module(adapter.ENTRIES[name][0])
 runtime=base.read(ROOT/runtime_path);queue=base.read(ROOT/queue_path)
 if name=='giflow140':
  module.protocol.verify(queue)
  assert module.memory_preflight_ready(queue)
 elif name=='spectral_operator70':module.binding(runtime)
 elif name=='cfm_retry285':module.original.verify(queue)
 elif name=='ls4_physio10':module.verify(runtime)
 assert len(queue['jobs'])==adapter.ENTRIES[name][1]
 observed.update(original_registry_jobs=len(queue['jobs']),actual_original_metadata_verifier_passed=True)
 print(json.dumps(observed));sys.exit(0)

roles={};before={};probes={}
pins={'scripts/flow_matching/run_idle_review_headroom_cached_bundle_v5.py',
 'tests/test_fm_idle_review_headroom_cached_bundle_v5.py',
 'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
 'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
 'scripts/flow_matching/cached_resource_snapshot_v1.py',
 'scripts/flow_matching/run_light_controller.py',
 'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py'}
for name,(pid,ct,runtime_path,queue_path,state_path) in ROUTES.items():
 process=psutil.Process(pid);assert process.create_time()==ct
 command=process.cmdline();index=command.index('-m')
 entry,count=adapter.ENTRIES[name];assert command[index+1]==entry
 expected=dict(pid=pid,create_time=ct,command=command)
 state=base.read(ROOT/state_path);children=base.descendants_of(process)
 assert adapter.idle_allowed(state,expected,children),(name,state.get('state'),children)
 roles[name]=dict(entry_module=entry,original_arguments=command[index+2:],
   previous_owner=expected,previous_state_path=state_path,
   original_runtime=runtime_path,original_queue=queue_path,required_original_slots=count)
 before[name]=dict(actual_private_bytes=process.memory_info().private,actual_state=state,descendants=children)
 probe=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--actual-role-probe',name],
   cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8',timeout=55,
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src')))
 value=json.loads(probe.stdout)
 assert not value['numpy_imported'] and not value['torch_imported'] and not value['entry_main_executed']
 assert value['actual_original_metadata_verifier_passed'] and value['actual_guard_snapshot']=='cached_snapshot'
 probes[name]=value
 pins.update({runtime_path,queue_path,entry.replace('.','/')+'.py'})
 for path in {runtime_path,queue_path}:
  data=base.read(ROOT/path)
  for pin in data.get('source_receipts',[]):
   assert base.digest(ROOT/pin['path'])==pin['sha256'],pin['path']
   pins.add(pin['path'])
config=dict(schema_version=5,created_utc=datetime.now(timezone.utc).isoformat(),
 original_scope=base.SCOPE,roles=roles,handoff_state_root='experiments/runs/fm_idle_review_headroom_cached_bundle_v5',
 source_receipts=[dict(path=p,sha256=base.digest(ROOT/p)) for p in sorted(pins)],
 boundary='Cold handoff only idle metadata owners. Retain every original entry argument, registry job, worker, numerical budget, resource mutex, admission minimum, emergency reserve, cap and cadence. No active numerical child may be stopped. This does not certify any experimental score.')
for name in roles:adapter.verify(config,name)
with (ROOT/CONFIG).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
target=ROOT/OUTPUT;target.mkdir(parents=True,exist_ok=False)
base.write(target/'registration_and_actual_guard_api_preflight.json',dict(passed=True,
 captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,configuration_sha256=base.digest(ROOT/CONFIG),
 original_scope=base.SCOPE,previous_live_idle_owners=before,actual_fresh_process_guard_api_probes=probes,
 numerical_training_executed=False,predecessor_processes_stopped=False))
shutil.copyfile(ROOT/'tmp/idle_review_headroom_cached_bundle_v5_tests_20261015.xml',target/'validation.xml')
shutil.copyfile(ROOT/'scripts/flow_matching/run_idle_review_headroom_cached_bundle_v5.py',target/'handoff_source.py')
shutil.copyfile(Path(__file__),target/'registration_source.py')
(target/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(target/'README.md').write_text('''# Idle full-registry cached resource telemetry

This handoff retains the exact original GiFlow140, Spectral70-registry operator
audit, CFM285-registry selected registered recovery and LS4 full10 controller
commands. It changes only fresh host resource telemetry binding. Every original
worker, selected canonical identity, queue, budget, mutex and resource safeguard
remains intact. The original running numerical processes are not touched.

All four actual resource APIs and their original metadata verifiers passed in
fresh processes without NumPy/Torch imports or numerical execution. Exact OS
PID/time/command and the idle/no-child/no-owned-shared-lock state are checked
twice around suspension before replacing a metadata owner. GPU resource waiting
is accepted solely as an idle host phase; GPU admission is not changed.

All30 relevant tests passed. Registration is not a handoff receipt, a completed
numerical experiment or a benchmark performance claim. The complete original
45-paper/681-method-record/273-data-task scope remains required.
''',encoding='utf-8')
print(json.dumps(dict(passed=True,configuration=CONFIG,frozen_source_pins=len(pins),
 actual_fresh_process_guard_probes={k:dict(original_registry_jobs=v['original_registry_jobs'],
  private_bytes=v['private_bytes']) for k,v in probes.items()})))
