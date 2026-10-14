import hashlib
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
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as adapter

CONFIG='configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_v3.json'
OUTPUT='projects/flow_matching_research/results/2026-10-15/idle_four_full_queues_cached_telemetry_handoff_v3'
roles={
 'spectral70':dict(pid=3292,ctime=1791737901.1597052,
   runtime='configs/runtime/fm_spectral_original_resource_order.v4.json',
   queue='configs/experiments/fm_spectral_original_resource_admission_queue.v3.json',
   state='experiments/runs/fm_spectral_mean_flow_original_v1/status.json'),
 'industrial9':dict(pid=29224,ctime=1791645511.5460331,
   runtime='configs/runtime/fm_industrial_empty_output_recovery.v1.json',
   queue='configs/experiments/fm_industrial_empty_output_recovery_queue.v1.json',
   state='experiments/runs/fm_industrial_empty_output_recovery_v1/status.json'),
 'strict175':dict(pid=25448,ctime=1791687813.1598577,
   runtime='configs/runtime/fm_bounded_strict_baseline_cpu.v2.json',
   queue='configs/experiments/fm_strict_baseline_queue.v1.json',
   state='experiments/runs/fm_bounded_strict_baseline_cpu_v2/strict_cpu/status.json'),
 'tranad25':dict(pid=15716,ctime=1791689728.5829513,
   runtime='configs/runtime/fm_tranad_reference_float64.v1.json',
   queue='configs/experiments/fm_tranad_reference_float64_queue.v1.json',
   state='experiments/runs/fm_tranad_reference_float64_v1/controller/status.json'),
}
pins={
 'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
 'tests/test_fm_idle_exact_command_cached_handoff_v3.py',
 'scripts/flow_matching/run_light_controller.py',
 'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
 'scripts/flow_matching/cached_resource_snapshot_v1.py',
 'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
 'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
 'scripts/flow_matching/heavy_resources.py',
 'scripts/flow_matching/heavy_scheduler_v2.py',
}
registration={}
before={}
for name,r in roles.items():
 p=psutil.Process(r['pid'])
 assert p.create_time()==r['ctime']
 command=p.cmdline();index=command.index('-m')
 entry,count=adapter.ENTRIES[name]
 assert command[index+1]==entry
 expected=dict(pid=r['pid'],create_time=r['ctime'],command=command)
 state=adapter.read(ROOT/r['state'])
 children=adapter.descendants_of(p)
 assert adapter.idle_allowed(state,expected,children),(name,state.get('state'),children)
 registration[name]=dict(entry_module=entry,original_arguments=command[index+2:],
   previous_owner=expected,previous_state_path=r['state'],original_runtime=r['runtime'],
   original_queue=r['queue'],required_original_slots=count)
 before[name]=dict(actual_private_bytes=p.memory_info().private,actual_state=state,
   descendants=children)
 pins.update({r['runtime'],r['queue'],entry.replace('.','/')+'.py'})
config=dict(schema_version=3,created_utc=datetime.now(timezone.utc).isoformat(),
 original_scope=adapter.SCOPE,roles=registration,
 handoff_state_root='experiments/runs/fm_idle_four_full_queues_cached_telemetry_v3',
 source_receipts=[dict(path=p,sha256=adapter.digest(ROOT/p)) for p in sorted(pins)],
 boundary='Metadata-only cold handoff. Preserve all original commands, full queues, scientific budgets, random seeds, locks, resource thresholds and guard formulas. No running numerical child may be stopped.')
with (ROOT/CONFIG).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
output=ROOT/OUTPUT;output.mkdir(parents=True,exist_ok=False)
probes={}
for name in registration:
 adapter.verify(config,name)
 response=subprocess.run([sys.executable,'-B','-X','utf8','-m',
  'scripts.flow_matching.run_idle_exact_command_cached_handoff_v3',
  '--config',CONFIG,'--role',name,'--probe-only'],cwd=ROOT,check=True,
  capture_output=True,text=True,encoding='utf-8',timeout=55,
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src')))
 probe=json.loads(response.stdout)
 assert not probe['numpy_imported'] and not probe['torch_imported']
 assert not probe['original_numerical_child_executed']
 assert probe['complete_original_slots']==adapter.ENTRIES[name][1]
 probes[name]=probe
receipt=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
 configuration=CONFIG,configuration_sha256=adapter.digest(ROOT/CONFIG),
 original_scope=adapter.SCOPE,all_four_original_queues_preserved=True,
 numerical_training_executed=False,predecessors_stopped=False,
 previous_live_idle_owners=before,actual_fresh_process_probes=probes)
adapter.write(output/'registration_and_actual_preflight.json',receipt)
shutil.copyfile(ROOT/'tmp/idle_exact_command_cached_handoff_v3_tests_20261015.xml',output/'validation.xml')
shutil.copyfile(ROOT/'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',output/'handoff_source.py')
shutil.copyfile(Path(__file__),output/'registration_source.py')
(output/'README.md').write_text('''# Four full-queue metadata controller handoff

This registration preserves all 70 Spectral/Diffusion-TS, 9 industrial recovery,
175 strict CPU baseline and 25 TranAD float64 jobs, including every existing
source, configuration, queue, seed, budget, lock and resource threshold.

Fresh-process probes import neither NumPy nor Torch. The successor uses the
existing cached Windows telemetry implementation, with the same guard formulas.
Exact live OS PID, creation time and complete command are checked before and
after suspension. Only an idle controller without a numerical child or an owned
shared lock may be replaced. Legacy state timestamps remain absent; observed
OS process identity is recorded separately.

Registration and probes are not numerical experiment results. Actual transition
evidence and successor state must be checked separately before claiming a handoff.
The complete 45-paper / 681-method-record / 273-data-task scope remains pending.
''',encoding='utf-8')
print(json.dumps(dict(configuration=CONFIG,proof=OUTPUT,actual_probes={k:dict(
  complete_slots=v['complete_original_slots'],private_bytes=v['private_bytes']) for k,v in probes.items()})))
