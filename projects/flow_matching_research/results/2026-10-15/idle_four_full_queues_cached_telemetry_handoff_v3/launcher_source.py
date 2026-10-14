import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as adapter
CONFIG='configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_v3.json'
OUTPUT=ROOT/'projects/flow_matching_research/results/2026-10-15/idle_four_full_queues_cached_telemetry_handoff_v3'
config=adapter.read(ROOT/CONFIG)
proof=adapter.read(OUTPUT/'registration_and_actual_preflight.json')
assert proof['passed'] and proof['configuration_sha256']==adapter.digest(ROOT/CONFIG)
assert not (OUTPUT/'actual_launch.json').exists()
workers={36952:1792006037.8557332,41992:1791948094.8603492,10012:1792016137.499748}
before={}
for pid,ct in workers.items():
 p=psutil.Process(pid);assert p.create_time()==ct
 before[str(pid)]=dict(pid=pid,create_time=ct,command=p.cmdline())
launches=[]
for name in config['roles']:
 role=adapter.verify(config,name)
 previous=psutil.Process(role['previous_owner']['pid'])
 assert adapter.exact_live(previous,role['previous_owner'])
 assert adapter.idle_allowed(adapter.read(ROOT/role['previous_state_path']),role['previous_owner'],adapter.descendants_of(previous))
 base=ROOT/config['handoff_state_root']/name
 assert not (base/'handoff.json').exists()
 base.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-B','-X','utf8','-u','-m',
  'scripts.flow_matching.run_idle_exact_command_cached_handoff_v3','--config',CONFIG,'--role',name]
 with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
  p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src')))
  launches.append(dict(role=name,pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=command))
 # This is a process-observation deadline, never a reason to relaunch or stop it.
 deadline=time.monotonic()+20
 while time.monotonic()<deadline and not (base/'successor_entry.json').exists():
  if p.poll() is not None:raise RuntimeError('Preserve terminal attempt; inspect '+str(base/'stderr.log'))
  time.sleep(.2)
 assert (base/'successor_entry.json').exists(),'Inspect the same live successor; do not relaunch'
 assert adapter.read(base/'handoff.json')['original_native_child_stopped'] is False
after={}
for pid,ct in workers.items():
 p=psutil.Process(pid);assert p.create_time()==ct and p.cmdline()==before[str(pid)]['command']
 after[str(pid)]=dict(exact_live=True,cpu_seconds=sum(p.cpu_times()[:2]))
receipt=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
 configuration_sha256=adapter.digest(ROOT/CONFIG),original_scope=adapter.SCOPE,
 actual_successors=launches,original_numerical_workers_before=before,
 original_numerical_workers_after=after,all_four_idle_handoffs_passed=True,
 original_training_processes_stopped=False,any_scientific_budget_or_resource_threshold_changed=False)
with (OUTPUT/'actual_launch.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(dict(all_four_idle_handoffs_passed=True,successors=[dict(role=x['role'],pid=x['pid']) for x in launches],original_three_training_processes_preserved=True)))
