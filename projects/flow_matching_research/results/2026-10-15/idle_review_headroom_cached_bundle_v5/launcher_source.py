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
from scripts.flow_matching import run_idle_review_headroom_cached_bundle_v5 as adapter
base=adapter.original
CONFIG='configs/runtime/fm_idle_review_headroom_cached_bundle.2026-10-15_v5.json'
OUTPUT=ROOT/'projects/flow_matching_research/results/2026-10-15/idle_review_headroom_cached_bundle_v5'
config=base.read(ROOT/CONFIG)
proof=base.read(OUTPUT/'registration_and_actual_guard_api_preflight.json')
assert proof['passed'] and proof['configuration_sha256']==base.digest(ROOT/CONFIG)
assert not (OUTPUT/'actual_launch.json').exists()
workers={5356:1792021652.8531024,32892:1792026271.4812093,
 36952:1792006037.8557332,41992:1791948094.8603492}
before={}
for pid,ct in workers.items():
 process=psutil.Process(pid);assert process.create_time()==ct
 before[str(pid)]=dict(pid=pid,create_time=ct,command=process.cmdline())
launched=[]
for name in config['roles']:
 role=adapter.verify(config,name)
 previous=psutil.Process(role['previous_owner']['pid'])
 assert base.exact_live(previous,role['previous_owner'])
 assert adapter.idle_allowed(base.read(ROOT/role['previous_state_path']),role['previous_owner'],base.descendants_of(previous))
 state=ROOT/config['handoff_state_root']/name
 assert not (state/'handoff.json').exists()
 state.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-B','-X','utf8','-u','-m',
  'scripts.flow_matching.run_idle_review_headroom_cached_bundle_v5','--config',CONFIG,'--role',name]
 with (state/'stdout.log').open('xb') as out,(state/'stderr.log').open('xb') as err:
  process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src')),
   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  row=dict(role=name,pid=process.pid,create_time=psutil.Process(process.pid).create_time(),command=command)
  launched.append(row)
 deadline=time.monotonic()+20
 while time.monotonic()<deadline and not (state/'successor_entry.json').exists():
  if process.poll() is not None:raise RuntimeError('Preserve stopped attempt; inspect '+str(state/'stderr.log'))
  time.sleep(.2)
 assert (state/'successor_entry.json').exists(),'Observation deadline: inspect the same live successor, never relaunch'
 assert not base.read(state/'handoff.json')['original_native_child_stopped']
after={}
for pid,ct in workers.items():
 process=psutil.Process(pid);assert base.exact_live(process,before[str(pid)])
 after[str(pid)]=dict(exact_live=True,cpu_seconds=sum(process.cpu_times()[:2]))
receipt=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
 configuration_sha256=base.digest(ROOT/CONFIG),original_scope=base.SCOPE,
 actual_successors=launched,original_numerical_workers_before=before,original_numerical_workers_after=after,
 all_four_idle_handoffs_passed=True,original_numerical_children_stopped=False,
 scientific_budgets_or_resource_settings_changed=False)
with (OUTPUT/'actual_launch.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(dict(all_four_idle_handoffs_passed=True,
 successors=[dict(role=r['role'],pid=r['pid']) for r in launched],four_original_trainings_preserved=True)))
