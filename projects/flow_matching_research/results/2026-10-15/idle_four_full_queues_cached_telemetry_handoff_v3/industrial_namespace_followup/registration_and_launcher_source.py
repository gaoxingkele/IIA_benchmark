import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as prior
from scripts.flow_matching import run_industrial_cached_entry_handoff_v4 as adapter

parent='configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_v3.json'
path='configs/runtime/fm_industrial_cached_namespace_handoff.2026-10-15_v4.json'
base=ROOT/'projects/flow_matching_research/results/2026-10-15/idle_four_full_queues_cached_telemetry_handoff_v3'
launch=prior.read(base/'actual_launch.json')
row=next(r for r in launch['actual_successors'] if r['role']=='industrial9')
owner=psutil.Process(row['pid'])
assert prior.exact_live(owner,row)
original=prior.read(ROOT/parent)['roles']['industrial9']
assert prior.idle_allowed(prior.read(ROOT/original['previous_state_path']),row,prior.descendants_of(owner))
pins=[parent,'scripts/flow_matching/run_industrial_cached_entry_handoff_v4.py',
 'tests/test_fm_industrial_cached_entry_handoff_v4.py']
config=dict(original_scope=prior.SCOPE,original_role='industrial9',original_registration=parent,
 original_arguments=original['original_arguments'],previous_owner={k:row[k] for k in ['pid','create_time','command']},
 previous_state_path=original['previous_state_path'],
 handoff_state_root='experiments/runs/fm_industrial_cached_namespace_handoff_v4',
 source_receipts=[dict(path=p,sha256=prior.digest(ROOT/p)) for p in pins],
 boundary='Preserve the original full9 entry and arguments, use its existing module namespace so the installed cached resource loader survives. All original scientific settings and resource guards remain exact.')
with (ROOT/path).open('x',encoding='utf-8') as f:json.dump(config,f,indent=2)
proof=base/'industrial_namespace_followup';proof.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'))
command=[sys.executable,'-B','-X','utf8','-u','-m',
 'scripts.flow_matching.run_industrial_cached_entry_handoff_v4','--config',path]
probe=subprocess.run(command+['--probe-only'],cwd=ROOT,check=True,capture_output=True,
 text=True,encoding='utf-8',timeout=55,env=env)
actual=json.loads(probe.stdout)
assert actual['passed'] and not actual['numpy_imported'] and not actual['torch_imported']
prior.write(proof/'actual_probe.json',actual)
adapter.verify(config)
statebase=ROOT/config['handoff_state_root'];statebase.mkdir(exist_ok=False)
with (statebase/'stdout.log').open('xb') as out,(statebase/'stderr.log').open('xb') as err:
 process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
  env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 ct=psutil.Process(process.pid).create_time()
deadline=time.monotonic()+20
while time.monotonic()<deadline and not (statebase/'successor_entry.json').exists():
 assert process.poll() is None,'Inspect preserved failure; never automatically relaunch'
 time.sleep(.2)
assert (statebase/'successor_entry.json').exists(),'Inspect this same process; do not relaunch'
prior.write(proof/'actual_launch.json',dict(configuration=path,configuration_sha256=prior.digest(ROOT/path),
 pid=process.pid,create_time=ct,command=command,previous_idle_successor=row,
 original_scientific_settings_unchanged=True,numerical_child_stopped=False,
 reason='The original light-controller module must retain its installed cached load_functions binding; runpy would redefine it in __main__.'))
shutil.copyfile(ROOT/'scripts/flow_matching/run_industrial_cached_entry_handoff_v4.py',proof/'handoff_source.py')
shutil.copyfile(Path(__file__),proof/'registration_and_launcher_source.py')
shutil.copyfile(ROOT/'tmp/idle_four_and_industrial_namespace_tests_20261015.xml',proof/'validation.xml')
print(json.dumps(dict(passed=True,pid=process.pid,actual_loaded_controller_snapshot=actual['actual_loaded_controller_resource_snapshot_name'],full_original_industrial_jobs=9)))
