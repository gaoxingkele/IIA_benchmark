from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_pi_causal1107_deferred_full_replay_v1 as deferred
from scripts.flow_matching import run_pi_causal1107_completed_full_replay_v1 as replay
from scripts.flow_matching import register_pi_causal1107_completed_full_replay_v1 as registration
CONFIG='configs/runtime/fm_pi_causal1107_deferred_full_replay.v1.json'
OUTPUT=ROOT/'projects/flow_matching_research/results/2026-10-15/pi_causal1107_deferred_full_replay_preparation_v1'
tests=ROOT/'tmp/Pi_causal1107_deferred_full_replay_validation_20261015_v1.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==66
assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
queue=registration.read(deferred.ORIGINAL_QUEUE)
job=next(j for j in queue['jobs'] if j['id']==replay.CANONICAL)
assert not (ROOT/job['output_directory']/'result.json').exists()
assert not (ROOT/deferred.RUNTIME).exists()
source=psutil.Process(5356)
assert source.create_time()==1792021652.8531024
expected=dict(pid=source.pid,create_time=source.create_time(),command=source.cmdline())
templates=[
 'configs/reproducibility/fm_pi_original_source_full_checkpoint_replay.2026-10-11_v1.json',
 'configs/reproducibility/fm_pi_original_source_recovery_completion.2026-10-11_v1.json',
 deferred.REVIEW_TEMPLATE]
paths={deferred.ORIGINAL_QUEUE,*templates,
 'scripts/flow_matching/run_pi_causal1107_deferred_full_replay_v1.py',
 'scripts/flow_matching/run_pi_causal1107_completed_full_replay_v1.py',
 'scripts/flow_matching/register_pi_causal1107_completed_full_replay_v1.py',
 'tests/test_fm_pi_causal1107_completed_full_replay_v1.py',
 'tests/test_fm_pi_causal1107_deferred_full_replay_v1.py',
 'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
 'scripts/flow_matching/cached_resource_snapshot_v1.py',
 'scripts/flow_matching/run_light_controller.py'}
for template in templates:
 for pin in registration.read(template)['source_receipts']:
  assert deferred.digest(ROOT/pin['path'])==pin['sha256'],pin['path']
  paths.add(pin['path'])
for pin in job['frozen_files']:
 assert deferred.digest(ROOT/pin['path'])==pin['sha256'],pin['path']
 paths.add(pin['path'])
config=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
 python=registration.read(deferred.REVIEW_TEMPLATE)['python'],original_queue=deferred.ORIGINAL_QUEUE,
 canonical_id=replay.CANONICAL,original_scope=replay.SCOPE,original_budget=replay.BUDGET,
 source_worker=expected,completed_replay_runtime=deferred.RUNTIME,poll_seconds=5,
 state_root='experiments/runs/fm_pi_causal1107_deferred_full_replay_v1',
 source_receipts=[dict(path=p,sha256=deferred.digest(ROOT/p)) for p in sorted(paths)],
 boundary='Only wait on the exact already-running original source. Require complete3x518 training and seven canonical artifact hashes before creating any replay configuration. Then freeze and launch the original full point/native/independent audit with unchanged guards, data, numerical budgets and all1140/six-seed scope. Never stop/restart source training, shorten a budget or retry numerical failure automatically.')
deferred.verify(config)
assert deferred.ready_or_exact_worker_live(config,queue) is False
with (ROOT/CONFIG).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
OUTPUT.mkdir(parents=True,exist_ok=False)
environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
command=[config['python'],'-B','-X','utf8','-u','-m','scripts.flow_matching.run_pi_causal1107_deferred_full_replay_v1','--config',CONFIG]
probe=subprocess.run(command+['--probe-only'],cwd=ROOT,env=environment,check=True,capture_output=True,text=True,encoding='utf-8',timeout=55)
actual=json.loads(probe.stdout)
assert actual['passed'] and actual['original_source_worker_exact_live'] and not actual['original_source_already_complete']
assert not actual['future_full_replay_runtime_created'] and not actual['numerical_training_executed']
assert not actual['torch_imported'] and not actual['numpy_imported']
state=ROOT/config['state_root'];state.mkdir(exist_ok=False)
workers={5356:1792021652.8531024,32892:1792026271.4812093,36952:1792006037.8557332,41992:1791948094.8603492}
before={}
for pid,ct in workers.items():
 process=psutil.Process(pid);assert process.create_time()==ct
 before[str(pid)]=dict(pid=pid,create_time=ct,command=process.cmdline())
with (state/'controller.stdout.log').open('xb') as out,(state/'controller.stderr.log').open('xb') as err:
 process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 identity=dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time(),command=command)
deadline=time.monotonic()+20
while time.monotonic()<deadline and not (state/'status.json').exists():
 assert process.poll() is None,'Inspect preserved failure; do not relaunch'
 time.sleep(.2)
assert (state/'status.json').exists(),'Observation deadline: inspect the same live process; never relaunch'
current=registration.read(str(state/'status.json'))
assert current['pid']==identity['pid'] and current['create_time']==identity['create_time']
assert current['state']=='waiting_exact_original_full_training_result'
pending=psutil.Process(identity['pid']);assert pending.cmdline()==identity['command']
after={}
for pid,ct in workers.items():
 original=psutil.Process(pid)
 assert original.create_time()==ct and original.cmdline()==before[str(pid)]['command']
 after[str(pid)]=dict(exact_live=True,cpu_seconds=sum(original.cpu_times()[:2]))
report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
 configuration=CONFIG,configuration_sha256=deferred.digest(ROOT/CONFIG),
 original_scope=replay.SCOPE,full_original_jobs=1140,original_training_seeds=[42,1103,1104,1105,1106,1107],
 canonical_id=replay.CANONICAL,source_result_already_complete=False,
 future_completed_replay_configuration_created=False,new_numerical_results_published=0,
 original_numerical_processes_modified=False,additional_training_seed_slots=0,
 original_four_numerical_workers_before=before,original_four_numerical_workers_after=after,
 actual_probe=actual,actual_watcher=identity,actual_state=current,
 watcher_private_bytes=pending.memory_info().private,validation_checks_passed=66,
 actual_runtime_and_score_replay_completed=False,full_goal_complete=False)
(OUTPUT/'preparation_actual_preflight_and_waiting_launch.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
shutil.copyfile(tests,OUTPUT/'validation.xml')
shutil.copyfile(Path(__file__),OUTPUT/'preparation_and_launcher_source.py')
(OUTPUT/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(OUTPUT/'README.md').write_text('''# Causal Pi-Transformer PSM1107 deferred full replay

The exact original causal-axis-repaired PSM seed1107 training is already running.
This metadata-only watcher observes its OS PID/time/command. It never stops or
restarts the original worker and does not retry numerical failures automatically.

Before any full-replay configuration is created, the source gate requires the
actual complete three epochs of518steps each,1,930,289 parameters, one active
attention layer, all132382training windows,13238200 full training-score points,
87800 covered test-score points with the original41-point omitted tail, and all
seven canonical output hashes. The entire1140-job registry and original seeds
42,1103,1104,1105,1106,1107 remain required. All15 saved/replayed score streams,
native batch256/width512/CPU2, original guards and tolerances are preserved.

After a valid source finishes, a separate frozen registration binds its exact
checkpoint/scores and runs the existing full point, native checkpoint and
independent replay stages. Only review boundary prose is aligned to the actual
canonical model before freezing; original templates and numerical settings stay
unchanged. Historical training guard evidence is not certified by later replay.

All66 source-readiness/identity/scope/replay checks and actual fresh-process
preflight passed. The watcher is live, imports neither NumPy nor Torch, and the
four current numerical workers kept their original identities and commands.
No future replay configuration, new training seed or numerical result existed
at this preparation snapshot. This is readiness evidence, not a completed
experiment, author-equivalence claim, strict TAB result or full-goal completion.
''',encoding='utf-8')
targets=[OUTPUT.relative_to(ROOT).as_posix(),CONFIG,
 'scripts/flow_matching/run_pi_causal1107_deferred_full_replay_v1.py',
 'scripts/flow_matching/run_pi_causal1107_completed_full_replay_v1.py',
 'scripts/flow_matching/register_pi_causal1107_completed_full_replay_v1.py',
 'tests/test_fm_pi_causal1107_completed_full_replay_v1.py',
 'tests/test_fm_pi_causal1107_deferred_full_replay_v1.py']
changed=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
 raw=subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True)
 changed.update(p for p in raw.splitlines() if not p.endswith('.log'))
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert changed and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(changed))
(ROOT/'tmp/Pi_causal1107_deferred_replay_preparation_manifest_20261015_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,watcher_pid=identity['pid'],watcher_create_time=identity['create_time'],
 watcher_private_MiB=pending.memory_info().private/1048576,actual_state=current['state'],
 frozen_input_pins=len(paths),validation_checks_passed=66,publication_files=len(changed),numerical_source_unchanged=True)))
