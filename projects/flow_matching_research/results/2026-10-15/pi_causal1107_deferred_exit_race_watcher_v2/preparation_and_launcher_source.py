import copy
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
from scripts.flow_matching import run_pi_causal1107_deferred_full_replay_v2 as adapter
original=adapter.original
PARENT='configs/runtime/fm_pi_causal1107_deferred_full_replay.v1.json'
CONFIG='configs/runtime/fm_pi_causal1107_deferred_full_replay.v2.json'
OUTPUT=ROOT/'projects/flow_matching_research/results/2026-10-15/pi_causal1107_deferred_exit_race_watcher_v2'
parent=original.registration.read(PARENT)
status=original.registration.read(parent['state_root']+'/status.json')
process=psutil.Process(status['pid'])
expected=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline())
assert process.create_time()==status['create_time']
assert adapter.idle_watcher(status,expected,adapter.cold.descendants_of(process))
assert not (ROOT/original.RUNTIME).exists()
tests=ROOT/'tmp/Pi_causal1107_deferred_full_replay_exit_race_validation_20261015_v2.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==71
assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
config=copy.deepcopy(parent)
config.update(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),
 state_root='experiments/runs/fm_pi_causal1107_deferred_full_replay_v2',
 previous_owner=expected,previous_watcher_configuration=PARENT,
 previous_watcher_configuration_sha256=original.digest(ROOT/PARENT),
 previous_state_path=parent['state_root']+'/status.json',
 boundary='Preserve the entire original v1 training/registration/replay contract. Cold-replace only its exact idle metadata watcher. Recheck complete source artifacts after source-exit/PID-query races. Never stop/restart training or weaken a numerical guard.')
extra=[PARENT,'scripts/flow_matching/run_pi_causal1107_deferred_full_replay_v2.py',
 'tests/test_fm_pi_causal1107_deferred_full_replay_v2.py',
 'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py']
pins={p['path']:p['sha256'] for p in parent['source_receipts']}
pins.update({p:original.digest(ROOT/p) for p in extra})
config['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
adapter.verify(config)
assert adapter.ready_or_exact_worker_live(config,original.registration.read(original.ORIGINAL_QUEUE)) is False
with (ROOT/CONFIG).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
OUTPUT.mkdir(parents=True,exist_ok=False)
command=[config['python'],'-B','-X','utf8','-u','-m',
 'scripts.flow_matching.run_pi_causal1107_deferred_full_replay_v2','--config',CONFIG]
environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run(command+['--probe-only'],cwd=ROOT,env=environment,check=True,
 capture_output=True,text=True,encoding='utf-8',timeout=55)
actual=json.loads(probe.stdout)
assert actual['passed'] and actual['original_source_worker_exact_live'] and not actual['original_source_already_complete']
assert not actual['torch_imported'] and not actual['numpy_imported'] and not actual['future_full_replay_runtime_created']
workers={5356:1792021652.8531024,32892:1792026271.4812093,36952:1792006037.8557332,41992:1791948094.8603492}
before={}
for pid,ct in workers.items():
 process=psutil.Process(pid);assert process.create_time()==ct
 before[str(pid)]=dict(pid=pid,create_time=ct,command=process.cmdline())
state=ROOT/config['state_root'];state.mkdir(exist_ok=False)
with (state/'controller.stdout.log').open('xb') as out,(state/'controller.stderr.log').open('xb') as err:
 child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
  stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 current=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
deadline=time.monotonic()+20
while time.monotonic()<deadline:
 if child.poll() is not None:raise RuntimeError('Inspect preserved watcher transition; never relaunch source training')
 if (state/'status.json').exists():
  observed=original.registration.read(str(state/'status.json'))
  if observed.get('state')=='waiting_exact_original_full_training_result':break
 time.sleep(.2)
else:raise RuntimeError('Observation deadline: inspect the same live watcher, never relaunch')
assert observed['pid']==current['pid'] and observed['create_time']==current['create_time']
watcher=psutil.Process(current['pid']);assert watcher.cmdline()==current['command']
try:assert not adapter.cold.exact_live(psutil.Process(expected['pid']),expected)
except psutil.NoSuchProcess:pass
handoff=state/'idle_watcher_transition'
for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json']:
 shutil.copyfile(handoff/file,OUTPUT/file)
assert not original.registration.read(str(handoff/'handoff.json'))['original_native_child_stopped']
after={}
for pid,ct in workers.items():
 worker=psutil.Process(pid);assert worker.create_time()==ct and worker.cmdline()==before[str(pid)]['command']
 after[str(pid)]=dict(exact_live=True,cpu_seconds=sum(worker.cpu_times()[:2]))
report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
 configuration_sha256=original.digest(ROOT/CONFIG),original_scope=original.replay.SCOPE,
 canonical_id=original.replay.CANONICAL,required_original_jobs=1140,
 original_training_seeds=[42,1103,1104,1105,1106,1107],
 actual_probe=actual,previous_idle_watcher=expected,actual_watcher=current,actual_state=observed,
 watcher_private_bytes=watcher.memory_info().private,source_exit_race_handling_tested=True,
 previous_frozen_code_or_configuration_modified=False,original_numerical_workers_before=before,
 original_numerical_workers_after=after,numerical_training_stopped_or_restarted=False,
 new_training_seed_slots=0,new_numerical_results_published=0,
 future_completed_replay_runtime_created=False,validation_checks_passed=71,full_goal_complete=False)
(OUTPUT/'actual_exit_race_safe_watcher_preflight_and_handoff.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
shutil.copyfile(tests,OUTPUT/'validation.xml')
shutil.copyfile(Path(__file__),OUTPUT/'preparation_and_launcher_source.py')
(OUTPUT/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(OUTPUT/'README.md').write_text('''# Causal PSM1107 source-exit-safe deferred replay

The v1 watcher was replaced only while its exact OS PID/time/command, canonical
identity, idle phase and absence of numerical children were verified twice.
Its original frozen code/configuration and dated preparation evidence remain.
The four original numerical training workers retained their identities/commands.

The new watcher retains the v1 full1140-job/six-seed source gate, original Python,
and exact future point/native/independent replay modules/configuration. It adds a
second complete-artifact check when the original source exits between the first
result-file read and the OS PID query. A retired/reused PID alone never promotes
an incomplete source or triggers numerical retraining. The metadata source gate
checks completed result budgets and all seven output hashes; full-score arrays
are then verified by the unchanged guarded numerical replay stages.

All71 relevant tests passed, including source-exit races, missing/incomplete
sources, changed identity, original scope, budgets, guards, tolerances and review
label alignment. Actual fresh-process preflight imported neither NumPy nor Torch.
The new watcher is live and waiting for the already-running canonical source;
no completed-source replay runtime or numerical result existed at this snapshot.
Historical training guard proof, paper equivalence and strict TAB certification
are not inferred from this preparation. The complete research goal remains open.
''',encoding='utf-8')
previous=ROOT/'projects/flow_matching_research/results/2026-10-15/pi_causal1107_deferred_full_replay_preparation_v1/README.md'
with previous.open('a',encoding='utf-8') as stream:
 stream.write('\nThis dated v1 waiting-host snapshot was subsequently superseded by the verified\nidle v2 metadata-watcher handoff. Original source training and all future\nnumerical replay settings remain unchanged; see the adjacent v2 preparation.\n')
old_manifest=json.loads((ROOT/'tmp/Pi_causal1107_deferred_replay_preparation_manifest_20261015_v1.json').read_text())
paths=set(old_manifest['paths'])
targets=[OUTPUT.relative_to(ROOT).as_posix(),CONFIG,
 'scripts/flow_matching/run_pi_causal1107_deferred_full_replay_v2.py',
 'tests/test_fm_pi_causal1107_deferred_full_replay_v2.py']
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
 raw=subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True)
 paths.update(p for p in raw.splitlines() if not p.endswith('.log'))
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),
 previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
(ROOT/'tmp/Pi_causal1107_deferred_exit_race_joint_publication_manifest_20261015_v2.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,waiting_watcher_pid=current['pid'],create_time=current['create_time'],
 watcher_private_MiB=watcher.memory_info().private/1048576,old_idle_watcher_only_stopped=True,
 original_four_trainings_unchanged=True,validation_checks_passed=71,publication_files=len(paths))))
