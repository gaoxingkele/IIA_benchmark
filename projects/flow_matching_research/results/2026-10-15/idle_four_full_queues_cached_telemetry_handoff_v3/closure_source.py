from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as adapter
from scripts.flow_matching import run_industrial_cached_entry_handoff_v4 as industrial

BASE=ROOT/'projects/flow_matching_research/results/2026-10-15/idle_four_full_queues_cached_telemetry_handoff_v3'
CONFIG='configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_v3.json'
INDUSTRIAL='configs/runtime/fm_industrial_cached_namespace_handoff.2026-10-15_v4.json'
config=adapter.read(ROOT/CONFIG)
registration=adapter.read(BASE/'registration_and_actual_preflight.json')
launch=adapter.read(BASE/'actual_launch.json')
followup=adapter.read(BASE/'industrial_namespace_followup/actual_launch.json')
industrial_config=adapter.read(ROOT/INDUSTRIAL)
industrial.verify(industrial_config)
assert registration['configuration_sha256']==adapter.digest(ROOT/CONFIG)
tests=ET.parse(ROOT/'tmp/idle_four_and_industrial_namespace_tests_20261015.xml')
suites=list(tests.getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==25
assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
current=[]
old_total=new_total=0
for row in launch['actual_successors']:
 name=row['role'];role=adapter.verify(config,name)
 expected=followup if name=='industrial9' else row
 process=psutil.Process(expected['pid'])
 assert adapter.exact_live(process,expected)
 state=adapter.read(ROOT/role['previous_state_path'])
 assert state['pid']==expected['pid'],(name,state.get('pid'))
 assert state.get('active_pid') is None and not state.get('shared_lock_owned')
 assert state['state'] in ['waiting_for_memory','waiting_for_heavy_lock']
 children=adapter.descendants_of(process)
 assert all(adapter.allowed_console_child(c['name'],c['exe']) for c in children)
 old=role['previous_owner']
 try:
  previous=psutil.Process(old['pid'])
  assert not adapter.exact_live(previous,old)
 except psutil.NoSuchProcess:pass
 handoff=ROOT/config['handoff_state_root']/name
 for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','successor_entry.json']:
  value=adapter.read(handoff/file)
  adapter.write(BASE/'actual_handoffs'/name/file,value)
 assert not adapter.read(handoff/'handoff.json')['original_native_child_stopped']
 old_private=registration['previous_live_idle_owners'][name]['actual_private_bytes']
 new_private=process.memory_info().private
 old_total+=old_private;new_total+=new_private
 current.append(dict(role=name,pid=process.pid,create_time=process.create_time(),command=process.cmdline(),
  previous_private_bytes=old_private,current_private_bytes=new_private,
  state_path=role['previous_state_path'],actual_state=state,
  complete_original_queue_slots=role['required_original_slots'],exact_live=True,
  only_console_descendants=True))
for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','successor_entry.json']:
 adapter.write(BASE/'industrial_namespace_followup/actual_handoff'/file,
  adapter.read(ROOT/industrial_config['handoff_state_root']/file))
assert not adapter.read(ROOT/industrial_config['handoff_state_root']/'handoff.json')['original_native_child_stopped']
numeric=[]
for key,expected in launch['original_numerical_workers_before'].items():
 p=psutil.Process(expected['pid']);assert adapter.exact_live(p,expected)
 numeric.append(dict(pid=p.pid,create_time=p.create_time(),exact_live=True,
  cpu_seconds=sum(p.cpu_times()[:2]),original_command_unchanged=True))
for version in ['v2','v3']:
 frozen=adapter.read(ROOT/('configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_'+version+'.json'))
 for pin in frozen['source_receipts']:assert adapter.digest(ROOT/pin['path'])==pin['sha256']
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
state=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
 original_scope=adapter.SCOPE,actual_live_original_queue_successors=current,
 previous_metadata_private_bytes=old_total,current_metadata_private_bytes=new_total,
 measured_metadata_private_bytes_released=old_total-new_total,
 original_three_numerical_workers=numeric,all_original_commands_queues_seeds_and_budgets_retained=True,
 every_original_admission_emergency_and_memory_cap_retained=True,
 original_numerical_children_stopped=False,raw_data_or_checkpoints_modified=False,
 actual_resource_snapshot=resource_snapshot(),
 validation=dict(passed=25,failures=0,errors=0,skipped=0),
 scientific_experiment_completion_claimed=False,
 initial_v2_schema_failure_preserved=True,industrial_v3_runpy_namespace_followup_preserved=True,
 full_goal_complete=False)
adapter.write(BASE/'actual_current_four_queue_handoff_closure.json',state)
shutil.copyfile(ROOT/'tmp/idle_four_and_industrial_namespace_tests_20261015.xml',BASE/'complete_validation.xml')
shutil.copyfile(ROOT/'tmp/launch_idle_four_cached_telemetry_20261015_v3.py',BASE/'launcher_source.py')
shutil.copyfile(Path(__file__),BASE/'closure_source.py')
with (BASE/'README.md').open('a',encoding='utf-8') as f:
 f.write('\n## Actual transition verification\n\nAll four original waiting metadata owners were replaced only after exact OS PID,\ncreation time, complete command, idle state, no numerical children, and no owned\nshared resource lock were checked twice. The original three numerical training\nprocesses remained live with unchanged identities. Actual current owner/state\nproofs and all transition receipts are retained.\n\nThe initial v2 registration failed before handoff because industrial recovery\nuses original_id instead of id; its source, test record and failure are preserved.\nThe successful v3 industrial successor was subsequently handed off while idle\nto v4 so the original main keeps the cached loader installed in its module\nnamespace. This preserves the original full9 command arguments and numerical\nchildren. All 25 relevant tests passed.\n\n')
 f.write(f'Measured waiting-controller private memory changed from {old_total/1048576:.2f} MiB\nto {new_total/1048576:.2f} MiB at the recorded observations. This is\nmetadata-process memory evidence, not benchmark performance or guaranteed global\ncommit headroom. Pi full checkpoint replay and the remaining full experiment\nqueues still require their unchanged resource admission conditions.\n')
paths=[
 'scripts/flow_matching/run_idle_exact_command_cached_handoff_v2.py',
 'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
 'scripts/flow_matching/run_industrial_cached_entry_handoff_v4.py',
 'tests/test_fm_idle_exact_command_cached_handoff_v2.py',
 'tests/test_fm_idle_exact_command_cached_handoff_v3.py',
 'tests/test_fm_industrial_cached_entry_handoff_v4.py',
 'configs/runtime/fm_idle_four_full_queues_cached_telemetry.2026-10-15_v2.json',
 CONFIG,INDUSTRIAL]
for folder in [BASE,BASE.with_name(BASE.name.replace('_v3','_v2'))]:
 paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),
 previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
adapter.write(ROOT/'tmp/Idle_four_cached_telemetry_publication_manifest_20261015_v3.json',manifest)
print(json.dumps(dict(passed=True,original_scope=adapter.SCOPE,actual_current_owner_pids=[r['pid'] for r in current],
 metadata_private_before_MiB=old_total/1048576,metadata_private_after_MiB=new_total/1048576,
 measured_private_released_MiB=(old_total-new_total)/1048576,tests_passed=25,publication_files=len(paths))))
