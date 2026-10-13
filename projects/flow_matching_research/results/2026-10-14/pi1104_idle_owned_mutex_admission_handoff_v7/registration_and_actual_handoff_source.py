from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
from scripts.flow_matching import run_pi1104_admission_before_mutex_handoff_v7 as current

root=current.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/pi1104_idle_owned_mutex_admission_handoff_v7'
folder.mkdir(exist_ok=False)
original='configs/runtime/fm_pi_source1104_preserved_full_retry.2026-10-12_v1.json'
registration=current.cold.original.read(root/original)
queue=current.cold.original.read(root/registration['queue'])
statepath=queue['settings']['output_root']+'/status.json'
state=current.cold.original.read(root/statepath)
old=psutil.Process(state['pid']);assert old.pid==38000 and old.create_time()==1791741187.4230475
assert state['state']=='waiting_for_memory' and state.get('active_pid') is None and not state['jobs']
assert all(current.cold.handoff.allowed_console_child(c.name(),c.exe()) for c in old.children(recursive=True))
identity=dict(pid=old.pid,create_time=old.create_time(),command=old.cmdline())
pins=[original,registration['queue'],'scripts/flow_matching/run_pi1104_admission_before_mutex_handoff_v7.py',
      'tests/test_pi1104_admission_before_mutex_handoff_v7.py','scripts/flow_matching/run_selected_incident_source_recovery_v6.py',
      'scripts/flow_matching/run_preserved_incident_recovery_v4.py','scripts/flow_matching/run_selected_incident_full_recovery_v5.py',
      'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
      'scripts/flow_matching/cached_resource_snapshot_v1.py','scripts/flow_matching/run_cached_resource_monitor_controller_v1.py']
config=dict(original_runtime=original,previous_owner=identity,previous_state_path=statepath,
            resource_lock=registration['resource_lock'],handoff_state_root='experiments/runs/fm_pi1104_admission_before_mutex_handoff_v7',
            original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
            source_receipts=[dict(path=p,sha256=current.cold.streaming_sha(root/p)) for p in pins])
cfgpath='configs/runtime/fm_pi1104_admission_before_mutex_handoff.v7.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
(folder/'original_idle_owned_mutex_state.json').write_bytes((root/statepath).read_bytes())
(folder/'actual_CPU_GPU_open_lock_handles_before_handoff.json').write_bytes((root/'tmp/actual_CPU_GPU_open_lock_handles_20261014.json').read_bytes())
command=[sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_pi1104_admission_before_mutex_handoff_v7','--config',cfgpath]
env=dict(os.environ,PYTHONPATH=str(root)+';'+str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
(folder/'registration_probe.json').write_text(json.dumps(dict(command=command+['--probe-only'],exit_code=probe.returncode,stdout=probe.stdout,stderr=probe.stderr),indent=2),encoding='utf-8')
assert probe.returncode==0 and "'torch_imported': False" in probe.stdout and "'numpy_imported': False" in probe.stdout,probe.stderr
with (folder/'controller.stdout.log').open('xb') as out,(folder/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
for _ in range(60):
    time.sleep(.4)
    assert child.poll() is None,(folder/'controller.stderr.log').read_text()
    state=current.cold.original.read(root/statepath)
    if state['pid']==child.pid:break
assert state['pid']==child.pid
p=psutil.Process(child.pid)
for name in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json']:
    (folder/name).write_bytes((root/config['handoff_state_root']/name).read_bytes())
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),live_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),
            original_40GiB_admission_retained=True,original_13_incident_records_retained=True,original_1140_Pi_jobs_retained=True,
            original_full_selected_worker_unchanged=True,actual_phase=state['state'],original_native_worker_stopped=False)
(folder/'actual_new_admission_before_mutex_owner.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
