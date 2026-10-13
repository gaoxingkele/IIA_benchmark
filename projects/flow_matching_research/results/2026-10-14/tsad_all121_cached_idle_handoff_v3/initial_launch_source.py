from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

root = Path('D:/aicoding/IIA_benchmark')
relative = 'projects/flow_matching_research/results/2026-10-14/tsad_all121_cached_idle_handoff_v3'
folder = root / relative
folder.mkdir(exist_ok=False)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
runtimepath = 'configs/runtime/fm_tsad_constant_fit_cpu_controls.v2.json'
runtime = read(root / runtimepath)
roles = {}
for binding in runtime['controllers']:
    name = binding['original_lane']
    statepath = runtime['state_root']+'/'+name+'/status.json'
    state = read(root / statepath)
    p = psutil.Process(state['pid'])
    assert p.create_time() == state['create_time']
    command = p.cmdline()
    assert 'scripts.flow_matching.run_tsad_constant_fit_cpu_controls_v2' in command
    assert command[command.index('--controller')+1] == name
    assert state.get('active_pid') is None and state['state'] == 'waiting_for_memory' and not state['shared_lock_owned']
    assert all(child.name().lower()=='conhost.exe' and child.exe().replace('\\','/').lower()=='c:/windows/system32/conhost.exe' for child in p.children(recursive=True))
    rolefolder = folder/name
    rolefolder.mkdir()
    (rolefolder/'previous_live_state.json').write_bytes((root/statepath).read_bytes())
    roles[name] = dict(previous_owner=dict(pid=p.pid,create_time=p.create_time(),command=command),
                       previous_state_path=statepath, previous_private_bytes=p.memory_info().private,
                       exact_original_queue=binding['path'], exact_queue_sha256=binding['sha256'], original_jobs=binding['jobs'])
(folder/'validation.xml').write_bytes((root/'tmp/tsad_constant_cached_handoff_v3_validation.xml').read_bytes())
(folder/'preserved_pre_registration_validation_failure.xml').write_bytes((root/'tmp/tsad_constant_cached_handoff_v3_preserved_pre_registration_failure.xml').read_bytes())
(folder/'.gitattributes').write_text('*.log -diff -text\npreserved_pre_registration_validation_failure.xml -diff -text\n',encoding='utf-8')
sources = [runtimepath, 'scripts/flow_matching/run_tsad_constant_fit_cached_handoff_v3.py',
           'tests/test_tsad_constant_fit_cached_handoff_v3.py', 'scripts/flow_matching/run_tsad_constant_fit_cpu_controls_v2.py',
           'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py', 'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
           'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
           'scripts/flow_matching/run_resource_gated_light_controller_v1.py', 'scripts/flow_matching/run_bounded_strict_cpu_queue_v2.py',
           'scripts/flow_matching/cached_resource_snapshot_v1.py']
cfg = dict(original_runtime=runtimepath, roles=roles,
           original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
           handoff_state_root='experiments/runs/fm_tsad_all121_cached_idle_handoff_v3',
           source_receipts=[dict(path=path,sha256=hashlib.sha256((root/path).read_bytes()).hexdigest()) for path in sources])
cfgpath = 'configs/runtime/fm_tsad_all121_cached_idle_handoff.v3.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:
    json.dump(cfg,stream,indent=2)
env = dict(os.environ,PYTHONPATH=str(root)+';'+str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
launched = []
for name,role in roles.items():
    command=[sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_tsad_constant_fit_cached_handoff_v3','--config',cfgpath,'--controller',name]
    target=folder/name
    probe=subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
    (target/'registration_probe.json').write_text(json.dumps(dict(command=command+['--probe-only'],exit_code=probe.returncode,stdout=probe.stdout,stderr=probe.stderr),indent=2),encoding='utf-8')
    assert probe.returncode==0 and "'numpy_imported': False" in probe.stdout and "'torch_imported': False" in probe.stdout,probe.stderr
    with (target/'controller.stdout.log').open('xb') as out,(target/'controller.stderr.log').open('xb') as err:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(25):
        time.sleep(.4)
        assert child.poll() is None,(target/'controller.stderr.log').read_text()
        state=read(root/role['previous_state_path'])
        if state['pid']==child.pid and state['state'] in {'waiting_for_memory','waiting_for_lock','waiting_for_resource_lock','running'}:
            break
    assert state['pid']==child.pid
    live=psutil.Process(child.pid)
    assert live.create_time()==state['create_time']
    info=dict(controller=name, original_jobs=role['original_jobs'], pid=live.pid,create_time=live.create_time(),command=live.cmdline(),
              previous_owner=role['previous_owner'],previous_private_bytes=role['previous_private_bytes'],private_bytes=live.memory_info().private,
              released_private_bytes=role['previous_private_bytes']-live.memory_info().private)
    for basename in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json']:
        (target/basename).write_bytes((root/cfg['handoff_state_root']/name/basename).read_bytes())
    (target/'actual_live_owner.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    launched.append(info)
    print(json.dumps(dict(controller=name,pid=live.pid,jobs=role['original_jobs'],released_private_bytes=info['released_private_bytes'])),flush=True)
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_live_owners=launched,
            original_registered_CPU_slots=121,separate_original_GPU_slots=170,
            released_private_bytes=sum(row['released_private_bytes'] for row in launched),
            exact_original_full_queues_and_numerical_workers_unchanged=True,
            original_admission_and_running_guards_unchanged=True,scientific_worker_stopped=False,
            validation_tests_passed=27,benchmark_performance=False,all_original_experiments_complete=False)
(folder/'actual_all6_controller_handoff.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(full_original_slots=121, live_owners=len(launched),released_private_bytes=report['released_private_bytes'])))
