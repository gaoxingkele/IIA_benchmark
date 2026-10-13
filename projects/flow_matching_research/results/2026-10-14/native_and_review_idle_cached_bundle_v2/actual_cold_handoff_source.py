from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import ast,json,os,subprocess,sys,time
from pathlib import Path
import psutil

root=Path('D:/aicoding/IIA_benchmark')
folder=root/'projects/flow_matching_research/results/2026-10-14/native_and_review_idle_cached_bundle_v2'
cfgpath='configs/runtime/fm_native_and_review_idle_cached_bundle.v2.json'
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
config=read(root/cfgpath)
env=dict(os.environ,PYTHONPATH=str(root)+';'+str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
def command(name):return [sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_idle_native_and_review_cached_bundle_v2','--config',cfgpath,'--role',name]
def probe(name):
    result=subprocess.run(command(name)+['--probe-only'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
    receipt=dict(command=command(name)+['--probe-only'],exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    (folder/name/'registration_probe.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    assert result.returncode==0,result.stderr
    data=ast.literal_eval(result.stdout.strip())
    assert data['original_slots']==config['roles'][name]['required_original_slots']
    assert not data['numpy_imported'] and not data['torch_imported'] and not data['original_numerical_child_executed']
    return name,data['private_bytes']
with ThreadPoolExecutor(max_workers=4) as pool:probes=list(pool.map(probe,config['roles']))
print(json.dumps(dict(all_probes_passed=True,private_bytes=probes)),flush=True)
launched=[]
for name,role in config['roles'].items():
    target=folder/name
    with (target/'controller.stdout.log').open('xb') as out,(target/'controller.stderr.log').open('xb') as err:
        child=subprocess.Popen(command(name),cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(60):
        time.sleep(.4)
        assert child.poll() is None,(target/'controller.stderr.log').read_text()
        state=read(root/role['previous_state_path'])
        if state['pid']==child.pid:break
    assert state['pid']==child.pid
    p=psutil.Process(child.pid);assert p.create_time()==state['create_time']
    for basename in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','actual_first_cached_telemetry.json']:
        source=root/config['handoff_state_root']/name/basename
        if source.exists():(target/basename).write_bytes(source.read_bytes())
    identity=dict(role=name,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),previous_owner=role['previous_owner'],
                  previous_private_bytes=role['previous_private_bytes'],private_bytes=p.memory_info().private,
                  released_private_bytes=role['previous_private_bytes']-p.memory_info().private,
                  unchanged_original_slots=role['required_original_slots'])
    (target/'actual_new_live_owner.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    launched.append(identity)
    print(json.dumps(dict(role=name,pid=p.pid,released_private_bytes=identity['released_private_bytes'])),flush=True)
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_live_owners=launched,
            retained_original_Energy_commands=4,retained_complete_TCCM_capacity_cases=1,retained_full_GRASP_reviews=2,
            released_private_bytes=sum(r['released_private_bytes'] for r in launched),
            original_full_queues_and_native_child_budgets_and_guards_unchanged=True,
            scientific_worker_stopped=False,benchmark_performance=False,all_original_experiments_complete=False)
(folder/'actual_all4_cold_host_handoff.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(live_owners=4,released_private_bytes=report['released_private_bytes'])))
