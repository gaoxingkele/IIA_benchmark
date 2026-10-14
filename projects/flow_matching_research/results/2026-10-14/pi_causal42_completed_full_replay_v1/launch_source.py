from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parents[1]
folder=root/'projects/flow_matching_research/results/2026-10-14/pi_causal42_completed_full_replay_v1'
folder.mkdir(exist_ok=True,parents=True)
host='C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'
config='configs/runtime/fm_pi_causal42_completed_full_replay.v1.json'
module='scripts.flow_matching.run_pi_causal42_completed_full_replay_v1'
bootstrap="import sys,runpy;from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader;install_cached_loader(lambda snapshot:None);sys.argv=['full-native-replay','--config',"+repr(config)+"];runpy.run_module("+repr(module)+",run_name='__main__')"
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run([host,'-X','utf8','-u','-m',module,'--config',config,'--probe-only'],
    cwd=root,env=env,capture_output=True,check=True)
report=json.loads(probe.stdout)
assert report['required_original_jobs']==1140 and report['all_original_score_streams_required']==15
assert not report['torch_imported'] and not report['numpy_imported']
(folder/'actual_all_full_score_streams_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
with (folder/'controller.stdout.log').open('xb') as out,(folder/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen([host,'-X','utf8','-u','-c',bootstrap],cwd=root,env=env,stdout=out,stderr=err,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
p=psutil.Process(child.pid)
with (folder/'actual_original_full_replay_controller_launch.json').open('x') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,create_time=p.create_time(),
        command=p.cmdline(),metadata_only_cached_resource_loader=True,original_numeric_workers_changed=False),stream,indent=2)
print(json.dumps(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())))
