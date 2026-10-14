from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parents[1]
folder=root/'projects/flow_matching_research/results/2026-10-15/pi_source1107_completed_full_replay_v1'
folder.mkdir(exist_ok=True,parents=True)
import xml.etree.ElementTree as ET
suites=list(ET.parse(root/'tmp/Pi_source1107_completed_full_replay_validation_20261015_v1.xml').getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==51 and all(int(x.get('failures',0))==int(x.get('errors',0))==0 for x in suites)
assert not (root/'experiments/runs/fm_pi_transformer_author_v1/jobs/pi_author__source_runtime__psm__full__seed1107/resource_usage.json').exists()
host='C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'
config='configs/runtime/fm_pi_source1107_completed_full_replay.v1.json'
module='scripts.flow_matching.run_pi_source1107_completed_full_replay_v1'
bootstrap="import sys,runpy;from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader;install_cached_loader(lambda snapshot:None);sys.argv=['full-native-replay','--config',"+repr(config)+"];runpy.run_module("+repr(module)+",run_name='__main__')"
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run([host,'-X','utf8','-u','-m',module,'--config',config,'--probe-only'],
    cwd=root,env=env,capture_output=True,check=True)
report=json.loads(probe.stdout)
assert report['required_original_jobs']==1140 and report['required_full_saved_score_streams']==15
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
