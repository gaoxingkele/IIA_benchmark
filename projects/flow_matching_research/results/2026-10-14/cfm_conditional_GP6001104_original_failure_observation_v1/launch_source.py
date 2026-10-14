import ast
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parents[1]
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP6001104_original_failure_observation_v1'
command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe','-X','utf8','-u','-m',
    'scripts.flow_matching.audit_cfm_conditional_GP6001104_full80_failure_v1','--config',
    'configs/runtime/fm_cfm_conditional_GP6001104_full80_failure_observation.v1.json']
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,check=True)
report=ast.literal_eval(probe.stdout.decode())
assert report['original_trajectories']==80 and report['points_each']==500 and report['optimizer_updates']==0
assert not report['torch_imported']
(folder/'actual_unchanged_full80_solver_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
with (folder/'controller.stdout.log').open('xb') as out,(folder/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
p=psutil.Process(child.pid)
with (folder/'actual_one_full80_diagnostic_controller_launch.json').open('x') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,create_time=p.create_time(),command=p.cmdline(),
        numerical_models_stopped_or_restarted=0,additional_optimizer_updates=0),stream,indent=2)
print(json.dumps(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())))
