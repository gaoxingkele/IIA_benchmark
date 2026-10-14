from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parents[1]
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_saved_GP4001106_boundary_handoff_v2'
host='C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'
command=[host,'-X','utf8','-u','-m','scripts.flow_matching.run_cfm_saved_score_queued_handoff_v2',
    '--config','configs/runtime/fm_cfm_saved_score_queued_handoff.v2.json']
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
preflight=subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,check=True)
report=json.loads(preflight.stdout)
assert not report['numpy_imported'] and not report['torch_imported'] and report['private_bytes']<64*1024**2
(folder/'actual_frugal_original_guards_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
source=psutil.Process(35452)
assert source.create_time()==1791769181.137207
original_tree=[dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()) for p in source.children(recursive=True)]
with (folder/'controller.stdout.log').open('xb') as out,(folder/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
p=psutil.Process(child.pid)
with (folder/'actual_new_boundary_controller_launch.json').open('x') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,create_time=p.create_time(),
        command=p.cmdline(),original_live_numeric_tree_before_handoff=original_tree,
        numeric_workers_stopped_or_restarted=0),stream,indent=2)
print(json.dumps(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())))
