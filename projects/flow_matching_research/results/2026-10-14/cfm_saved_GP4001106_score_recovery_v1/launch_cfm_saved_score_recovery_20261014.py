from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root = Path(__file__).resolve().parents[1]
folder = root/'projects/flow_matching_research/results/2026-10-14/cfm_saved_GP4001106_score_recovery_v1'
host = 'C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'
rows = []
for module, config, name in [
    ('scripts.flow_matching.run_cfm_saved_full_score_recovery_v1','configs/runtime/fm_cfm_saved_score_recovery.v1.json','saved_scoring'),
    ('scripts.flow_matching.run_cfm_saved_score_full_replay_v1','configs/runtime/fm_cfm_saved_score_full_replay.v1.json','full_replay')]:
    command = [host,'-X','utf8','-u','-m',module,'--config',config]
    env = dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    probe = subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,check=True)
    report = json.loads(probe.stdout)
    assert not report['torch_imported'] and not report['numpy_imported']
    (folder/(name+'_preflight.json')).write_text(json.dumps(report,indent=2)+'\n')
    with (folder/(name+'_controller.stdout.log')).open('xb') as out, (folder/(name+'_controller.stderr.log')).open('xb') as err:
        child = subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    process = psutil.Process(child.pid)
    rows.append(dict(pid=child.pid,create_time=process.create_time(),command=process.cmdline(),module=module))
with (folder/'actual_guarded_controller_launch.json').open('x') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_handles=rows,
        source_workers_stopped_or_restarted=0,original_guards_changed=False),stream,indent=2)
print(json.dumps(rows,indent=2))
