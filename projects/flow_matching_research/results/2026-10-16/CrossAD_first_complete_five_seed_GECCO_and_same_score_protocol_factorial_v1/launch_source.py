import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

root = Path(__file__).resolve().parents[1]
folder = Path('F:/aicoding/IIA_Data/experiments/crossad_same_score_protocol_factorial_v1_launch')
folder.mkdir(exist_ok=False)
command = [sys.executable, '-B', '-X', 'utf8', '-u', '-m',
           'scripts.flow_matching.audit_crossad_same_score_protocol_factorial_v1']
with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
    child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
handle = psutil.Process(child.pid)
receipt = dict(pid=child.pid, create_time=handle.create_time(), command=command,
               cwd=str(root), original_numerical_workers_modified=False)
(folder/'launch.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
