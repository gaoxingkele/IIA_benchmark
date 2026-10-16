import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

root=Path(__file__).resolve().parents[1]
folder=Path('F:/aicoding/IIA_Data/experiments/source_CNF_full120_v1_launch')
folder.mkdir(exist_ok=False)
receipts=[]
for label,module in [('source','scripts.flow_matching.run_source_cnf_full120_v1'),
    ('replay','scripts.flow_matching.audit_source_cnf_full120_v1')]:
    command=[sys.executable,'-B','-X','utf8','-u','-m',module]
    with (folder/(label+'.stdout.log')).open('xb') as out,(folder/(label+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen(command,cwd=root,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    handle=psutil.Process(child.pid)
    receipts.append(dict(label=label,pid=child.pid,create_time=handle.create_time(),command=command))
(folder/'launch.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print(json.dumps(receipts))
