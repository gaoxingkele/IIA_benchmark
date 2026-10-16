from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_common105_TAB577_fair_entities_v1 as s
tests=Path('F:/aicoding/IIA_Data/experiments/common105_TAB577_fair_entity_continuation_20261016_v2_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==44
assert not any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped'])
c=s.read(s.ROOT/s.CONFIG);s.verify(c)
p=psutil.Process(c['previous_owner']['pid'])
assert p.create_time()==c['previous_owner']['create_time'] and p.cmdline()==c['previous_owner']['command']
base=s.ROOT/c['state_root'];base.mkdir(exist_ok=False)
command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe','-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
 child=subprocess.Popen(command,cwd=s.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src')),
     creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
actor=psutil.Process(child.pid)
s.fair.exclusive(base/'launch.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
    validation_path=str(tests),validation_sha256=s.sha(tests),old_numerical_worker_stopped=False,
    launch_role='metadata_scheduler_waiting_for_proven_terminal_entity_and_unowned_audit_lock'))
print(actor.pid,actor.create_time())
