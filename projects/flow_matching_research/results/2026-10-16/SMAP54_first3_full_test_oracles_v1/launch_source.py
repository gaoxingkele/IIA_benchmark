from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_common105_test_label_oracle_v1 as s
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
tests=Path('F:/aicoding/IIA_Data/experiments/common105_all577_test_label_oracle_20261016_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==13
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
c=s.read(s.ROOT/s.CONFIG);s.verify(c);base=s.ROOT/c['state_root'];base.mkdir(exist_ok=False)
command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe','-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
 p=subprocess.Popen(command,cwd=s.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src')),
    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
actor=psutil.Process(p.pid)
exclusive(base/'launch.json',dict(pid=p.pid,create_time=actor.create_time(),command=actor.cmdline(),
    validation_path=str(tests),validation_sha256=s.sha(tests),settings=c['settings'],original_source_and_native_gates_modified=False))
print(p.pid,actor.create_time())
