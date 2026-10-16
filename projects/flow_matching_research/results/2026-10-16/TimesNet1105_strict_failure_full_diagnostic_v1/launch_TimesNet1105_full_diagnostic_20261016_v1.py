from pathlib import Path
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import diagnose_timesnet1105_full_replay_v1 as s

c=s.read(s.ROOT/s.CONFIG);s.verify(c)
tests=Path('tmp/TimesNet1105_full_replay_diagnostic_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==81
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
saved=Path('F:/aicoding/IIA_Data/experiments/TimesNet1105_full_replay_diagnostic_v1_tests.xml')
assert not saved.exists();shutil.copyfile(tests,saved)
g=s.read(c['preserved_failure']['receipt']);old=g['exact_child_handle']
try:assert psutil.Process(old['pid']).create_time()!=old['create_time']
except psutil.NoSuchProcess:pass
base=s.ROOT/c['state_root'];base.mkdir(exist_ok=False)
command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe','-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
    process=subprocess.Popen(command,cwd=s.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src')),
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
actor=psutil.Process(process.pid)
s.old.exclusive(base/'launch.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
    configuration_sha256=s.sha(s.ROOT/s.CONFIG),validation_path=str(saved),validation_sha256=s.sha(saved),
    original_strict_failure_sha256=c['preserved_failure']['receipt_sha256'],
    role='full51_separate_checkpoint_diagnostic_original_strict_failure_preserved',
    numerical_process_stopped=False))
print(actor.pid,actor.create_time())
