from pathlib import Path
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_spectral70_queued_handoff_v2 as s

config=s.read(s.ROOT/s.CONFIG)
c,parent,cases=s.verify(config)
tests=Path('tmp/Spectral70_queued_handoff_20261016_v2_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==23
assert not any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped'])
frozen=Path('F:/aicoding/IIA_Data/experiments/Spectral70_queued_handoff_20261016_v2_tests.xml')
assert not frozen.exists()
shutil.copyfile(tests,frozen)
old=s.exact_process(config['previous_owner'])
observer=s.exact_process(config['previous_observer'])
state=s.read(s.ROOT/c['state_root']/'status.json')
handle=state['exact_child_handle']
child=psutil.Process(handle['pid'])
assert child.create_time()==handle['create_time'] and child.cmdline()==handle['command']
assert state['active_condition']=='spectral_original__diffusion_ts__energy__released_author_data__seed43'
base=s.ROOT/config['state_root'];base.mkdir(exist_ok=False)
command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe',
    '-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
    process=subprocess.Popen(command,cwd=s.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src')),
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
actor=psutil.Process(process.pid)
s.previous.fair.exclusive(base/'launch.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
    validation_path=str(frozen),validation_sha256=s.sha(frozen),configuration_sha256=s.sha(s.ROOT/s.CONFIG),
    original_numeric_handle=handle,original_numeric_condition=state['active_condition'],
    previous_owner=config['previous_owner'],previous_observer=config['previous_observer'],
    role='queued_actual_GPU_lock_then_twice_verified_idle_metadata_handoff',numerical_worker_stopped=False))
time.sleep(1)
assert process.poll() is None
assert not (base/'controller.stderr.log').read_bytes()
observed=s.read(base/'status.json')
assert observed['state']=='waiting_pending_Windows_GPU_lock_no_process_stopped'
assert not observed['kernel_lock_owned'] and not (base/'handoff.json').exists()
print(actor.pid,actor.create_time(),observed['state'])
