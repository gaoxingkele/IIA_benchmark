from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import psutil

from scripts.flow_matching import run_source_cnf_full120_v1 as old
from scripts.flow_matching import run_source_cnf_full120_v2 as new
from scripts.flow_matching import audit_source_cnf_full120_v1 as old_audit
from scripts.flow_matching import audit_source_cnf_full120_v2 as new_audit
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,write
from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades

q=read(ROOT/old.QUEUE);c=read(ROOT/old_audit.CONFIG)
new.verify(read(ROOT/new.QUEUE));new_audit.verify(read(ROOT/new_audit.CONFIG))
api=install_facades();locks=[]
folder=Path('F:/aicoding/IIA_Data/experiments/source_CNF_full120_v2b_launch');folder.mkdir(exist_ok=False)
try:
    for path in [q['resource_lock'],c['resource_lock']]:
        lock=api['exclusive_lock'](ROOT/path)
        if lock is None:raise RuntimeError('Shared numerical resource is owned;do not retire any process')
        locks.append(lock)
    owners=read(Path('F:/aicoding/IIA_Data/experiments/source_CNF_full120_v1_launch/launch.json'))
    records=[];handles=[]
    for owner,config in zip(owners,[q,c]):
        p=psutil.Process(owner['pid']);state_path=ROOT/config['state_root']/'status.json';state=read(state_path)
        assert p.create_time()==owner['create_time'] and p.cmdline()==owner['command'] and p.is_running()
        assert state['pid']==p.pid and state['active_pid'] is None
        children=p.children(recursive=True)
        assert all(ch.name().lower()=='conhost.exe' and ch.exe().lower()=='c:\windows\system32\conhost.exe' and ch.cmdline()==[r'\??\C:\WINDOWS\system32\conhost.exe','0x4'] for ch in children)
        assert state.get('completed_ids',state.get('verified_ids'))==[] and not state['failed_preserved']
        assert p.memory_info().private<64*2**20
        assert not (ROOT/config['state_root']/'attempts').exists()
        if owner['label']=='source':
            assert not (ROOT/q['state_root']/'native_framework_lifecycle').exists()
            assert not Path(q['lifecycle_artifact_root']).exists()
        records.append(dict(owner=owner,actual_state=state,state_path=str(state_path),
            private_bytes=p.memory_info().private,numerical_descendants=[],allowed_native_console_descendants=[dict(pid=ch.pid,create_time=ch.create_time(),exe=ch.exe(),command=ch.cmdline()) for ch in children],shared_numerical_lock_unowned=True))
        handles.append(p)
    write(folder/'verified_metadata_only_predecessors.json',dict(records=records,
        captured_utc=datetime.now(timezone.utc).isoformat(),numerical_workers_stopped=False))
    for p,record in zip(handles,records):
        assert p.create_time()==record['owner']['create_time']
        assert [ch.pid for ch in p.children(recursive=True)]==[ch['pid'] for ch in record['allowed_native_console_descendants']]
        p.terminate();p.wait(timeout=10)
        write(Path(record['state_path']),dict(record['actual_state'],state='retired_verified_metadata_only_for_accurate_CNF_validation_v2',
            successor_configuration=new.QUEUE if record['owner']['label']=='source' else new_audit.CONFIG,
            numerical_workers_stopped=False))
finally:
    for lock in reversed(locks):lock.close()
launch=[]
for label,module in [('source',new.MODULE),('replay',new_audit.MODULE)]:
    command=[sys.executable,'-B','-X','utf8','-u','-m',module]
    with (folder/(label+'.stdout.log')).open('xb') as out,(folder/(label+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    launch.append(dict(label=label,pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command))
write(folder/'launch.json',launch)
print(__import__('json').dumps(launch))
