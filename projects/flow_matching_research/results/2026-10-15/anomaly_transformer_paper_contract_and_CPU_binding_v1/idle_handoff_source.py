"""Replace only exact idle metadata owners; never stop numerical children."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT,read,sha,write
from scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3 import verify

configpath='configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v3.json'
config=read(ROOT/configpath);verify(config)
records=[]
cases=[('author',26492,1792040566.1641924,'scripts.flow_matching.run_anomaly_transformer_author_full_v1'),
       ('restored',30512,1792040981.7732399,'scripts.flow_matching.run_anomaly_transformer_swat_restored_v1')]
for branch,pid,ct,module in cases:
    selected=read(ROOT/config['cases'][branch]);statepath=ROOT/selected['state_root']/'status.json'
    process=psutil.Process(pid)
    def inspect_idle():
        if abs(process.create_time()-ct)>.001:raise RuntimeError('Original metadata PID was reused')
        command=process.cmdline()
        if command[1:]!=['-B','-X','utf8','-u','-m',module,'--config',config['cases'][branch]]:
            raise RuntimeError('Original owner command changed')
        state=read(statepath)
        if state.get('pid')!=pid or state.get('active_pid') is not None or state.get('state') not in ['waiting_for_memory','waiting_for_heavy_lock']:
            raise RuntimeError('Original controller is not idle')
        children=[]
        for child in process.children(recursive=True):
            entry=dict(pid=child.pid,name=child.name(),command=child.cmdline(),create_time=child.create_time())
            if child.name().lower()!='conhost.exe':raise RuntimeError('A live numerical/unknown child prevents handoff')
            children.append(entry)
        return dict(pid=pid,create_time=ct,command=command,state=state,only_console_host_children=children)
    before=inspect_idle();process.suspend()
    try:
        suspended=inspect_idle()
        process.terminate();process.wait(timeout=15)
    except BaseException:
        if process.is_running():process.resume()
        raise
    base=ROOT/config['state_root']/branch;base.mkdir(parents=True,exist_ok=True)
    command=[config['python'],'-B','-X','utf8','-u','-m',
             'scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3',
             '--config',configpath,'--branch',branch]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'))
    with (base/'stdout.log').open('xb') as stdout,(base/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    newprocess=psutil.Process(child.pid);newct=newprocess.create_time()
    deadline=time.monotonic()+30
    while True:
        if child.poll() is not None:raise RuntimeError('New CPU binding exited; inspect preserved logs')
        current=read(statepath)
        if current.get('pid')==child.pid:break
        if time.monotonic()>deadline:raise TimeoutError('New state observation timed out; do not restart')
        time.sleep(.25)
    record=dict(branch=branch,retired_exact_idle_metadata_owner=before,suspended_recheck=suspended,
                new_pid=child.pid,new_create_time=newct,new_command=newprocess.cmdline(),new_state=current,
                numerical_workers_stopped=0,model_hyperparameters_or_guards_changed=False)
    records.append(record)
    write(base/'idle_handoff_receipt.json',record)
write(ROOT/config['state_root']/'idle_handoff_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    configuration=configpath,configuration_sha256=sha(ROOT/configpath),records=records,
    numerical_workers_stopped=0,all_original_experiments_complete=False))
print(json.dumps(dict(exact_idle_owners_replaced=2,records=[dict(branch=r['branch'],pid=r['new_pid'],
    create_time=r['new_create_time'],state=r['new_state']['state']) for r in records],numerical_workers_stopped=0)))
