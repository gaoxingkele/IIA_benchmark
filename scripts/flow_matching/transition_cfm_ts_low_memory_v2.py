"""Replace only the observed idle CFM-TS parent; preserve all model processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT,read,sha,write,binding


def idle_observation(runtime):
    old=runtime['predecessor'];state=read(ROOT/old['status_path'])
    try:
        process=psutil.Process(old['pid'])
        if process.create_time()!=old['create_time'] or process.cmdline()!=old['command']:
            return None
        if state['pid']!=process.pid or state.get('active_pid') or state.get('active_job'):
            return None
        if state.get('state') not in ('waiting_for_memory','between_jobs') or process.children(recursive=True):
            return None
        return process,state
    except (psutil.NoSuchProcess,psutil.AccessDenied):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime-config',required=True)
    args=parser.parse_args();runtime=read(ROOT/args.runtime_config)
    for name in runtime['controllers']:binding(runtime,name)
    base=ROOT/runtime['handoff_root'];base.mkdir(parents=True,exist_ok=True)
    report=base/'transition.json'
    if report.exists():raise FileExistsError('Preserve observed transition')
    observed=idle_observation(runtime)
    if observed is None:raise RuntimeError('Predecessor active or identity changed; defer, do not stop a model')
    process,state=observed;before=process.memory_info().private
    process.suspend()
    try:
        observed=idle_observation(runtime)
        if observed is None:raise RuntimeError('Idle boundary changed while suspending; resume predecessor')
        write(base/'old_controller.json',{'state':state,'predecessor':runtime['predecessor'],
            'private_bytes':before,'no_model_descendants_verified':True})
        process.terminate();process.wait(timeout=10)
    finally:
        if process.is_running():process.resume()
    launches=[]
    for name in ['original','recovery']:
        command=[sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
            '--runtime-config',args.runtime_config,'--controller',name]
        with (base/f'{name}.stdout.log').open('xb') as out,(base/f'{name}.stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            actual=psutil.Process(child.pid)
            launches.append({'controller':name,'pid':child.pid,'create_time':actual.create_time(),'command':actual.cmdline()})
    write(report,{'runtime_config':args.runtime_config,'runtime_sha256':sha(ROOT/args.runtime_config),
        'predecessor':runtime['predecessor'],'old_private_bytes':before,'launched':launches,
        'model_training_processes_stopped':False,'scientific_protocol_changed':False})
    print(json.dumps({'launched':launches,'old_private_bytes':before}))


if __name__=='__main__':main()
