"""Restore pinned full saved-score evaluators, archiving previous resource receipts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from scripts.flow_matching.run_light_controller import ROOT,digest,load_controller
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();path=ROOT/args.config;config=read(path)
    for source in config['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen range restart source differs')
    original=read(ROOT/config['original_runtime']);bindings=[]
    for name in config['controllers']:
        _,_,binding=load_controller(ROOT,original,name)
        settings=read(ROOT/binding['queue_path'])
        for source in settings['source_receipts']:
            if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen complete range evaluator differs')
        bindings.append((binding,settings))
    for process in psutil.process_iter(['pid','name','cmdline']):
        command=process.info['cmdline'] or []
        if process.info['name'].lower()=='python.exe' and '--controller' in command:
            supplied=command[command.index('--controller')+1]
            if supplied in config['controllers']:raise RuntimeError('Corresponding range controller already live: '+str(process.pid))
    if args.probe_only:
        print(json.dumps(dict(controllers=config['controllers'],full_score_and_metric_semantics_unchanged=True,
            torch_imported='torch' in __import__('sys').modules)));return
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'launch_receipt.json').exists():raise FileExistsError('Preserve previous launch receipt')
    owner=resource_api()['exclusive_lock'](base/'registration.lock')
    if owner is None:raise RuntimeError('Corresponding range restore already running')
    observations=[]
    try:
        for binding,settings in bindings:
            name=binding['name'];old_root=ROOT/settings['output_root']
            check=resource_api()['exclusive_lock'](old_root/'evaluation.lock')
            if check is None:raise RuntimeError('Corresponding original evaluation lane remains owned')
            try:
                archive=base/'preserved_original_receipts'/name
                if archive.exists():raise FileExistsError('Previous controller/archive must remain intact')
                archive.mkdir(parents=True);preserved=[]
                candidates=[old_root/'status.json',*sorted((old_root/'logs').glob('*/resource_usage.json'))]
                for source in candidates:
                    if not source.exists():continue
                    target=archive/source.relative_to(old_root);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
                    if digest(source)!=digest(target):raise ValueError('Preserved resource evidence differs')
                    preserved.append(dict(original=source.relative_to(ROOT).as_posix(),archived=target.relative_to(ROOT).as_posix(),sha256=digest(target)))
                write(archive/'manifest.json',dict(queue=binding['queue_path'],queue_sha256=binding['queue_sha256'],preserved=preserved))
            finally:check.close()
            logs=base/'logs';logs.mkdir(exist_ok=True)
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_light_controller',
                '--runtime-config',config['original_runtime'],'--controller',name]
            with (logs/(name+'.stdout.log')).open('xb') as out,(logs/(name+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=psutil.Process(child.pid)
            observations.append(dict(name=name,pid=child.pid,create_time=handle.create_time(),
                command=command,command_token='scripts.flow_matching.run_light_controller',status_path=binding['status_path'],
                queue=binding['queue_path'],queue_sha256=binding['queue_sha256'],prior_receipts_archived=len(preserved)))
            write(base/'launch_receipt.json',dict(config_sha256=digest(path),controllers=observations,
                captured_utc=datetime.now(timezone.utc).isoformat(),benchmark_results_inferred=False))
        print(json.dumps(observations))
    finally:owner.close()


if __name__=='__main__':main()
