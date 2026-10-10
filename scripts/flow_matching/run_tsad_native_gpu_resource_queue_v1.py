"""Run unchanged TSAD/industrial GPU commands under the shared native GPU guard."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from scripts.flow_matching.run_light_controller import ROOT,digest,load_controller
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


@contextmanager
def native_slot(api,registration,update):
    settings=registration['settings']
    while True:
        try:gpu=api['gpu_snapshot'](settings)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
            update(dict(state='waiting_gpu_observation_retry',gpu_observation_error=type(error).__name__))
            time.sleep(10);continue
        if gpu['gpu_free_bytes']<settings['minimum_gpu_free_bytes']:
            update(dict(state='waiting_full_gpu_headroom',**gpu));time.sleep(10);continue
        with api['resource_slot'](ROOT/registration['resource_lock'],settings,update):
            try:gpu=api['gpu_snapshot'](settings)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
                update(dict(state='waiting_gpu_observation_retry',gpu_observation_error=type(error).__name__))
            else:
                if gpu['gpu_free_bytes']>=settings['minimum_gpu_free_bytes']:
                    yield;return
        time.sleep(10)


class NativeGPUChild:
    """Complete the original full child under its GPU guard before returning poll."""
    def __init__(self,process,reservation,api,settings,update,receipt,metadata):
        self.process,self.reservation,self.api=process,reservation,api
        self.settings,self.update,self.receipt,self.metadata=settings,update,receipt,metadata
        self.completed=False;self.exit_code=None
    def __getattr__(self,name):return getattr(self.process,name)
    def poll(self):
        if not self.completed:
            code,usage=self.api['guarded_gpu_wait'](self.process,self.settings,self.api,self.update)
            write(self.receipt,dict(self.metadata,exit_code=code,**usage))
            self.reservation.__exit__(None,None,None)
            self.exit_code=code;self.completed=True
            self.update(dict(state='between_full_jobs',active_pid=None,active_job=None,exit_code=code))
            time.sleep(self.settings['controller_yield_seconds'])
        return self.exit_code
    def wait(self,timeout=None):
        if timeout is not None:raise ValueError('Registered original GPU controllers do not impose timed model waits')
        return self.poll()


def guarded_factory(api,registration,update,queue_path,queue,state_root):
    ids={j['id'] for j in queue['jobs']}
    def launch(command,*args,**kwargs):
        command=[str(x) for x in command]
        if '--job-id' not in command or '--queue' not in command:raise ValueError('Original full model command required')
        identity=command[command.index('--job-id')+1]
        supplied=Path(command[command.index('--queue')+1])
        worker=command[command.index('-m')+1]
        if (identity not in ids or (ROOT/supplied).resolve()!=queue_path.resolve()
                or worker!=registration['model_worker']):raise ValueError('Original registered GPU model identity differs')
        receipt=state_root/'receipts'/(identity+'.json')
        if receipt.exists():raise FileExistsError('Prior full model resource receipt remains intact')
        update(dict(state='waiting_native_gpu_slot',requested_job=identity,requested_command=command))
        reservation=native_slot(api,registration,update);reservation.__enter__()
        try:
            process=subprocess.Popen(command,*args,**kwargs);handle=psutil.Process(process.pid)
        except BaseException:
            reservation.__exit__(*sys.exc_info());raise
        metadata=dict(original_job=identity,command=command,pid=process.pid,create_time=handle.create_time(),
            queue_sha256=digest(queue_path),full_original_budget_retained=True)
        update(dict(state='running',active_job=identity,active_pid=process.pid,active_create_time=handle.create_time(),
            actual_command=handle.cmdline()))
        return NativeGPUChild(process,reservation,api,registration['settings'],update,receipt,metadata)
    return launch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    path=ROOT/args.runtime_config;registration=json.loads(path.read_text(encoding='utf-8'))
    for source in registration['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Frozen full GPU adapter source differs')
    original=json.loads((ROOT/registration['original_runtime']).read_text(encoding='utf-8'))
    controller,namespace,binding=load_controller(ROOT,original,registration['controller'])
    queue_path=ROOT/binding['queue_path'];queue=json.loads(queue_path.read_text(encoding='utf-8'))
    if queue['lane']!='gpu':raise ValueError('Only original full CUDA queues are registered')
    if args.probe_only:
        print(json.dumps(dict(controller=registration['controller'],full_jobs=len(queue['jobs']),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            original_phase_gates_retained=True,shared_native_gpu_guard_registered=True)));return
    api=guarded_api();base=ROOT/registration['state_root'];owner=api['exclusive_lock'](base/'adapter.lock')
    if owner is None:raise RuntimeError('This exact original GPU adapter already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),runtime_sha256=digest(path),
        queue_sha256=digest(queue_path),state='waiting_original_phase_gates',active_job=None,active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    update({})
    proxy=SimpleNamespace(**vars(subprocess));proxy.Popen=guarded_factory(api,registration,update,queue_path,queue,base)
    namespace['subprocess']=proxy
    sys.argv=[binding['source_path'],binding['queue_argument'],binding['queue_path']]
    try:
        result=controller.main();update(dict(state='controller_terminal',original_returncode=result,active_pid=None))
    finally:owner.close()


if __name__=='__main__':main()
