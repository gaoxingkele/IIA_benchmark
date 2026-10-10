"""Lightweight, guarded queue for all full released original Table2 pipelines."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_spectral_table2_original_v1 import ROOT,read,verify,fingerprint
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    if args.probe_only:
        import psutil,sys
        print(json.dumps(dict(registered_full_jobs=len(queue['jobs']),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,tensorflow_imported='tensorflow' in sys.modules,
            private_bytes=psutil.Process().memory_info().private)));return
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('Table2 controller already live')
    state=dict(pid=os.getpid(),queue_sha256=fingerprint(queue),state='starting',active_pid=None,active_job=None,
        jobs={j['id']:'pending' for j in queue['jobs']})
    def update(values):state.update(values);write(base/'status.json',state)
    environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
        PYTHONUTF8='1',TF_CPP_MIN_LOG_LEVEL='2')
    @contextmanager
    def gpu_slot():
        while True:
            snapshot=api['gpu_snapshot'](queue['settings'])
            if snapshot['gpu_free_bytes']>=queue['settings']['minimum_gpu_free_bytes']:
                with api['resource_slot'](ROOT/queue['resource_lock'],queue['settings'],update):
                    if api['gpu_snapshot'](queue['settings'])['gpu_free_bytes']>=queue['settings']['minimum_gpu_free_bytes']:
                        yield;return
            update(dict(state='waiting_for_gpu_memory'));time.sleep(10)
    def launch(command,identity,settings):
        logs=base/'logs';logs.mkdir(exist_ok=True)
        with (logs/(identity+'.stdout.log')).open('xb') as out,(logs/(identity+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(dict(active_pid=child.pid,active_job=identity,state='running'))
            code,peaks=api['guarded_wait'](child,settings,update)
        write(base/'receipts'/(identity+'.json'),dict(identity=identity,exitcode=code,resource_observations=peaks))
        update(dict(active_pid=None,active_job=None,state='between_jobs'))
        return code
    try:
        for kind in ('cpu','gpu'):
            proof_path=base/'preflight'/(kind+'_v1.json')
            if proof_path.exists():
                proof=read(proof_path)
                if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=fingerprint(queue):
                    raise ValueError('Original full preflight binding differs')
                continue
            if (base/'logs'/('preflight_'+kind+'.stdout.log')).exists():
                raise FileExistsError('Previous failed preflight preserved; requires explicit registered retry')
            command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_table2_original_v1',
                '--queue',args.queue,'--preflight',kind,'--output',proof_path.relative_to(ROOT).as_posix()]
            if kind=='cpu':
                with api['resource_slot'](ROOT/queue['cpu_preflight_lock'],queue['cpu_preflight_settings'],update):
                    code=launch(command,'preflight_'+kind,queue['cpu_preflight_settings'])
            else:
                with gpu_slot():code=launch(command,'preflight_'+kind,queue['settings'])
            if code!=0 or not proof_path.exists():raise RuntimeError('Original '+kind+' preflight failed; preserve diagnostics')
            proof=read(proof_path)
            if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=fingerprint(queue):
                raise ValueError('Full source preflight proof differs')
        def result_valid(job):
            if not (ROOT/job['output_directory']/'result.json').exists():return False
            command=[str(ROOT/queue['python']),'-X','utf8','-m','scripts.flow_matching.run_spectral_table2_original_v1',
                '--queue',args.queue,'--job-id',job['id'],'--verify-result']
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode!=0:write(base/'verification'/(job['id']+'.json'),dict(exitcode=result.returncode,
                stderr=result.stderr.decode('utf-8',errors='replace')[-4000:]))
            return result.returncode==0
        for job in queue['jobs']:
            with gpu_slot():
                if result_valid(job):state['jobs'][job['id']]='completed';continue
                if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                    state['jobs'][job['id']]='failed_or_partial_preserved';continue
                command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_table2_original_v1',
                    '--queue',args.queue,'--job-id',job['id']]
                code=launch(command,job['id'],queue['settings'])
                valid=result_valid(job) if code==0 else False
            state['jobs'][job['id']]='completed' if valid else 'failed_or_partial_preserved'
            update(dict(state='between_jobs'))
        update(dict(state='completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'))
    except BaseException as error:
        update(dict(state='failed_or_partial_preserved',error=repr(error)));raise
    finally:lane.close()


if __name__=='__main__':main()
