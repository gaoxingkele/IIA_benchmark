"""Keep all30 full Table2 jobs; admit each method only after its full interface proof."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_spectral_table2_guarded_queue_v2 import complete_resource_api
from scripts.flow_matching.run_spectral_table2_original_v1 import read, verify, fingerprint
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def method_ready(queue, method, registration):
    path = ROOT / registration['method_proofs'][method]
    if not path.exists():
        return False
    proof = read(path)
    if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256'] != fingerprint(queue):
        raise ValueError('Full original method proof binding differs')
    records = proof['native_full_shape_checks']
    wanted = {(name, dataset) for dataset in ('sine', 'stock', 'mujoco')
              for name in (('sdformer_vq', 'sdformer_ar') if method == 'sdformer_ar' else ('spectral_flow',))}
    actual = [r for r in records if (r['method'], r['dataset']) in wanted]
    if len(actual) != len(wanted) or {(r['method'],r['dataset']) for r in actual} != wanted:
        raise ValueError('All full original method/dataset checks are required')
    for record in actual:
        job = next(j for j in queue['jobs'] if j['method']==method and j['dataset']==record['dataset'])
        stage = job['stages'][0 if record['method']!='sdformer_ar' else 1]
        flags = stage['original_arguments']
        size = int(flags[flags.index('--batch-size')+1])
        if (not record['backward_and_optimizer_step'] or not math.isfinite(record['loss'])
                or record['batch_shape'] != [size, *job['data']['shape'][1:]]):
            raise ValueError('Original full model batch/update proof differs')
    loaders=proof['full_original_eight_worker_loaders']
    if len(loaders)!=3 or {r['dataset'] for r in loaders}!={'sine','stock','mujoco'}:
        raise ValueError('All original datasets need full eight-worker loader evidence')
    for record in loaders:
        job=next(j for j in queue['jobs'] if j['dataset']==record['dataset'])
        if (record['num_workers']!=8 or record['drop_last']
                or len(record['worker_exit_codes'])!=8 or any(code!=0 for code in record['worker_exit_codes'])
                or len(set(record['worker_pids']))!=8 or any(pid<=0 for pid in record['worker_pids'])
                or record['full_shape']!=job['data']['shape'] or record['rows']!=job['data']['shape'][0]
                or record['sampler']!='RandomSampler'):
            raise ValueError('Original full loader behavior differs')
    return True


@contextmanager
def complete_native_slot(api, registration, update):
    """Acquire GPU and CPU slots together, releasing both while unable to start."""
    settings=registration['settings']
    def gpu_observation():
        try:
            return api['gpu_snapshot'](settings)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
            update(dict(state='waiting_gpu_observation_retry',gpu_observation_error=type(error).__name__))
            return None
    while True:
        memory=api['resource_snapshot']()
        gpu=gpu_observation()
        if gpu is None:
            time.sleep(10);continue
        gpu_lock=cpu_lock=None
        if api['has_headroom'](memory,settings) and gpu['gpu_free_bytes']>=settings['minimum_gpu_free_bytes']:
            gpu_lock=api['exclusive_lock'](ROOT/registration['resource_lock'])
            if gpu_lock is not None:
                cpu_lock=api['exclusive_lock'](ROOT/registration['cpu_reservation_lock'])
                if cpu_lock is not None:
                    try:
                        memory=api['resource_snapshot']();gpu=gpu_observation()
                        if gpu is not None and api['has_headroom'](memory,settings) and gpu['gpu_free_bytes']>=settings['minimum_gpu_free_bytes']:
                            break
                    except BaseException:
                        cpu_lock.close();gpu_lock.close();raise
                    cpu_lock.close()
                gpu_lock.close()
        update(dict(state='waiting_full_native_resources', shared_lock_owned=False, **memory, **(gpu or {})))
        time.sleep(10)
    try:
        yield
    finally:
        cpu_lock.close();gpu_lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();path=ROOT/args.runtime_config;registration=read(path)
    for source in registration['source_receipts']:
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Full method dispatcher source differs')
    queue=read(ROOT/registration['queue']);verify(queue)
    if digest(ROOT/registration['queue'])!=registration['queue_sha256']:raise ValueError('All30 canonical jobs changed')
    cpu=read(ROOT/registration['cpu_preflight'])
    if not cpu['passed'] or not cpu['diagnostic_only'] or cpu['queue_sha256']!=fingerprint(queue):
        raise ValueError('Complete original data/TensorFlow CPU preflight is required')
    readiness={m:method_ready(queue,m,registration) for m in registration['method_proofs']}
    if args.probe_only:
        print(json.dumps(dict(full_jobs=len(queue['jobs']),method_readiness=readiness,
            resource_admission_bytes=registration['settings']['minimum_commit_headroom_bytes'])));return
    api=complete_resource_api(guarded_api);base=ROOT/registration['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](ROOT/queue['state_root']/'controller.lock')
    if lane is None:raise RuntimeError('Original Table2 pipeline already has a controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_job=None,active_pid=None,
        runtime_sha256=digest(path),queue_sha256=fingerprint(queue),jobs={j['id']:'pending' for j in queue['jobs']})
    def update(values):state.update(values);write(base/'status.json',state)
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
        PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2')
    def command(job,check=False):
        result=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_table2_original_v1',
            '--queue',registration['queue'],'--job-id',job['id']]
        return result+(['--verify-result'] if check else [])
    def full_result(job):
        if not (ROOT/job['output_directory']/'result.json').exists():return False
        child=subprocess.run(command(job,True),cwd=ROOT,env=env,capture_output=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode:raise ValueError('Complete original Table2 artifact audit failed: '+job['id'])
        return True
    try:
        while True:
            waiting=False
            for job in queue['jobs']:
                if state['jobs'][job['id']] in {'completed','failed_or_partial_preserved'}:continue
                if not method_ready(queue,job['method'],registration):
                    state['jobs'][job['id']]='awaiting_full_method_gpu_preflight';waiting=True;continue
                if full_result(job):state['jobs'][job['id']]='completed';continue
                if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                    state['jobs'][job['id']]='failed_or_partial_preserved';continue
                update(dict(state='waiting_full_native_resources',requested_job=job['id'],
                    requested_command=command(job)))
                with complete_native_slot(api,registration,update):
                    logs=base/'logs';logs.mkdir(exist_ok=True)
                    with (logs/(job['id']+'.stdout.log')).open('xb') as out,(logs/(job['id']+'.stderr.log')).open('xb') as err:
                        child=subprocess.Popen(command(job),cwd=ROOT,env=env,stdout=out,stderr=err,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=psutil.Process(child.pid)
                        update(dict(state='running',active_job=job['id'],active_pid=child.pid,
                            active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                        code,usage=api['guarded_gpu_wait'](child,registration['settings'],api,update)
                    valid=full_result(job) if code==0 else False
                    write(base/'receipts'/(job['id']+'.json'),dict(exit_code=code,full_result_validated=valid,**usage))
                state['jobs'][job['id']]='completed' if valid else 'failed_or_partial_preserved'
                update(dict(state='between_jobs',active_job=None,active_pid=None));time.sleep(2)
            if not waiting:
                update(dict(state='completed' if all(s=='completed' for s in state['jobs'].values())
                    else 'completed_with_unresolved_jobs'));break
            update(dict(state='waiting_remaining_full_method_preflight',active_job=None,active_pid=None));time.sleep(30)
    finally:lane.close()


if __name__=='__main__':main()
