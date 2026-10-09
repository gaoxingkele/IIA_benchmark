"""One guarded GPU worker for full GRASP entity training and sensitivities."""
import argparse
import json
import os
import subprocess
import sys
import time
import psutil

from scripts.flow_matching.grasp_protocol_runner import ROOT, complete, sha, verify, write_json
from scripts.flow_matching.run_light_controller import load_controller


def gpu_snapshot(settings):
    text = subprocess.check_output(['nvidia-smi','--id='+str(settings['gpu_index']),
                 '--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    index, uuid, free = [part.strip() for part in text.strip().split(',')]
    if int(index) != settings['gpu_index'] or uuid != settings['gpu_uuid']:
        raise RuntimeError('Registered GPU identity differs')
    return {'gpu_index':int(index),'gpu_uuid':uuid,'gpu_free_bytes':int(free)*1024**2}


def guarded_gpu_wait(child, settings, api, update):
    peaks = {'peak_tree_rss_bytes':0,'peak_tree_private_bytes':0,
             'minimum_commit_available_bytes':None,'minimum_gpu_free_bytes':None,
             'resource_guard_aborted':False,'gpu_observation_retries':0}
    while child.poll() is None:
        memory = api['resource_snapshot']()
        try:
            gpu = gpu_snapshot(settings)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
            gpu = None
            peaks['gpu_observation_retries'] += 1
            update({'gpu_observation_error':type(error).__name__,'state':'running_gpu_observation_retry'})
        except RuntimeError:
            api['terminate_tree'](child)
            peaks.update(resource_guard_aborted=True,gpu_identity_changed=True)
            update({**peaks,'state':'resource_guard_aborted'})
            break
        observations = [('minimum_commit_available_bytes',memory['commit_available_bytes'])]
        if gpu is not None:
            observations.append(('minimum_gpu_free_bytes',gpu['gpu_free_bytes']))
        for key, value in observations:
            peaks[key] = value if peaks[key] is None else min(value,peaks[key])
        try:
            parent = psutil.Process(child.pid)
            values = [p.memory_info() for p in [parent,*parent.children(recursive=True)]]
            peaks['peak_tree_rss_bytes'] = max(peaks['peak_tree_rss_bytes'],sum(v.rss for v in values))
            peaks['peak_tree_private_bytes'] = max(peaks['peak_tree_private_bytes'],sum(getattr(v,'private',v.vms) for v in values))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
        if (memory['commit_available_bytes']<settings['emergency_commit_headroom_bytes']
                or gpu is not None and gpu['gpu_free_bytes']<settings['emergency_gpu_free_bytes']):
            peaks.update(resource_guard_aborted=True,abort_resource_snapshot={**memory,**(gpu or {})})
            api['terminate_tree'](child)
            update({**peaks,'state':'resource_guard_aborted'})
            break
        update({**memory,**(gpu or {}),**peaks,'state':'running' if gpu else 'running_gpu_observation_retry'})
        time.sleep(2)
    return child.wait(), peaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default='configs/experiments/fm_grasp_protocol_queue.v2.json')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.queue
    queue = json.loads(path.read_text(encoding='utf-8'))
    verify(queue)
    if args.verify_only:
        print(json.dumps({'verified_jobs':len(queue['jobs']),'cohorts':queue['cohort_count'],'queue_sha256':sha(path)}))
        return
    runtime = json.loads((ROOT / queue['runtime_config']).read_text(encoding='utf-8'))
    _, api, _ = load_controller(ROOT, runtime, 'crossad')
    base = ROOT / queue['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base / 'worker.lock')
    if lane is None:
        raise RuntimeError('Full GRASP GPU queue already owned')
    state = {'pid':os.getpid(),'queue_sha256':sha(path),'jobs':{},'active_pid':None}
    def update(values):
        state.update(values)
        write_json(base / 'status.json', state)
    for job in queue['jobs']:
        if complete(job):
            state['jobs'][job['id']] = 'completed'
            continue
        output = ROOT / job['output_directory']
        if output.exists():
            state['jobs'][job['id']] = 'failed_or_partial_preserved'
            continue
        with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
            gpu = gpu_snapshot(queue['settings'])
            while gpu['gpu_free_bytes']<queue['settings']['minimum_gpu_free_bytes']:
                update({**gpu,'state':'waiting_for_gpu_memory','active_job':None,'active_pid':None})
                time.sleep(10)
                gpu = gpu_snapshot(queue['settings'])
            # Recheck host resources after any GPU-memory wait.
            api['wait_for_headroom'](queue['settings'],update)
            logs = base / 'logs'
            logs.mkdir(exist_ok=True)
            environment = {k.upper():v for k,v in os.environ.items()}
            environment.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
                               OMP_NUM_THREADS=str(queue['settings']['cpu_threads']),
                               MKL_NUM_THREADS=str(queue['settings']['cpu_threads']),
                               CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES=str(queue['settings']['gpu_index']))
            command = [queue['python'],'-u','-m','scripts.flow_matching.grasp_protocol_runner',
                       '--queue',str(path),'--job-id',job['id']]
            with (logs/(job['id']+'.stdout.log')).open('a',encoding='utf-8') as out, (logs/(job['id']+'.stderr.log')).open('a',encoding='utf-8') as err:
                child = subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                update({'state':'running','active_job':job['id'],'active_pid':child.pid})
                code, usage = guarded_gpu_wait(child,queue['settings'],api,update)
            done = complete(job) if not code else False
            if output.exists():
                write_json(output/'resource_usage.json',usage)
                if not done:
                    write_json(output/'failure.json',{'exit_code':code,'resource_usage':usage})
            state['jobs'][job['id']] = 'completed' if done else 'failed_or_partial_preserved'
            update({'active_pid':None,'active_job':None})
    update({'state':'completed' if all(v=='completed' for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'})


if __name__ == '__main__':
    main()
