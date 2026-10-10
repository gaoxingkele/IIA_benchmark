"""Resume all original SAITS/BRITS jobs with the shared native GPU resource slot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.phase_gate import check_queue
from scripts.flow_matching.run_spectral_table2_guarded_queue_v3 import complete_resource_api
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_binding(runtime):
    for entry in runtime['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen original SAITS runtime differs: ' + entry['path'])
    path = ROOT / runtime['queue']; queue = read(path)
    if digest(path) != runtime['queue_sha256']:
        raise ValueError('Full original SAITS queue differs')
    for job in queue['jobs']:
        path = ROOT / job['config']
        if digest(path) != job['config_sha256'] or job['runner'] != 'scripts/flow_matching/run_saits.py':
            raise ValueError('Full original SAITS model command differs')
        config = read(path)
        if config.get('diagnostic') or config.get('limit_test_cases') or config['device'] != 'cuda':
            raise ValueError('Full original CUDA job required')
    if queue.get('requires_complete_queue') and check_queue(ROOT / queue['requires_complete_queue']):
        raise ValueError('Original PhysioNet prerequisite remains incomplete')
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True); parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); path = ROOT / args.runtime_config; runtime = read(path)
    queue = verify_binding(runtime)
    if args.probe_only:
        print(json.dumps(dict(original_jobs=len(queue['jobs']),
            missing_full_results=len(check_queue(ROOT/runtime['queue'])),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    api = resource_api(); base = ROOT/runtime['state_root']; base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](ROOT/runtime['original_queue_lock'])
    if lane is None:
        raise RuntimeError('Original imputation GPU lane still owned by another live queue')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        runtime_sha256=digest(path), queue_sha256=runtime['queue_sha256'], state='starting',
        active_pid=None, active_job=None, jobs={})
    def update(values):
        state.update(values); write(base/'status.json', state)
    environment = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    def full_result(job):
        config = read(ROOT/job['config']); output = ROOT/config['output_root']
        if not (output/'result.json').exists():
            return False
        command = [str(ROOT/queue['python']), '-X', 'utf8', '-m',
            'scripts.flow_matching.validate_saits_full_result_v1', '--config', job['config']]
        check = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        receipt = dict(exit_code=check.returncode, stdout=check.stdout.decode('utf-8', errors='replace'),
            stderr=check.stderr.decode('utf-8', errors='replace'))
        write(base/'full_result_audits'/(config['id']+'.json'), receipt)
        if check.returncode:
            raise ValueError('Existing full result failed original checkpoint/data audit: '+config['id'])
        return True
    try:
        for job in queue['jobs']:
            config = read(ROOT/job['config']); identity = config['id']; output = ROOT/config['output_root']
            if full_result(job):
                state['jobs'][identity]='completed'; update({}); continue
            # Do not duplicate an independently running native model.
            for process in psutil.process_iter(['pid', 'cmdline']):
                command = process.info['cmdline'] or []
                if '--config' in command and any('run_saits.py' in a for a in command):
                    supplied = command[command.index('--config')+1]
                    if (ROOT/supplied).resolve() == (ROOT/job['config']).resolve():
                        raise RuntimeError('Corresponding original SAITS model already live: '+str(process.pid))
            with api['resource_slot'](ROOT/runtime['resource_lock'], runtime['settings'], update):
                # The shared slot serializes SAITS against all native GPU pipelines.
                gpu_api = complete_resource_api(guarded_api); gpu = gpu_api['gpu_snapshot'](runtime['settings'])
                if gpu['gpu_free_bytes'] < runtime['settings']['minimum_gpu_free_bytes']:
                    raise RuntimeError('Insufficient registered original GPU headroom')
                output.mkdir(parents=True, exist_ok=True)
                archive = base/'preserved_resume_inputs'/identity
                if archive.exists():
                    raise FileExistsError('Preserve previous native resume attempt')
                archive.mkdir(parents=True)
                preserved=[]
                for name in ('resume.pt', 'best.pt', 'failure.json', 'training.log', 'queue.stdout.log', 'queue.stderr.log'):
                    source=output/name
                    if source.exists():
                        target=archive/name; shutil.copy2(source,target)
                        if digest(source)!=digest(target):raise ValueError('Epoch checkpoint archive differs')
                        preserved.append(dict(path=str(source), archived=str(target), sha256=digest(target)))
                write(archive/'manifest.json',dict(original_config_sha256=job['config_sha256'], preserved=preserved,
                    native_epoch_boundary_resume=True, created_utc=datetime.now(timezone.utc).isoformat()))
                command=[str(ROOT/queue['python']),'-X','utf8','-u',str(ROOT/job['runner']),
                    '--config',str(ROOT/job['config'])]
                logs=base/'logs'; logs.mkdir(exist_ok=True)
                with (logs/(identity+'.stdout.log')).open('xb') as out,(logs/(identity+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    handle=psutil.Process(child.pid)
                    update(dict(state='running',active_job=identity,active_pid=child.pid,
                        active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                    code,usage=gpu_api['guarded_gpu_wait'](child,runtime['settings'],gpu_api,update)
                valid=full_result(job) if code==0 else False
                write(base/'receipts'/(identity+'.json'),dict(exit_code=code,full_result_validated=valid,**usage))
            state['jobs'][identity]='completed' if valid else 'failed_or_partial_preserved'
            update(dict(active_job=None,active_pid=None,state='between_jobs'))
            if not valid:
                # Checkpoints and original failure remain available for native resume.
                raise RuntimeError('Full original SAITS job failed; preserve its exact epoch state')
            time.sleep(runtime['settings'].get('controller_yield_seconds',2))
        update(dict(state='completed'))
    finally:
        lane.close()


if __name__=='__main__':main()
