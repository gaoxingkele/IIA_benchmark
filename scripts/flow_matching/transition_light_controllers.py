"""Replace only verified idle controllers; never interrupt their model children."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def idle_observation(binding):
    path = ROOT / binding['status_path']
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding='utf-8'))
    try:
        process = psutil.Process(state['pid'])
        command = process.cmdline()
        if not any(binding['old_worker_token'] in a for a in command):
            return None
        if process.children(recursive=True) or state.get('active_pid') or state.get('active_job') or state.get('active_case'):
            return None
        if state.get('state') not in ('waiting_for_memory', 'waiting_for_heavy_lock', 'waiting_for_remaining_native_preflight'):
            return None
        return process, state, command, process.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    path = ROOT / args.runtime_config
    config = json.loads(path.read_text(encoding='utf-8'))
    report_path = ROOT / args.report
    if report_path.exists():
        raise ValueError('Transition receipt already exists; inspect live state instead of repeating')
    base = ROOT / config['transition_archive']
    base.mkdir(parents=True, exist_ok=True)
    records = []
    for binding in config['controllers']:
        load_controller(ROOT, config, binding['name'])  # Full source/queue receipt validation first.
        observation = idle_observation(binding)
        if observation is None:
            records.append({'controller':binding['name'], 'status':'deferred_active_or_unverified'})
            continue
        process, state, command, created = observation
        process.suspend()
        try:
            again = idle_observation(binding)
            if again is None or again[0].pid != process.pid or again[3] != created:
                records.append({'controller':binding['name'], 'status':'deferred_race_revalidated'})
                continue
            saved = {'controller':binding['name'], 'old_pid':process.pid, 'old_create_time':created,
                     'command':command, 'state':state, 'private_bytes':getattr(process.memory_info(),'private',0),
                     'no_model_children_verified_before_stop':True, 'runtime_config_sha256':digest(path)}
            archive = base / (binding['name']+'_idle_controller.json')
            archive.write_text(json.dumps(saved, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            process.terminate()
            process.wait(timeout=10)
        finally:
            if process.is_running():
                process.resume()
        if process.is_running():
            raise RuntimeError('Old controller remains live; replacement not launched')
        logs = base / (binding['name']+'_new')
        logs.mkdir(exist_ok=True)
        new_command = [sys.executable, '-u', '-m', 'scripts.flow_matching.run_light_controller',
                       '--runtime-config', args.runtime_config, '--controller', binding['name']]
        env = {k.upper():v for k,v in os.environ.items()}
        env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if sys.platform=='win32' else 0
        with (logs/'stdout.log').open('a', encoding='utf-8') as out, (logs/'stderr.log').open('a', encoding='utf-8') as err:
            child = subprocess.Popen(new_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, creationflags=flags)
        records.append(dict(saved, status='replaced_verified_idle_controller', new_pid=child.pid,
                            new_command=new_command, started_unix=time.time()))
    report = {'runtime_config':args.runtime_config, 'runtime_config_sha256':digest(path), 'records':records,
              'model_training_processes_stopped':False, 'experiment_job_dictionaries_changed':False,
              'budgets_seeds_metric_formulas_and_emergency_guard_changed':False, 'all_experiments_complete':False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'replaced':sum(r['status']=='replaced_verified_idle_controller' for r in records),
                      'deferred':sum(r['status'].startswith('deferred') for r in records)}, indent=2))


if __name__ == '__main__':
    main()
