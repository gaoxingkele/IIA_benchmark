"""Replace only proven-idle range controllers, preserving all original scores."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_readonly_tab_range_queue_v1 import bind


def verify_idle_identity(name, selected, state, observed, parent_runtime):
    command = observed['command']
    def argument(flag):
        return command[command.index(flag)+1] if flag in command else None
    if (observed['pid'] != selected['old_controller_pid']
            or observed['create_time'] != selected['old_controller_create_time']
            or state.get('pid') != observed['pid']
            or 'scripts.flow_matching.run_light_controller' not in command
            or argument('--runtime-config') != parent_runtime or argument('--controller') != name
            or state.get('config_sha256') != selected['queue_sha256']
            or state.get('state') != 'waiting_for_heavy_lock'
            or state.get('active_pid') is not None or state.get('active_job') is not None
            or any(child['name'].lower() != 'conhost.exe' for child in observed['children'])):
        raise ValueError('Only the exact owned idle controller without a model/evaluator child may be retired')
    return dict(controller=name, original_status=state, actual_process=observed, verified_idle=True)


def observation(process):
    return dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(),
                children=[dict(pid=child.pid, create_time=child.create_time(), name=child.name())
                          for child in process.children(recursive=True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    registration = json.loads((ROOT / args.registration).read_text(encoding='utf-8'))
    for receipt in registration['source_receipts']:
        if digest(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen idle-controller handoff registration changed')
    runtime_path = registration['runtime']
    runtime = json.loads((ROOT / runtime_path).read_text(encoding='utf-8'))
    report = ROOT / registration['report_directory']
    if (report / 'handoff_receipt.json').exists():
        raise FileExistsError('Preserve prior controller handoff receipts')
    verified = []
    for name, selected in runtime['controllers'].items():
        bind(ROOT, runtime, name)
        state = json.loads((ROOT / selected['status_path']).read_text(encoding='utf-8'))
        process = psutil.Process(selected['old_controller_pid'])
        verified.append(verify_idle_identity(name, selected, state, observation(process), runtime['parent_runtime']))
    report.mkdir(parents=True, exist_ok=True)
    for record in verified:
        name = record['controller']
        source = ROOT / runtime['controllers'][name]['status_path']
        target = report / ('preserved_idle_' + name + '_status.json')
        if target.exists():
            raise FileExistsError('Original idle status snapshot already exists')
        shutil.copyfile(source, target)
        record['preserved_status_sha256'] = digest(target)
    for record in verified:
        name = record['controller']
        selected = runtime['controllers'][name]
        process = psutil.Process(selected['old_controller_pid'])
        state = json.loads((ROOT / selected['status_path']).read_text(encoding='utf-8'))
        verify_idle_identity(name, selected, state, observation(process), runtime['parent_runtime'])
        process.terminate()
        # An observation timeout is not proof of termination: no replacement is
        # started unless the original handle actually returns a terminal exit.
        exit_code = process.wait(timeout=10)
        record.update(terminal_exit_code=exit_code, retired_utc=datetime.now(timezone.utc).isoformat())
    launches = []
    for name in runtime['controllers']:
        command = [runtime['python'], '-X', 'utf8', '-u', '-m',
                   'scripts.flow_matching.run_readonly_tab_range_queue_v1',
                   '--runtime-config', runtime_path, '--controller', name]
        with (report / (name + '.controller.stdout.log')).open('xb') as out, \
                (report / (name + '.controller.stderr.log')).open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                         PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process = psutil.Process(child.pid)
            launches.append(dict(controller=name, pid=child.pid, create_time=process.create_time(),
                                 actual_command=process.cmdline(), registered_command=command))
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime_path,
        runtime_sha256=digest(ROOT / runtime_path), retired_controllers=verified, launched_controllers=launches,
        original_full_attempt_slots=runtime['registered_full_attempts'], original_cpu_threads=runtime['cpu_threads'],
        full_scientific_evaluator_and_all_input_configs_unchanged=True, model_training_workers_retired=0,
        new_performance_rows=0, additional_training_seed_slots=0, all_original_experiments_complete=False)
    (report / 'handoff_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                                              encoding='utf-8', newline='\n')
    print(json.dumps(dict(retired_verified_idle_controllers=4, readonly_controllers_started=4,
                          full_original_attempt_slots=runtime['registered_full_attempts'], training_workers_stopped=0)))


if __name__ == '__main__':
    main()
