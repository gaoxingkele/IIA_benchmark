"""Replace only verified idle controller processes, never an experiment child."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock

ROOT = Path(__file__).resolve().parents[2]


class BusyController(RuntimeError):
    pass


def stop_idle(state_path, expected_script, archive_path):
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if archive_path.exists():
        record = json.loads(archive_path.read_text(encoding='utf-8'))
        if psutil.pid_exists(record['pid']):
            p = psutil.Process(record['pid'])
            assert p.create_time() != record['created_unix'], 'Archived controller still alive'
        return record
    try:
        process = psutil.Process(state['pid'])
    except psutil.NoSuchProcess:
        record = {'pid': state['pid'], 'created_unix': None, 'state_before_transition': state,
                  'state_path': state_path.relative_to(ROOT).as_posix(), 'state_sha256': sha(state_path),
                  'controller_handle_missing_verified': True}
        write_json(archive_path, record)
        return record
    command = process.cmdline()
    assert any(expected_script in a for a in command), command
    if state.get('active_pid') is not None or state.get('active_job') is not None or state.get('active_case') is not None:
        raise BusyController(f'Live experiment must finish: {process.pid}')
    if state.get('state') not in ('waiting_for_memory', 'waiting_for_heavy_lock', 'completed', 'completed_with_unresolved_cases'):
        raise BusyController(f'Controller not at verified idle boundary: {process.pid}')
    process.suspend()
    try:
        children = process.children(recursive=True)
        if not all(any('conhost.exe' in a.lower() for a in c.cmdline()) for c in children):
            raise BusyController('Experiment child exists')
        # Re-read after suspension: a controller cannot launch a child while
        # this evidence is inspected and archived.
        latest = json.loads(state_path.read_text(encoding='utf-8'))
        assert latest['pid'] == process.pid
        if latest.get('active_pid') is not None:
            raise BusyController('Experiment dispatched before suspension')
        record = {'pid': process.pid, 'created_unix': process.create_time(), 'command': command,
                  'children': [{'pid': c.pid, 'command': c.cmdline()} for c in children],
                  'state_before_transition': latest, 'state_path': state_path.relative_to(ROOT).as_posix(),
                  'state_sha256': sha(state_path), 'suspended_and_verified_no_experiment_child': True}
        write_json(archive_path, record)
        process.terminate()
        process.wait(timeout=10)
        return record
    except BaseException:
        if process.is_running():
            process.resume()
        raise


def launch(command, base, log_name):
    base.mkdir(parents=True, exist_ok=True)
    env = {k.upper(): v for k, v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    with (base / (log_name + '.stdout.log')).open('a', encoding='utf-8') as out, (base / (log_name + '.stderr.log')).open('a', encoding='utf-8') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=flags, start_new_session=sys.platform != 'win32')
    return {'launcher_pid': process.pid, 'command': command, 'started_unix': time.time()}


def main():
    archive = ROOT / 'experiments/runs/fm_heavy_scheduler_transition_20261009'
    archive.mkdir(parents=True, exist_ok=True)
    controller_lock = exclusive_lock(archive / 'worker.lock')
    if controller_lock is None:
        raise RuntimeError('Transition already has a live owner')
    if (archive / 'launched.json').exists():
        raise RuntimeError('Transition already launched; inspect live handles instead of restarting')
    old = []
    for name, state, script in [
        ('recovery_v2', 'experiments/runs/fm_resource_recovery_20261009_v2/status.json', 'scripts.flow_matching.run_resource_recovery'),
        ('crossad_v5', 'experiments/runs/fm_crossad_author_v1/status.json', 'run_crossad_case_gated_queue.py'),
        ('preflight_v4', 'experiments/runs/fm_crossad_complete_preflight_v4/status.json', 'scripts.flow_matching.preflight_crossad_complete'),
    ]:
        while True:
            try:
                old.append(stop_idle(ROOT / state, script, archive / (name + '_idle_controller.json')))
                break
            except BusyController as exc:
                write_json(archive / 'status.json', {'pid': os.getpid(), 'state': 'waiting_live_experiment_boundary',
                           'controller': name, 'reason': str(exc), 'old_idle_controllers': old})
                time.sleep(2)
    queue_path = 'configs/experiments/fm_crossad_complete_queue.v6.json'
    queue = json.loads((ROOT / queue_path).read_text(encoding='utf-8'))
    launched = {
        'crossad': launch([str(ROOT / queue['settings']['python']), str(ROOT / 'scripts/flow_matching/run_crossad_case_gated_queue_v2.py'),
                           '--queue', str(ROOT / queue_path)], ROOT / queue['settings']['output_root'], 'scheduler_v2'),
        'recovery': launch([sys.executable, '-u', '-m', 'scripts.flow_matching.run_resource_recovery_v2', '--queue',
                            'configs/experiments/fm_resource_recovery_queue.v3.json'],
                           ROOT / 'experiments/runs/fm_resource_recovery_20261009_v2', 'scheduler_v2'),
        'preflight': launch([sys.executable, '-u', '-m', 'scripts.flow_matching.preflight_crossad_complete_v2', '--queue',
                             'configs/experiments/fm_crossad_complete_queue.v4.json'],
                            ROOT / 'experiments/runs/fm_crossad_complete_preflight_v4', 'scheduler_v2'),
    }
    report = {'old_idle_controllers': old, 'launched': launched,
              'no_experiment_children_stopped': True, 'all_original_jobs_and_limits_preserved': True,
              'all_experiments_complete': False}
    write_json(archive / 'launched.json', launched)
    write_json(archive / 'status.json', {'pid': os.getpid(), 'state': 'completed', 'launched': launched})
    write_json(ROOT / 'docs/reports/fm_heavy_scheduler_transition_2026-10-09.json', report)
    print(json.dumps(launched, indent=2))


if __name__ == '__main__':
    main()
