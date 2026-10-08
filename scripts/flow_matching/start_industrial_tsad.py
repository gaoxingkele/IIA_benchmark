"""Idempotently launch hidden industrial model and fixed TAB metric workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import write_json
from scripts.flow_matching.run_tsad_experiment import verify_job


def matching_process(root, script, flag, target):
    normalized = str((root/target).resolve()).replace('\\', '/').lower()
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if any(script in a for a in args) and flag in args:
            index = args.index(flag)+1
            if index < len(args) and str((root/args[index]).resolve()).replace('\\', '/').lower() == normalized:
                return process.pid
    return None


def launch(root, script, flag, target, output, name):
    found = matching_process(root, script, flag, target)
    if found:
        return {'pid': found, 'status': 'already_running'}
    base = root/output
    base.mkdir(parents=True, exist_ok=True)
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(root/'scripts/flow_matching'/script), flag, str(root/target)]
    with (base/(name+'_console.log')).open('a', encoding='utf-8') as stdout, (base/(name+'_stderr.log')).open('a', encoding='utf-8') as stderr:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment,
                                 creationflags=flags, start_new_session=sys.platform != 'win32')
    return {'pid': child.pid, 'status': 'launched', 'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    frozen = {}
    for path in settings['queue_paths'].values():
        queue = json.loads((ROOT/path).read_text(encoding='utf-8'))
        for job in queue['jobs']:
            for record in job['frozen_files']:
                if record['path'] in frozen and frozen[record['path']] != record['sha256']:
                    raise ValueError('Inconsistent industrial frozen file identities')
                frozen[record['path']] = record['sha256']
    verify_job(ROOT, {'frozen_files': [{'path': path, 'sha256': digest} for path, digest in frozen.items()]})
    if args.verify_only:
        print(json.dumps({'status': 'all_industrial_frozen_files_verified', 'files': len(frozen)}))
        return
    records = {lane: launch(ROOT, 'run_industrial_tsad_queue.py', '--queue', path, settings['output_root'], lane) for lane, path in settings['queue_paths'].items()}
    records['TAB_ranges'] = launch(ROOT, 'run_tab_ranges_for_config.py', '--config', settings['range_config'], settings['range_output_root'], 'range')
    write_json(ROOT/settings['output_root']/'launch.json', records)
    print(json.dumps(records))


if __name__ == '__main__':
    main()
