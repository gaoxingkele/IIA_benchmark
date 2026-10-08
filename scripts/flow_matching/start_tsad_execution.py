"""Launch idempotent hidden CPU/GPU continuations for the frozen TSAD queues."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import write_json


def launch(queue_path, state_root, lane):
    target = str((ROOT / queue_path).resolve()).replace('\\', '/').lower()
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if any('run_tsad_queue.py' in arg for arg in args) and '--queue' in args:
            given = args[args.index('--queue') + 1]
            if str((ROOT / given).resolve()).replace('\\', '/').lower() == target:
                return {'pid': process.pid, 'status': 'already_running'}
    base = ROOT / state_root
    base.mkdir(parents=True, exist_ok=True)
    environment = {key.upper(): value for key, value in os.environ.items()}
    environment['PYTHONUNBUFFERED'] = '1'
    command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_tsad_queue.py'), '--queue', str(ROOT / queue_path)]
    with (base / (lane + '_console.log')).open('a', encoding='utf-8') as stdout, (base / (lane + '_stderr.log')).open('a', encoding='utf-8') as stderr:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 env=environment, creationflags=flags, start_new_session=sys.platform != 'win32')
    return {'pid': child.pid, 'status': 'launched', 'command': command}


def main():
    settings = json.loads((ROOT / 'configs/experiments/fm_tsad_execution.v1.json').read_text(encoding='utf-8'))
    records = {lane: launch(path, settings['output_root'], lane) for lane, path in settings['queue_paths'].items()}
    write_json(ROOT / settings['output_root'] / 'launch.json', records)
    print(json.dumps(records))


if __name__ == '__main__':
    main()
