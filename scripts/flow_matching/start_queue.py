"""Launch the registered queue hidden, with persistent logs and no GUI window."""
from pathlib import Path
import json
import os
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/runs/flow_matching_campaign'


def launch(script, label, arguments=None):
    target = str(ROOT / 'scripts/flow_matching' / script).replace('\\', '/')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if any(arg.replace('\\', '/') == target for arg in (process.info['cmdline'] or [])):
            return {'pid': process.pid, 'status': 'already_running'}
    BASE.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/flow_matching' / script)] + (arguments or [])
    with (BASE / f'{label}_console.log').open('a', encoding='utf-8') as stdout, (BASE / f'{label}_stderr.log').open('a', encoding='utf-8') as stderr:
        environment = {key.upper(): value for key, value in os.environ.items()}
        environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=environment, creationflags=flags, start_new_session=sys.platform != 'win32')
    record = {'pid': process.pid, 'status': 'launched', 'command': command}
    (BASE / f'{label}_process.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    queue = launch('run_queue.py', 'queue', ['--config', str(ROOT / 'configs/experiments/fm_original_queue.v1.json')])
    record = {'original_queue': queue}
    if (BASE / 'air36_cpu_preflight/report.json').exists():
        record['original_continuation'] = launch('run_original_continuation.py', 'original_continuation')
    if (BASE / 'saits_cpu_preflight_report.json').exists():
        record['saits_continuation'] = launch('run_saits_when_ready.py', 'saits_continuation')
    preflight = BASE / 'cfmi_cpu_preflight/report.json'
    if preflight.exists():
        checks = json.loads(preflight.read_text(encoding='utf-8'))['records']
        if {r['dataset'] for r in checks if r['finite']} == {'physio', 'tep', 'skab', 'pronto'}:
            record['transfer_gate'] = launch('run_transfer_when_ready.py', 'transfer_gate')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
