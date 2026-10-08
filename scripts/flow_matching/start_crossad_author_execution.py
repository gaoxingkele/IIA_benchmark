"""Idempotent hidden launch, requiring all native checkpoints to pass preflight."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha


def main():
    runner = ROOT/'scripts/flow_matching/run_crossad_author_queue.py'
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(str(runner).lower() == arg.lower() for arg in process.info['cmdline'] or []):
                print(json.dumps({'status': 'already_running', 'pid': process.pid})); return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    queue_path = ROOT/'configs/experiments/fm_crossad_author_queue.v1.json'
    queue = json.loads(queue_path.read_text()); settings = queue['settings']
    preflight = json.loads((ROOT/'docs/reports/fm_crossad_author_preflight_2026-10-09.json').read_text())
    assert preflight['queue_sha256'] == sha(queue_path) and len(preflight['records']) == 7
    base = ROOT/settings['output_root']; base.mkdir(parents=True, exist_ok=True)
    env = {k.upper(): v for k, v in os.environ.items()}; env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    with (base/'console.log').open('a', encoding='utf-8') as out, (base/'stderr.log').open('a', encoding='utf-8') as err:
        child = subprocess.Popen([str(ROOT/settings['python']), str(runner), '--queue', str(queue_path)], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags, start_new_session=sys.platform != 'win32')
    print(json.dumps({'status': 'launched', 'pid': child.pid, 'registered_jobs': len(queue['jobs'])}))


if __name__ == '__main__':
    main()
