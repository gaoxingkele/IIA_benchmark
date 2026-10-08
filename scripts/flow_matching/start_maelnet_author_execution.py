"""Start one hidden full-author CPU pipeline worker without duplicating it."""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    runner = ROOT / 'scripts/flow_matching/run_maelnet_author_queue.py'
    queue = ROOT / 'configs/experiments/fm_maelnet_author_queue.v1.json'
    for process in psutil.process_iter(['pid', 'cmdline']):
        if any(str(runner).lower() == a.lower() for a in process.info['cmdline'] or []):
            print(json.dumps({'status': 'already_running', 'pid': process.pid}))
            return
    settings = json.loads(queue.read_text(encoding='utf-8'))['settings']
    base = ROOT / settings['output_root']
    base.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    env = {key.upper(): value for key, value in os.environ.items()}
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with (base / 'console.log').open('a', encoding='utf-8') as out, (base / 'stderr.log').open('a', encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable, str(runner), '--queue', str(queue)], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags,
                                 start_new_session=sys.platform != 'win32')
    print(json.dumps({'status': 'launched', 'pid': child.pid, 'jobs': 150, 'author_stages': 600}))


if __name__ == '__main__':
    main()
