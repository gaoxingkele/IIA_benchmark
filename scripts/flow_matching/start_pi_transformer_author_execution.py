"""Launch one hidden pinned Pi CPU worker; do not restart a live worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    runner = ROOT / 'scripts/flow_matching/run_pi_transformer_author_queue.py'
    queue = ROOT / 'configs/experiments/fm_pi_transformer_author_queue.v1.json'
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(str(runner).lower() == a.lower() for a in process.info['cmdline'] or []):
                print(json.dumps({'status': 'already_running', 'pid': process.pid}))
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    settings = json.loads(queue.read_text(encoding='utf-8'))['settings']
    if not (ROOT / settings['python']).is_file():
        raise FileNotFoundError('Registered Pi Python environment missing')
    base = ROOT / settings['output_root']
    base.mkdir(parents=True, exist_ok=True)
    env = {key.upper(): value for key, value in os.environ.items()}
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    with (base / 'console.log').open('a', encoding='utf-8') as out, (base / 'stderr.log').open('a', encoding='utf-8') as err:
        child = subprocess.Popen([str(ROOT / settings['python']), str(runner), '--queue', str(queue)], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags,
                                 start_new_session=sys.platform != 'win32')
    print(json.dumps({'status': 'launched', 'pid': child.pid}))


if __name__ == '__main__':
    main()
