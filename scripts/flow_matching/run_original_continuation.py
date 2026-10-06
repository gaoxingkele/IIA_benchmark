"""Continue with the registered Air-36 author experiments after PhysioNet."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.phase_gate import check_queue


def main():
    queue_path = ROOT / 'configs/experiments/fm_air36_original_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    state = ROOT / 'experiments/runs/flow_matching_campaign/original_continuation_status.json'
    target = str(ROOT / 'scripts/flow_matching/run_queue.py').replace('\\', '/')
    while True:
        pending = check_queue(ROOT / queue['requires_complete_queue'])
        state.write_text(json.dumps({'status': 'waiting_physionet', 'pid': os.getpid(), 'pending_jobs': len(pending)}), encoding='utf-8')
        active = any(any(a.replace('\\', '/') == target for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
        if not pending and not active:
            break
        time.sleep(30)
    state.write_text(json.dumps({'status': 'running_air36', 'pid': os.getpid()}), encoding='utf-8')
    process = subprocess.run([str(ROOT / queue['python']), target, '--config', str(queue_path)], cwd=ROOT,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    state.write_text(json.dumps({'status': 'completed' if process.returncode == 0 else 'failed', 'pid': os.getpid()}), encoding='utf-8')
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
