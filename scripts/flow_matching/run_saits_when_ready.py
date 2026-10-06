"""Continue SAITS-paper original PhysioNet experiments after Air-36 finishes."""
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
    queue_path = ROOT / 'configs/experiments/fm_saits_physio_original_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    preflight = json.loads((ROOT / 'experiments/runs/flow_matching_campaign/saits_cpu_preflight_report.json').read_text(encoding='utf-8'))
    if preflight['status'] != 'passed' or {r['model'] for r in preflight['records'] if all(r['checks'].values())} != {'saits', 'brits'}:
        raise ValueError('SAITS preflight incomplete')
    state = ROOT / 'experiments/runs/flow_matching_campaign/saits_continuation_status.json'
    target = str(ROOT / 'scripts/flow_matching/run_queue.py').replace('\\', '/')
    while True:
        pending = check_queue(ROOT / queue['requires_complete_queue'])
        active = any(any(a.replace('\\', '/') == target for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
        state.write_text(json.dumps({'status': 'waiting_air36', 'pid': os.getpid(), 'pending_jobs': len(pending)}), encoding='utf-8')
        if not pending and not active:
            break
        time.sleep(30)
    state.write_text(json.dumps({'status': 'running_saits', 'pid': os.getpid()}), encoding='utf-8')
    completed = subprocess.run([str(ROOT / queue['python']), target, '--config', str(queue_path)], cwd=ROOT,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    state.write_text(json.dumps({'status': 'completed' if completed.returncode == 0 else 'failed', 'pid': os.getpid()}), encoding='utf-8')
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
