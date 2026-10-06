"""Continue validated GRIN jobs after the frozen SAITS time-series queue."""
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
    queue_path = ROOT / 'configs/experiments/fm_grin_original_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    preflight = json.loads((base / 'grin_cpu_preflight_report.json').read_text(encoding='utf-8'))
    if preflight['status'] != 'passed' or not all(preflight['checks'].values()) or set(preflight['datasets']) != {'air36', 'air'}:
        raise ValueError('Both original graph datasets and exact CPU resume must pass first')
    state = base / 'grin_continuation_status.json'
    target = str(ROOT / 'scripts/flow_matching/run_queue.py').replace('\\', '/')
    while True:
        pending = check_queue(ROOT / queue['requires_complete_queue'])
        active = any(any(a.replace('\\', '/') == target for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
        state.write_text(json.dumps({'status': 'waiting_saits_timeseries', 'pid': os.getpid(), 'pending_jobs': len(pending)}), encoding='utf-8')
        if not pending and not active:
            break
        time.sleep(30)
    state.write_text(json.dumps({'status': 'running', 'pid': os.getpid()}), encoding='utf-8')
    result = subprocess.run([str(ROOT / queue['python']), target, '--config', str(queue_path)], cwd=ROOT,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    state.write_text(json.dumps({'status': 'completed' if result.returncode == 0 else 'failed', 'pid': os.getpid()}), encoding='utf-8')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
