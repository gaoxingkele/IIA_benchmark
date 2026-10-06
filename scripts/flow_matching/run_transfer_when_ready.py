"""Wait without occupying the GPU; run transfer after the first original queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.flow_matching.phase_gate import check_queue

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/runs/flow_matching_campaign'


def main():
    queue_path = ROOT / 'configs/experiments/fm_transfer_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    while True:
        pending = check_queue(ROOT / queue['requires_complete_queue'])
        campaign = json.loads((ROOT / queue['requires_original_scope_review']).read_text(encoding='utf-8'))
        scope_ready = campaign['execution'].get('original_scope_status') == 'completed_or_documented_unavailable'
        state = {'pid': os.getpid(), 'status': 'waiting_original_phase' if pending or not scope_ready else 'ready',
                 'pending_original_jobs': len(pending), 'original_scope_review_complete': scope_ready}
        (BASE / 'transfer_gate_status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        if not pending and scope_ready:
            active = any(any('run_queue.py' in arg for arg in (p.info['cmdline'] or [])) for p in psutil.process_iter(['cmdline']))
            if not active:
                break
        time.sleep(30)
    result = subprocess.run([str(ROOT / queue['python']), str(ROOT / 'scripts/flow_matching/run_queue.py'), '--config', str(queue_path)], cwd=ROOT,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    (BASE / 'transfer_gate_status.json').write_text(json.dumps({'pid': os.getpid(), 'status': 'completed' if result.returncode == 0 else 'failed', 'returncode': result.returncode}, indent=2), encoding='utf-8')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
