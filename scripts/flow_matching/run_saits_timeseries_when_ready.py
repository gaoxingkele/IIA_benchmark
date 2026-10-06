"""Continue original/corrected SAITS time-series jobs after PhysioNet baselines."""
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
    queue_path = ROOT / 'configs/experiments/fm_saits_timeseries_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    report = json.loads((ROOT / 'experiments/runs/flow_matching_campaign/saits_timeseries_cpu_preflight_report.json').read_text(encoding='utf-8'))
    combinations = {(r['dataset'], r['method'], r['variant']) for r in report['records'] if r['finite']}
    expected = {(dataset, method, variant) for dataset, methods in [('air_quality_132', ['saits', 'brits']),
                ('electricity', ['saits', 'brits']), ('ett', ['saits_base'])] for method in methods for variant in ['author', 'corrected']}
    if report['status'] != 'passed' or combinations != expected or not all(report['corrected_dropout_resume_checks'].values()):
        raise ValueError('All real-data author/corrected preflights must pass first')
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    state = base / 'saits_timeseries_continuation_status.json'
    target = str(ROOT / 'scripts/flow_matching/run_queue.py').replace('\\', '/')
    while True:
        pending = check_queue(ROOT / queue['requires_complete_queue'])
        active = any(any(a.replace('\\', '/') == target for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
        state.write_text(json.dumps({'status': 'waiting_saits_physio', 'pid': os.getpid(), 'pending_jobs': len(pending)}), encoding='utf-8')
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
