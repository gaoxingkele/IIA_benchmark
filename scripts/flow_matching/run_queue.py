"""Resume a frozen single-GPU queue and adopt an already-running first job."""
from __future__ import annotations

import argparse
import hashlib
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


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def find_running(config_path):
    target = (ROOT / config_path).resolve()
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if '--config' not in args or not any(any(name in a for name in ('run_csdi.py', 'run_cfmi.py', 'run_saits.py', 'run_grin.py')) for a in args):
            continue
        index = args.index('--config') + 1
        if index < len(args) and (ROOT / args[index]).resolve() == target:
            return process
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_original_queue.v1.json')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if settings.get('requires_complete_queue') and check_queue(ROOT / settings['requires_complete_queue']):
        raise ValueError('Original phase is incomplete; transfer must wait')
    if settings.get('requires_original_scope_review'):
        campaign = json.loads((ROOT / settings['requires_original_scope_review']).read_text(encoding='utf-8'))
        if campaign['execution'].get('original_scope_status') != 'completed_or_documented_unavailable':
            raise ValueError('Other original-paper experiments remain pending; transfer must wait')
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE / 'queue.lock').open('a+b')
    lock.seek(0)
    lock.write(b'1')
    lock.flush()
    lock.seek(0)
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = []
    state_path = BASE / 'queue_status.json'
    for job in settings['jobs']:
        cfg_path = ROOT / job['config']
        if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError('Queued configuration changed after registration')
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        output = ROOT / cfg['output_root']
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / 'result.json'
        process = find_running(job['config'])
        if not result_path.exists() and process is None:
            command = [str(ROOT / settings['python']), str(ROOT / job['runner']), '--config', str(cfg_path)]
            with (output / 'queue.stdout.log').open('a', encoding='utf-8') as stdout, (output / 'queue.stderr.log').open('a', encoding='utf-8') as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                process = psutil.Process(child.pid)
        while process is not None and process.is_running():
            atomic_json(state_path, {'status': 'running', 'queue_pid': os.getpid(), 'active_job': cfg['id'],
                                    'active_pid': process.pid, 'completed_jobs': len(records), 'total_jobs': len(settings['jobs']),
                                    'records': records})
            time.sleep(10)
            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
        if not result_path.exists():
            records.append({'id': cfg['id'], 'status': 'failed', 'output_root': cfg['output_root']})
            atomic_json(state_path, {'status': 'stopped_on_failure', 'records': records, 'active_job': cfg['id']})
            return 1
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['config_sha256'] != job['config_sha256']:
            raise ValueError('Result configuration differs from frozen job')
        records.append({'id': cfg['id'], 'status': 'completed', 'metrics': result['metrics']})
        subprocess.run([sys.executable, str(ROOT / 'scripts/flow_matching/compare_results.py')], cwd=ROOT, check=True)
        if cfg.get('paper_id') == 'saits':
            comparison = 'compare_saits_timeseries.py' if cfg.get('dataset') else 'compare_saits.py'
            subprocess.run([sys.executable, str(ROOT / 'scripts/flow_matching' / comparison)], cwd=ROOT, check=True)
        if cfg.get('paper_id') == 'grin':
            subprocess.run([sys.executable, str(ROOT / 'scripts/flow_matching/compare_grin.py')], cwd=ROOT, check=True)
        if cfg.get('metric_protocol') == 'transfer_ensemble_v1':
            subprocess.run([sys.executable, str(ROOT / 'scripts/flow_matching/compare_transfer.py')], cwd=ROOT, check=True)
    atomic_json(state_path, {'status': 'completed', 'records': records, 'total_jobs': len(settings['jobs'])})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
