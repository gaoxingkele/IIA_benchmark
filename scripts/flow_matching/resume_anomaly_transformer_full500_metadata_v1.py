"""Recover the stopped, never-started full500 review controller after WinError5.

Only status.json publication is changed in this fresh metadata interpreter.
Original numerical modules, runtime, guards, tolerances and children stay pinned.
Permission retries apply to metadata replacement only, never a numerical stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as runner

ROOT = runner.ROOT


def status_writer(target, fallback, *, replace=os.replace, pause=time.sleep,
                  attempts=80, delay_seconds=.1):
    """Bounded retry of an identical metadata rename; all other writes unchanged."""
    target = Path(target).resolve()

    def write(path, value):
        path = Path(path)
        if path.resolve() != target:
            return fallback(path, value)
        content = json.dumps(value, indent=2, allow_nan=False) + '\n'
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.metadata-next')
        with temporary.open('x', encoding='utf-8', newline='') as stream:
            stream.write(content)
        for index in range(attempts):
            try:
                replace(temporary, path)
                return
            except PermissionError as error:
                if getattr(error, 'winerror', None) not in (5, 32, 33) or index + 1 == attempts:
                    raise
                pause(delay_seconds)
        raise AssertionError('Unreachable bounded metadata replacement loop')

    return write


def owner_missing(handle, process_factory=None):
    import psutil
    process_factory = process_factory or psutil.Process
    try:
        return process_factory(handle['pid']).create_time() != handle['create_time']
    except psutil.NoSuchProcess:
        return True


def verify_recovery(config):
    for receipt in config['source_receipts']:
        if runner.sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Recovery input changed: ' + receipt['path'])
    original_path = ROOT / config['original_runtime']
    original = runner.read(original_path)
    if runner.sha(original_path) != config['original_runtime_sha256']:
        raise ValueError('Original frozen runtime changed')
    cases, _ = runner.verify(original)
    if len(cases) != 500 or config['metadata_replace_attempts'] != 80 or config['metadata_replace_delay_seconds'] != .1:
        raise ValueError('Full scope and bounded metadata-only policy required')
    return original, cases


def stopped_before_any_numerical_attempt(config, original):
    launch = runner.read(ROOT / config['prior_launch'])
    status = runner.read(ROOT / config['prior_status'])
    stderr = (ROOT / config['prior_stderr']).read_text(encoding='utf-8')
    handle = config['prior_owner']
    if not owner_missing(handle) or any(launch[k] != handle[k] for k in ('pid', 'create_time')):
        raise ValueError('Prior exact controller is still live or differs')
    if (status['pid'] != handle['pid'] or status['create_time'] != handle['create_time']
            or status['state'] != 'waiting_actual_full_training_completion'
            or status['active_pid'] is not None or status['verified_model_count'] != 0
            or status['failed_preserved'] or status['required_models'] != 500):
        raise ValueError('Only an idle never-started metadata controller may recover')
    if ('PermissionError: [WinError 5]' not in stderr or 'status.next.json' not in stderr
            or 'os.replace(temporary, path)' not in stderr):
        raise ValueError('Exact metadata replacement failure required')
    for directory in (ROOT / original['state_root'] / 'attempts', Path(original['artifact_root'])):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Numerical attempts/artifacts exist; explicit diagnosis required')
    return dict(prior_exact_owner_missing=True, prior_active_child=None,
                numerical_attempts_found=0, numerical_retries_authorized=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = runner.read(ROOT / args.config)
    original, cases = verify_recovery(config)
    evidence = stopped_before_any_numerical_attempt(config, original)
    if args.probe_only:
        print(json.dumps(dict(passed=True, required_models=len(cases), evidence=evidence,
            numerical_workers_started=0, numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules)))
        return
    recovery = ROOT / config['recovery_directory']
    recovery.mkdir(parents=True, exist_ok=False)
    for label, source in [('prior_launch.json', config['prior_launch']),
                          ('prior_status.json', config['prior_status']),
                          ('prior_stderr.log', config['prior_stderr']),
                          ('prior_pending_status.json', config['prior_pending_status'])]:
        (recovery / label).write_bytes((ROOT / source).read_bytes())
    import psutil
    runner.write(recovery / 'recovery_preflight.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        create_time=psutil.Process().create_time(), configuration=args.config,
        configuration_sha256=runner.sha(ROOT / args.config), **evidence,
        original_runtime_sha256=runner.sha(ROOT / config['original_runtime']),
        original_numeric_module=runner.MODULE,
        changed_behavior='bounded status.json rename on Windows sharing/access failures only',
        original_numerical_sources_changed=False, all_original_experiments_complete=False))
    original['_config_path'] = config['original_runtime']
    runner.write = status_writer(ROOT / original['state_root'] / 'status.json', runner.write,
        attempts=config['metadata_replace_attempts'], delay_seconds=config['metadata_replace_delay_seconds'])
    runner.controller(original, cases)


if __name__ == '__main__':
    main()
