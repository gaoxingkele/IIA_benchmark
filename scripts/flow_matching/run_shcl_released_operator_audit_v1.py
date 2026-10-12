"""Admit all native SHCL operator profiles under the existing read-only caps."""
import argparse
from datetime import datetime, timezone
import os
import subprocess

from scripts.flow_matching.audit_shcl_released_operators_v1 import native, verify
from scripts.flow_matching.timegan_native_common_v1 import write


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = native.read(native.ROOT / args.config)
    verify(config)
    if (native.ROOT / config['output_root']).exists():
        raise FileExistsError('Preserve every full-profile audit attempt')
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); api['resource_snapshot'] = resource_snapshot
    base = native.ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full released operator audit already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), actual_command=psutil.Process().cmdline(),
        state='starting', active_pid=None, configuration_sha256=native.sha(native.ROOT / args.config))

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)

    try:
        update({})
        with api['resource_slot'](native.ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                'scripts.flow_matching.audit_shcl_released_operators_v1', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=native.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_full_released_operator_audit', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), worker_command=command))
                code, usage = capped_wait(child, config, api, update)
        path = native.ROOT / config['output_root'] / 'full_released_operator_audit.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and path.exists()
        if passed:
            result = native.read(path)
            if not result['passed'] or result['full_operator_profiles'] != 13 or result['optimizer_updates_executed'] != 0:
                raise ValueError('Complete full-profile smoke evidence required')
        write(base / 'receipt.json', dict(exit_code=code, passed=passed, **usage,
            report_sha256=native.sha(path) if path.exists() else None,
            synthetic_operator_only_not_benchmark_performance=True))
        update(dict(state='completed_operator_audit' if passed else 'failed_operator_audit_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
