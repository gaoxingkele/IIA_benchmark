"""Guard real author batches across24 native conditions without optimizer updates.

The full120 training grid is separate and never shortened by these plumbing checks.
"""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v3 as native

MODULE = 'scripts.flow_matching.preflight_otcfm_singlecell_native_v3'


def cases(queue):
    selected = [j for j in queue['jobs'] if j['seed'] == 42]
    expected = {(d, p, m, t) for d in native.contract.DATASETS for p in native.contract.PROTOCOLS
        for m in native.contract.METHODS for t in native.contract.LEAVEOUTS}
    actual = [(j['dataset'], j['protocol'], j['method'], j['leaveout_timepoint']) for j in selected]
    if len(actual) != 24 or set(actual) != expected:
        raise ValueError('Every full-data/native-method/protocol/leftout combination is required')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config)
    queue = config['complete_native_queue']; native.verify(queue); selected = cases(queue)
    if args.worker:
        results = []
        for job in selected:
            results.append(native.worker(queue, job, probe=True)); gc.collect()
        exclusive(ROOT / config['report'], dict(status='passed', actual_native_cases=results,
            complete_registered_training_cases=len(queue['jobs']), model_optimizer_updates=0,
            benchmark_performance_claim=False, all_original_experiments_complete=False,
            queue_candidate=queue, runtime_config_sha256=sha(ROOT / args.config)))
        return
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Native preflight already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    try:
        if (base / 'guard_receipt.json').exists() or (ROOT / config['report']).exists():
            raise FileExistsError('Preserve existing complete or failed preflight evidence')
        with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
            command = [queue['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg', WANDB_MODE='disabled'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                update(dict(state='running_actual_native_preflight', exact_child_handle=handle, active_pid=child.pid))
                code, usage = capped_wait(child, queue, api, update)
            passed = code == 0 and not usage['resource_guard_aborted'] and (ROOT / config['report']).exists()
            exclusive(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, native_report_sha256=sha(ROOT / config['report']) if passed else None))
        update(dict(state='passed' if passed else 'failed_preserved', active_pid=None))
        if not passed: raise RuntimeError('Native preflight failed; preserve logs and diagnose before full training')
    finally:
        owner.close()


if __name__ == '__main__': main()
