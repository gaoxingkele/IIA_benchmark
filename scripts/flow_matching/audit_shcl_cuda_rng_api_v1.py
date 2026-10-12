"""Guarded native Torch1.11 CUDA RNG/backend API diagnostic; no training credit."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching import run_shcl_training_rng_job_v1 as job_api
from scripts.flow_matching.run_shcl_native_script_queue_v1 import gpu_snapshot
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive
from scripts.flow_matching.timegan_native_common_v1 import write


def verify(config):
    native = job_api.replay.original.native
    for pin in config['source_receipts']:
        if native.sha(native.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen native CUDA API audit changed')
    reference = native.read(native.ROOT / config['full49_reference_configuration'])
    job_api.verify(reference)
    for key in ('settings', 'gpu_settings', 'resource_lock', 'python', 'environment', 'original_scope'):
        if config[key] != reference[key]:
            raise ValueError('Preserve every original saved-state audit setting')
    if config['canonical_training_jobs_credited'] != 0 or not config['diagnostic_only']:
        raise ValueError('Native API diagnostic cannot satisfy any original training obligation')
    return reference


def worker(config, path):
    native = job_api.replay.original.native
    reference = verify(config)
    if job_api.replay.original.environment() != config['environment']:
        raise ValueError('Exact acquired native environment required')
    import torch
    from scripts.flow_matching.shcl_native_observer_v1 import Observer
    torch.set_num_threads(2)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_settings']['gpu_uuid']
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('One original UUID-bound actual CUDA device required; never use CPU fallback')
    torch.cuda.get_rng_state_all()
    observation = Observer({}, '.')
    state = observation.rng()
    original_flags = observation.backend_flags()
    assert len(state['torch_cuda']) == 1
    # Use newly captured native API state, not a relabeled original CUDA epoch.
    # Opposite backend settings exercise actual restoration instead of setting
    # values that happen to equal the caller's defaults.
    checkpoint = dict(original_rng=state, backend_flags={key: not value for key, value in original_flags.items()})
    result = job_api.replay.restore_and_verify_draws(checkpoint)
    job_api.replay.original.verify_rng(state)
    assert observation.backend_flags() == original_flags
    if result['original_visible_CUDA_generators_verified'] != 1:
        raise ValueError('Actual native CUDA stream must be verified')
    missing_cuda = copy.deepcopy(checkpoint)
    missing_cuda['original_rng']['torch_cuda'] *= 2
    rejected = False
    try:
        job_api.replay.restore_and_verify_draws(missing_cuda)
    except ValueError:
        rejected = True
    if not rejected:
        raise ValueError('Wrong native CUDA generator count must fail')
    job_api.replay.original.verify_rng(state)
    assert observation.backend_flags() == original_flags
    target = native.ROOT / config['output_root']
    target.mkdir(parents=True, exist_ok=False)
    exclusive(target / 'actual_native_CUDA_rng_api_audit.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), configuration=path,
        configuration_sha256=native.sha(native.ROOT / path), environment=config['environment'],
        actual_native_device_name=torch.cuda.get_device_name(0), visible_CUDA_uuid=config['gpu_settings']['gpu_uuid'],
        actual_native_API_restoration=result, all_six_backend_flags_changed_restored_and_caller_preserved=True,
        wrong_visible_CUDA_generator_count_rejected=True, CPU_NumPy_Python_and_CUDA_caller_states_bitwise_preserved=True,
        synthetic_operator_only_not_benchmark_performance=True, source_original_CUDA_training_checkpoint_used=False,
        original_saved_training_rng_audits_still_required=len(reference['training_job_ids']),
        full_original_training_epochs=reference['full_original_training_epochs'], full_original_source_commands=413,
        original_scope=config['original_scope'], new_optimizer_updates_executed=0,
        canonical_training_jobs_credited=0, new_benchmark_results=0, original_paper_equivalence_certified=False,
        all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    native = job_api.replay.original.native
    config = native.read(native.ROOT / args.config)
    verify(config)
    if args.worker:
        worker(config, args.config)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    api['resource_snapshot'] = resource_snapshot
    base = native.ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    if (native.ROOT / config['output_root']).exists():
        raise FileExistsError('Preserve every native CUDA API audit attempt')
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Same native CUDA diagnostic already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), command=psutil.Process().cmdline(),
        active_pid=None, configuration_sha256=native.sha(native.ROOT / args.config), all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    handle = None
    try:
        while handle is None:
            snapshot = dict(resource_snapshot(), **gpu_snapshot(config['gpu_settings']))
            ready = api['has_headroom'](snapshot, config['settings']) and snapshot['gpu_free_bytes'] >= config['gpu_settings']['minimum_gpu_free_bytes']
            handle = api['exclusive_lock'](native.ROOT / config['resource_lock']) if ready else None
            if handle is not None:
                confirmed = dict(resource_snapshot(), **gpu_snapshot(config['gpu_settings']))
                if api['has_headroom'](confirmed, config['settings']) and confirmed['gpu_free_bytes'] >= config['gpu_settings']['minimum_gpu_free_bytes']:
                    break
                handle.close()
                handle = None
            update(dict(snapshot, state='waiting_for_original_readonly_native_CUDA_capacity'))
            time.sleep(10)
        command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_shcl_cuda_rng_api_v1', '--config', args.config, '--worker']
        with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=native.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    CUDA_VISIBLE_DEVICES=config['gpu_settings']['gpu_uuid']), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            update(dict(state='checking_actual_native_CUDA_rng_and_backend_APIs', active_pid=child.pid,
                active_create_time=psutil.Process(child.pid).create_time(), worker_command=command))
            gpu_abort = {}
            def guarded_update(values):
                gpu = gpu_snapshot(config['gpu_settings'])
                if gpu['gpu_free_bytes'] < config['gpu_settings']['emergency_gpu_free_bytes']:
                    gpu_abort.update(resource_guard_aborted=True, abort_gpu_snapshot=gpu)
                    api['terminate_tree'](child)
                update(dict(values, **gpu))
            code, usage = capped_wait(child, config, api, guarded_update)
        report = native.ROOT / config['output_root'] / 'actual_native_CUDA_rng_api_audit.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and not gpu_abort and report.exists()
        if passed:
            evidence = native.read(report)
            passed = evidence['passed'] and evidence['new_optimizer_updates_executed'] == 0 and evidence['canonical_training_jobs_credited'] == 0
        write(base / 'receipt.json', dict(usage, **gpu_abort, passed=passed, exit_code=code,
            report_sha256=native.sha(report) if report.exists() else None))
        update(dict(state='native_CUDA_rng_API_verified_original_training_states_pending' if passed else 'failed_native_CUDA_rng_API_audit_preserved', active_pid=None))
    finally:
        if handle is not None:
            handle.close()
        owner.close()


if __name__ == '__main__':
    main()
