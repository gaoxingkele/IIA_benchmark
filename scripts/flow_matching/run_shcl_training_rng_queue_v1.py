"""Guard the saved fixture, then all49 complete native training RNG audits."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching import run_shcl_training_rng_job_v1 as job_api
from scripts.flow_matching.run_shcl_native_script_queue_v1 import gpu_snapshot, complete


def accepted(report, job, fixture):
    return (report.get('passed') is True and report.get('source_training_job') == job
        and report.get('every_original_training_RNG_and_backend_state_verified') is True
        and report.get('all_original_training_epoch_model_Adam_RNG_backend_restore_verified') is True
        and report.get('synthetic_operator_only_not_benchmark_performance') is fixture
        and report.get('new_optimizer_updates_executed') == 0 and report.get('new_benchmark_results') == 0
        and len(report.get('training_rng_checkpoints', [])) == job['num_epochs']
        and all(row.get('all_original_training_RNG_states_bitwise_restored') is True
            and row.get('all_original_training_backend_flags_exactly_restored') is True
            and row.get('independent_native_stream_draws_bitwise_verified') is True
            and (fixture or row.get('original_visible_CUDA_generators_verified') == 1)
            for row in report.get('training_rng_checkpoints', [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fixture-only', action='store_true')
    args = parser.parse_args()
    native = job_api.replay.original.native
    config = native.read(native.ROOT / args.config)
    queue, model = job_api.verify(config)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    api['resource_snapshot'] = resource_snapshot
    root = native.ROOT / (config['fixture_state_root'] if args.fixture_only else config['state_root'])
    root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](root / 'controller.lock')
    if owner is None:
        raise RuntimeError('Same independent training RNG audit already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), command=psutil.Process().cmdline(),
        configuration_sha256=native.sha(native.ROOT / args.config), verified={}, failures={}, active_pid=None,
        all_original_experiments_complete=False, original_scope=queue['original_scope'],
        full_original_source_commands=413, full_training_rng_obligations=49,
        synthetic_fixture_is_not_canonical_credit=True)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        native.write(root / 'status.json', state)
    @contextmanager
    def slot(fixture):
        while True:
            snapshot = resource_snapshot()
            gpu = {} if fixture else gpu_snapshot(config['gpu_settings'])
            ready = api['has_headroom'](snapshot, config['settings']) and (not gpu or gpu['gpu_free_bytes'] >= config['gpu_settings']['minimum_gpu_free_bytes'])
            handle = api['exclusive_lock'](native.ROOT / config['resource_lock']) if ready else None
            if handle is not None:
                confirmed = resource_snapshot()
                gpu = {} if fixture else gpu_snapshot(config['gpu_settings'])
                if api['has_headroom'](confirmed, config['settings']) and (not gpu or gpu['gpu_free_bytes'] >= config['gpu_settings']['minimum_gpu_free_bytes']):
                    try:
                        yield
                    finally:
                        handle.close()
                    return
                handle.close()
            update(dict(snapshot, **gpu, state='waiting_for_original_readonly_capacity', shared_slot_owned=False))
            time.sleep(10)
    def execute(job, fixture):
        identifier = 'saved_full_native_fixture' if fixture else job['id']
        output = native.ROOT / config['fixture_output_root'] if fixture else Path(config['full_training_output_root']) / identifier
        report = output / 'full_checkpoint_rng_restore.json'
        if report.exists():
            evidence = native.read(report)
            if not accepted(evidence, job, fixture) or evidence['configuration_sha256'] != state['configuration_sha256']:
                raise ValueError('Only exact complete full-state evidence satisfies the independent gate')
            state['verified'][identifier] = dict(report=str(report), sha256=native.sha(report))
            return
        if output.exists():
            state['failures'][identifier] = 'partial_rng_audit_preserved'
            return
        with slot(fixture):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.run_shcl_training_rng_job_v1', '--config', args.config]
            command += ['--fixture'] if fixture else ['--job-id', identifier]
            logs = root / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / (identifier + '.stdout.log')).open('xb') as out, (logs / (identifier + '.stderr.log')).open('xb') as err:
                child = subprocess.Popen(command, cwd=native.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        CUDA_VISIBLE_DEVICES='-1' if fixture else config['gpu_settings']['gpu_uuid']),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='restoring_every_saved_training_RNG_and_backend_state', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), worker_command=command))
                gpu_abort = {}
                def guarded_update(values):
                    gpu = {} if fixture else gpu_snapshot(config['gpu_settings'])
                    if gpu and gpu['gpu_free_bytes'] < config['gpu_settings']['emergency_gpu_free_bytes']:
                        gpu_abort.update(resource_guard_aborted=True, abort_gpu_snapshot=gpu)
                        api['terminate_tree'](child)
                    update(dict(values, **gpu))
                code, usage = capped_wait(child, config, api, guarded_update)
            passed = code == 0 and not usage['resource_guard_aborted'] and not gpu_abort and report.exists()
            if passed:
                passed = accepted(native.read(report), job, fixture)
            native.write(root / 'receipts' / (identifier + '.json'), dict(usage, **gpu_abort, exit_code=code, passed=passed))
            if passed:
                state['verified'][identifier] = dict(report=str(report), sha256=native.sha(report))
            else:
                state['failures'][identifier] = 'failed_rng_audit_preserved'
            update(dict(active_pid=None))
    try:
        update({})
        if args.fixture_only:
            reference = native.read(native.ROOT / config['saved_fixture_configuration'])
            fixture_job = native.read(reference['fixture_training'])['job']
            execute(fixture_job, True)
            update(dict(state='saved_native_checkpoint_rng_fixture_verified' if state['verified'] else 'failed_fixture_preserved'))
            return
        fixture = native.read(native.ROOT / config['fixture_output_root'] / 'full_checkpoint_rng_restore.json')
        if not fixture['passed'] or not fixture['synthetic_operator_only_not_benchmark_performance'] or fixture['new_benchmark_results'] != 0:
            raise ValueError('Saved fixture must pass without canonical performance credit')
        while True:
            for job in queue['jobs']:
                if job['mode'] != 'train' or job['id'] in state['verified'] or job['id'] in state['failures']:
                    continue
                if complete(job):
                    execute(job, False)
            update(dict(state='all49_complete_training_rng_states_verified_full413_and_TAB_gates_retained' if len(state['verified']) == 49
                else 'waiting_for_full_original_training_checkpoints', verified_count=len(state['verified']),
                failed_count=len(state['failures']), required_count=49, pending_count=49-len(state['verified'])))
            if len(state['verified']) == 49:
                return
            time.sleep(30)
    finally:
        owner.close()


if __name__ == '__main__':
    main()
