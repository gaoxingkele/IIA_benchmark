"""Guarded remaining native TimeGAN predictor training; unchanged family guards."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.timegan_posthoc_recovery_v1 import binding, compile_recovery, verify_recovery_result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    queue, model, job, records, evaluators = binding(config)
    if args.probe_only:
        function, verified_queue, recovered_job = compile_recovery(config)
        print(dict(canonical_job_id=job['id'], original_jobs=len(queue['jobs']),
                   full_fresh_predictor_updates_required=50000, generator_retraining=False,
                   tensorflow_imported='tensorflow' in sys.modules, training_executed=False))
        return
    if args.worker:
        function, verified_queue, recovered_job = compile_recovery(config)
        function(verified_queue, recovered_job, config['original_queue'])
        verify_recovery_result(config)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    state_root = ROOT / config['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](state_root / 'controller.lock')
    if owner is None:
        raise RuntimeError('Posthoc recovery owner already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
                 canonical_job_id=job['id'], active_pid=None, configuration_sha256=sha(ROOT / args.config),
                 independent_training_seed_slots_added=0, all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(state_root / 'status.json', state)

    try:
        if (ROOT / config['output_directory']).exists():
            raise FileExistsError('Never overwrite a partial or completed recovery')
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            argv = ['native_posthoc_recovery', '--config', args.config, '--worker']
            bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");sys.argv=" + repr(argv) + ";"
                         "runpy.run_module('scripts.flow_matching.run_timegan_posthoc_recovery_v1',run_name='__main__')")
            command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (state_root / 'worker.stdout.log').open('xb') as out, (state_root / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                             OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_all_ten_full_native_predictors', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        passed = False
        if code == 0 and usage['resource_guard_aborted'] is False:
            verify_recovery_result(config)
            passed = True
        write(state_root / 'receipt.json', dict(exit_code=code, full_native_posthoc_recovery_verified=passed, **usage))
        update(dict(state='completed_pending_independent_full_replay' if passed else 'failed_attempt_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
