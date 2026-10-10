"""Guarded full-budget TimeGAN recovery sharing the original CPU family lock."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.timegan_stage_resume_v2 import binding, compile_observed_resume, verify_recovery_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    queue, model, job, records = binding(config)
    for receipt in config.get('recovery_source_receipts', []):
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen recovery implementation changed')
    if args.probe_only:
        function, verified_queue, recovered_job = compile_observed_resume(config)
        print(dict(canonical_job_id=job['id'], full_scope_jobs=len(queue['jobs']),
                   native_budget_preserved=True, independent_seed_slots_added=0,
                   tensorflow_imported='tensorflow' in sys.modules))
        return
    if args.worker:
        function, verified_queue, recovered_job = compile_observed_resume(config)
        function(verified_queue, recovered_job, config['original_queue'])
        verify_recovery_result(config)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    state_root = ROOT / config['state_root']
    state_root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](state_root / 'controller.lock')
    if owner is None:
        raise RuntimeError('Recovery controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
                 canonical_job_id=job['id'], active_pid=None, configuration_sha256=sha(ROOT / args.config),
                 independent_training_seed_slots_added=0, all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(state_root / 'status.json', state)

    try:
        if (ROOT / config['output_directory']).exists():
            raise FileExistsError('Preserve partial recovery; choose a new registered attempt')
        with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
            bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");"
                         "sys.argv=['run_timegan_stage_resume_v2','--config'," + repr(args.config) + ",'--worker'];"
                         "runpy.run_module('scripts.flow_matching.run_timegan_stage_resume_v2',run_name='__main__')")
            command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (state_root / 'worker.stdout.log').open('xb') as out, (state_root / 'worker.stderr.log').open('xb') as err:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                             OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_full_original_joint_training_and_evaluators', active_pid=process.pid,
                    active_create_time=psutil.Process(process.pid).create_time(), actual_command=command))
                def running(values):
                    update(dict(values, state='running_full_original_joint_training_and_evaluators'))
                code, usage = capped_wait(process, queue, api, running)
        passed = False
        if code == 0:
            verify_recovery_result(config)
            passed = True
        write(state_root / 'receipt.json', dict(exit_code=code, full_native_recovery_result_verified=passed, **usage))
        update(dict(state='completed' if passed else 'failed_attempt_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
