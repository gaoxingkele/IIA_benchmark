"""Guard complete saved-stage re-inference with unchanged read-only audit caps."""
import argparse
from datetime import datetime, timezone
import os
import subprocess

from scripts.flow_matching.audit_timegan_saved_predecessor_native_v1 import original, verify, compiled_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = original.read(original.ROOT / args.config)
    queue, model, job, records, evaluators = verify(config)
    compiled_worker()
    output = original.ROOT / config['output_root'] / job['id']
    if output.exists():
        raise FileExistsError('Preserve every full saved-stage replay attempt')
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    base = original.ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Same complete saved-stage native audit already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None,
        canonical_job_id=job['id'], configuration_sha256=original.sha(original.ROOT / args.config),
        all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        original.write(base / 'status.json', state)

    try:
        with api['resource_slot'](original.ROOT / config['resource_lock'], config['settings'], update):
            argv = ['native_saved_stage_replay', '--config', args.config]
            bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(original.ROOT)) + ");sys.argv=" + repr(argv) + ";"
                "runpy.run_module('scripts.flow_matching.audit_timegan_saved_predecessor_native_v1',run_name='__main__')")
            command = [config['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=original.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_complete_saved_predecessor_native_replay', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        report_path = output / 'saved_predecessor_native_replay.json'
        passed = code == 0 and usage['resource_guard_aborted'] is False and report_path.exists()
        if passed:
            report = original.read(report_path)
            if (report['passed'] is not True or report['actual_saved_checkpoint_restores'] != 13
                    or report['actual_saved_trained_evaluator_replays'] != 10
                    or report['original_source_protocol_iterations_reconstructed'] != 20000
                    or report['numerical_optimizer_updates_executed'] != 0
                    or report['final_full45_checkpoint_restores_still_required'] != 1035
                    or report['final_full45_trained_evaluator_replays_still_required'] != 900
                    or report['experiment_complete'] is not False):
                raise ValueError('All actual saved-stage replays required; final obligations cannot be waived')
        original.write(base / 'receipt.json', dict(exit_code=code, full_saved_predecessor_native_replay_passed=passed,
            report_sha256=original.sha(report_path) if report_path.exists() else None, **usage,
            full_experiment_complete=False, verification_only_not_benchmark_training=True))
        update(dict(state='completed_saved_stage_audit_full_predictors_and_final_replay_pending' if passed
            else 'failed_saved_stage_audit_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
