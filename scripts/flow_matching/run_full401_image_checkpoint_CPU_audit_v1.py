"""Use original complete-checkpoint CPU guards without any CUDA context."""
import argparse
from datetime import datetime, timezone
import os
import subprocess

from scripts.flow_matching import audit_full401_image_checkpoint_CPU_v1 as audit
from scripts.flow_matching.timegan_native_common_v1 import write


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = audit.read(audit.ROOT/args.config); audit.verify(config)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); api['resource_snapshot'] = resource_snapshot
    base = audit.ROOT/config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Same full401 checkpoint audit already owned')
    if (audit.ROOT/config['output_root']).exists(): raise FileExistsError('Preserve every complete or failed audit attempt')
    actor = psutil.Process(); state = dict(pid=actor.pid, create_time=actor.create_time(), command=actor.cmdline(),
        active_pid=None, configuration_sha256=audit.sha(audit.ROOT/args.config), all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json', state)
    try:
        with api['resource_slot'](audit.ROOT/config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_full401_image_checkpoint_CPU_v1', '--config', args.config]
            with (base/'worker.stdout.log').open('xb') as out, (base/'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=audit.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='verifying_full401_CPU_checkpoint_without_CUDA', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), worker_command=command))
                code, usage = capped_wait(child, config, api, update)
        report = audit.ROOT/config['output_root']/'full401_native_model_optimizer_CPU_rng_restore.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and report.exists()
        if passed:
            result = audit.read(report)
            passed = result['passed'] and result['completed_original_epochs']==401 and not result['CUDA_initialized']
        write(base/'receipt.json', dict(usage, exit_code=code, passed=passed, report_sha256=audit.sha(report) if report.exists() else None))
        update(dict(usage, state='full401_CPU_state_verified_GPU_resume_pending' if passed else 'failed_full401_CPU_audit_preserved', active_pid=None))
    finally: owner.close()


if __name__ == '__main__': main()
