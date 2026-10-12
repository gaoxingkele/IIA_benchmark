"""Safely hand off an unstarted Sine retry and audit its saved full stages."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.timegan_stage_resume_v2 import binding


def verify(runtime):
    for pin in runtime['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen full Sine stage audit source changed: ' + pin['path'])
    config = read(ROOT / runtime['audit_configuration'])
    queue, model, job, records = binding(config)
    previous = read(ROOT / runtime['previous_registration'])
    incident = read(ROOT / previous['terminal_incident_evidence'])
    from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
    stopped = previous['stopped_native_handle']
    if process_identity(stopped['pid'], stopped['create_time'], stopped['command_token'])['live'] is not False:
        raise ValueError('Original memory-aborted native handle must be terminal')
    if incident['original_job'] != job or incident['actual_resource_observations']['resource_guard_aborted'] is not True:
        raise ValueError('Exact original memory-aborted canonical Sine job required')
    receipt = incident['original_terminal_evidence']
    if sha(ROOT / receipt['path']) != receipt['sha256'] or read(ROOT / receipt['path']) != receipt['record']:
        raise ValueError('Original terminal failure receipt changed')
    if (ROOT / previous['selected_output_directory']).exists():
        raise ValueError('Previous fresh retry has started; preserve its attempt')
    reference = read(ROOT / runtime['reference_audit_runtime'])
    if (runtime['settings'] != reference['settings'] or runtime['python'] != queue['python']
            or runtime['resource_lock'] != reference['resource_lock'] or job['dataset'] != 'sine'
            or runtime['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Full original scope/interpreter and existing audit guards must be retained')
    return config, queue, job, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    runtime = read(ROOT / args.runtime_config)
    config, queue, job, records = verify(runtime)
    from scripts.flow_matching.audit_timegan_sine_stage_restore_v3 import corrected_audit
    corrected_audit()
    if args.probe_only:
        print(json.dumps(dict(passed=True, full_original_jobs=45, canonical_job_id=job['id'],
            full_original_shape=[10000, 24, 5], training_executed=False,
            tensorflow_imported='tensorflow' in sys.modules, verification_only_not_benchmark=True)))
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import safe_handoff
    output = ROOT / config['native_restore_audit']
    if output.exists():
        raise FileExistsError('Never overwrite a saved native restore audit')
    api = resource_api()
    base = ROOT / runtime['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full native Sine stage audit already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None,
        configuration_sha256=sha(ROOT / args.runtime_config), canonical_job_id=job['id'],
        all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)

    try:
        update({})
        safe_handoff('original', runtime['handoff'], base, update, 10)
        verify(runtime)
        with api['resource_slot'](ROOT / runtime['resource_lock'], runtime['settings'], update):
            argv = ['full_native_sine_restore_audit', '--config', runtime['audit_configuration']]
            bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");sys.argv=" + repr(argv) + ";"
                "runpy.run_module('scripts.flow_matching.audit_timegan_sine_stage_restore_v3',run_name='__main__')")
            command = [runtime['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_full_native_restore_and_exact_transition_audit', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, runtime, api, update)
        passed = False
        if code == 0 and usage['resource_guard_aborted'] is False:
            proof = read(output)
            if (proof.get('passed') is not True or proof['canonical_job_id'] != job['id']
                    or proof['full_original_data_shape'] != [10000, 24, 5]
                    or proof['all_transition_variable_values_bitwise_equal'] is not True
                    or proof['numpy_minibatch_noise_trajectory_identical'] is not True
                    or proof['checkpoint_all_variable_restoration']['all_variable_values_bitwise_equal'] is not True
                    or proof['restored_completed_stage_updates'] != records[-1]['optimizer_updates']
                    or proof['full_native_joint_transition_iterations_per_path'] != config['audit_joint_iterations']):
                raise ValueError('Complete original model/optimizer/RNG restoration and exact transitions required')
            passed = True
        write(base / 'receipt.json', dict(exit_code=code, full_native_restore_audit_passed=passed, **usage,
            audit_sha256=sha(output) if output.exists() else None, verification_only_not_benchmark_training=True))
        update(dict(state='completed_restore_audit_pending_full_training' if passed else 'failed_audit_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
