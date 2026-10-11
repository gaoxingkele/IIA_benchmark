"""Capture terminal resource failures and live identities without PID assumptions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import psutil

from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha


def process_identity(pid, expected_create_time=None, expected_command_token=None):
    # psutil.Process(None) selects this auditor: an idle worker must not be
    # mistaken for a running experiment. Also reject bool (an int subclass).
    if type(pid) is not int or pid <= 0:
        return dict(live=False, reason='no_positive_integer_pid')
    try:
        handle = psutil.Process(pid)
        created = handle.create_time()
        command = handle.cmdline()
        if expected_create_time is not None and created != expected_create_time:
            return dict(live=False, reason='pid_reused', pid=pid, actual_create_time=created)
        if expected_command_token and not any(expected_command_token in arg for arg in command):
            return dict(live=False, reason='command_identity_differs', pid=pid, actual_command=command)
        return dict(live=handle.is_running(), pid=pid, create_time=created,
                    actual_command=command, cpu_seconds=sum(handle.cpu_times()[:2]),
                    create_time_verified=expected_create_time is not None)
    except psutil.NoSuchProcess:
        return dict(live=False, reason='process_missing', pid=pid)
    except psutil.AccessDenied:
        return dict(live=None, reason='identity_access_denied', pid=pid)


def capture_state(path, token):
    # A live controller atomically replaces its state. Hash the exact bytes
    # parsed, rather than a later version read separately for its checksum.
    content = (ROOT / path).read_bytes()
    state = json.loads(content.decode('utf-8'))
    return dict(path=path, sha256_at_capture=hashlib.sha256(content).hexdigest(), state=state,
        controller_identity=process_identity(state.get('pid'), state.get('create_time'), token),
        worker_identity=process_identity(state.get('active_pid'), state.get('active_create_time')),
        state_file_alone_is_completion_proof=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        failed_attempts=[], runtime=[], all_original_experiments_complete=False,
        numeric_leaderboard_unchanged=True,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273))
    for name, token, receipt_name in [
        ('fm_timegan_stock_stage_recovery_v2', 'run_timegan_stage_resume_v2', 'receipt.json'),
        ('fm_rcgan_native_original_mnist500_v2', 'run_rcgan_native_original_queue_v2', 'receipts/rcgan_mnistfull__seed1103.json'),
        ('fm_cfm_ts_exact_recovery_v4', 'run_cfm_ts', 'receipts/cfm_ts_original__main_text__node_e400__pendulum__seed1103__exact_recovery_v4.json'),
    ]:
        prefix = 'experiments/runs/' + name
        captured = capture_state(prefix + '/status.json', token)
        receipt_path = prefix + '/' + receipt_name
        receipt = read(ROOT / receipt_path)
        observations = receipt.get('resource_observations', receipt)
        if receipt['exit_code'] == 0 or not observations['resource_guard_aborted']:
            raise ValueError('Authoritative failed resource-abort receipt required')
        report['failed_attempts'].append(dict(
            **captured, terminal_receipt_path=receipt_path, terminal_receipt_sha256=sha(ROOT / receipt_path),
            terminal_receipt=receipt, failed_attempt_preserved=True,
            subsequent_job_runtime_does_not_resolve_this_failed_seed=True))
    paths = [
        ('fm_timegan_stock_stage_recovery_v3', 'run_timegan_stage_resume_v2'),
        ('fm_timegan_native_original_v1', 'run_timegan_native_queue_v1'),
        ('fm_timegan_energy_stage_recovery_v1', 'run_timegan_stage_resume_v2'),
        ('fm_cfm_ts_exact_recovery_v5', 'run_cfm_ts_receipt_runtime_v3'),
        ('fm_cfm_ts_original_v1', 'run_cfm_ts_low_memory_queue_v2'),
        ('fm_grasp_tau01104_full_array_audit_v1', 'audit_registered_grasp_results_v1'),
        ('fm_grasp_tau01104_full_checkpoint_replay_v1', 'audit_grasp_checkpoint_replay_v1'),
        ('fm_pi_original_source_full_checkpoint_replay_v1', 'replay_pi_original_full_checkpoint_v1'),
        ('fm_full_image_cuda_capacity_v2', 'run_full_image_cuda_capacity_queue_v2'),
        ('fm_grasp_protocol_v4', 'run_grasp_gpu_memory_fair_queue_v6'),
        ('fm_resource_recovery_20261011_v4', 'run_preserved_incident_recovery_v4'),
    ]
    for name, token in paths:
        path = 'experiments/runs/' + name + '/status.json'
        if (ROOT / path).exists():
            report['runtime'].append(capture_state(path, token))
    report['boundaries'] = [
        'Terminal job receipts supersede stale resource flags retained in controller state.',
        'A live parent or later seed does not certify an earlier failed seed.',
        'A missing PID or idle worker is not an experiment; no PID=None self-attestation.',
        'Resource waiting, trained artifacts, full-array audit and full-model replay are separate stages.',
        'Capacity checks and CPU fixtures do not constitute benchmark performance.',
    ]
    exclusive_json(ROOT / args.output, report)
    print(json.dumps(dict(failed_attempts=len(report['failed_attempts']),
                          runtime_states=len(report['runtime']), benchmark_results_added=0)))


if __name__ == '__main__':
    main()
