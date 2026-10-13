"""Register full-budget same-seed attempts after two actual startup commit aborts."""
import copy
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive
from scripts.flow_matching.gan_zero_checkpoint_retry_v1 import verify_zero_checkpoint_incident

EVIDENCE = 'projects/flow_matching_research/results/2026-10-13/gan_zero_checkpoint_commit_retries_v1'
TG_CONFIG = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v4.json'
RC_CONFIG = 'configs/runtime/fm_rcgan_seed1107_zero_checkpoint_full_retry.v1.json'
RC_QUEUE = 'configs/experiments/fm_rcgan_seed1107_zero_checkpoint_full_retry_queue.v1.json'


def pin(path):
    path = ROOT / path
    return dict(path=path.as_posix(), sha256=sha(path))


def incident(folder, prefix, queue, identity, previous_state, receipt_path, logs_root):
    canonical = next(job for job in queue['jobs'] if job['id'] == identity)
    receipt = read(ROOT / receipt_path)
    output = ROOT / canonical['output_directory']
    paths = [receipt_path, previous_state,
        logs_root + '/' + identity + '.stdout.log', logs_root + '/' + identity + '.stderr.log']
    artifacts = []
    for source in paths:
        original = ROOT / source
        target = folder / (prefix + '_' + original.name)
        with target.open('xb') as stream:
            stream.write(original.read_bytes())
        artifacts.append(dict(source_path=source, archive_name=target.name, sha256=sha(target)))
    status = read(ROOT / previous_state)
    if not status.get('active_create_time') or not status.get('actual_command'):
        raise ValueError('Exact source creation-time and command evidence required')
    record = dict(canonical_job=canonical, native_python=queue['python'], terminal_receipt=receipt_path,
        stopped_native_identity=dict(pid=None, pid_recording='not_retained_in_terminal_controller_state',
            create_time=status['active_create_time'], command=status['actual_command']),
        source_receipts=[pin(ROOT / receipt_path)] + [pin(folder / item['archive_name']) for item in artifacts],
        completed_training_budget_credited=0, failed_unsaved_optimizer_updates='not_recorded',
        original_failed_output_preserved=True, original_terminal_receipt=receipt,
        archived_artifacts=artifacts, captured_utc=datetime.now(timezone.utc).isoformat())
    verify_zero_checkpoint_incident(record)
    exclusive(folder / (prefix + '_zero_checkpoint_incident.json'), record)
    return record


def main():
    import psutil
    folder = ROOT / EVIDENCE
    folder.mkdir(parents=True, exist_ok=False)
    previous_tg = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v3.json'
    tg = copy.deepcopy(read(ROOT / previous_tg))
    tg_queue = read(ROOT / tg['original_queue'])
    tg_incident = incident(folder, 'timegan_sine1104', tg_queue, 'timegan_native__gru__sine__seed1104',
        'experiments/runs/fm_timegan_corrected_observer_original_v3/status.json',
        'experiments/runs/fm_timegan_corrected_observer_original_v3/receipts/timegan_native__gru__sine__seed1104.json',
        'experiments/runs/fm_timegan_corrected_observer_original_v3/logs')
    tg_incident['execution_output_directory'] = 'F:/aicoding/IIA_Data/experiments/timegan_sine1104_zero_checkpoint_full_retry_v1/jobs/timegan_native__gru__sine__seed1104'
    owner = psutil.Process(35888)
    tg.update(runtime_configuration=TG_CONFIG, zero_checkpoint_retry=tg_incident,
        retry_admission_commit_headroom_bytes=40 * 1024 ** 3)
    tg['roles']['original'].update(runtime_configuration=TG_CONFIG,
        previous_owner=dict(pid=owner.pid, create_time=owner.create_time(), command=owner.cmdline()),
        previous_state_path='experiments/runs/fm_timegan_corrected_observer_original_v3/status.json',
        state_root='experiments/runs/fm_timegan_corrected_observer_original_v4')
    original_rc = 'configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json'
    rc_original = read(ROOT / original_rc)
    rc_incident = incident(folder, 'rcgan_seed1107', rc_original, 'rcgan_mnistfull__seed1107',
        'experiments/runs/fm_rcgan_native_original_mnist500_v2/status.json',
        'experiments/runs/fm_rcgan_native_original_mnist500_v2/receipts/rcgan_mnistfull__seed1107.json',
        'experiments/runs/fm_rcgan_native_original_mnist500_v2/logs')
    rc_incident['execution_output_directory'] = 'F:/aicoding/IIA_Data/experiments/rcgan_seed1107_zero_checkpoint_full_retry_v1/jobs/seed1107'
    rc_queue = copy.deepcopy(rc_original)
    rc_queue['state_root'] = 'experiments/runs/fm_rcgan_seed1107_zero_checkpoint_full_retry_v1'
    next(job for job in rc_queue['jobs'] if job['id'] == rc_incident['canonical_job']['id'])['output_directory'] = rc_incident['execution_output_directory']
    exclusive(ROOT / RC_QUEUE, rc_queue)
    added = [previous_tg, original_rc, RC_QUEUE,
        'scripts/flow_matching/gan_zero_checkpoint_retry_v1.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v4.py',
        'scripts/flow_matching/run_rcgan_zero_checkpoint_full_retry_v1.py',
        'scripts/flow_matching/register_gan_zero_checkpoint_retries_v1.py',
        'tests/test_fm_gan_zero_checkpoint_retries_v1.py']
    added += [path.relative_to(ROOT).as_posix() for path in folder.iterdir()]
    receipts = {item['path']: item['sha256'] for item in tg['source_receipts']}
    for path in added:
        receipts[path] = sha(ROOT / path)
    tg['source_receipts'] = [dict(path=path, sha256=value) for path, value in sorted(receipts.items())]
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v4 import verify_runtime
    verify_runtime(tg)
    exclusive(ROOT / TG_CONFIG, tg)
    rc = dict(original_queue=original_rc, queue=RC_QUEUE, state_root=rc_queue['state_root'],
        retry_admission_commit_headroom_bytes=40 * 1024 ** 3, zero_checkpoint_retry=rc_incident,
        source_receipts=[pin(ROOT / path) for path in added],
        original_scope=tg['original_scope'], controller_python=tg_queue['controller_python'] if 'controller_python' in tg_queue else str(Path(__import__('sys').executable)))
    from scripts.flow_matching.run_rcgan_zero_checkpoint_full_retry_v1 import verify
    verify(rc)
    exclusive(ROOT / RC_CONFIG, rc)
    exclusive(folder / 'registered_complete_retry_obligations.json', dict(
        required_original_TimeGAN_jobs=45, required_original_RCGAN_jobs=5,
        TimeGAN_full_training_loops_each=50000, TimeGAN_training_stages=3,
        TimeGAN_full_native_evaluators_each=20, RCGAN_full_epochs_each=500,
        RCGAN_full_evaluators_each=500, credited_prior_completed_stages_or_epochs=0,
        original_scope=tg['original_scope'], stronger_admission_commit_headroom_bytes=40 * 1024 ** 3,
        unchanged_emergency_commit_headroom_bytes=16 * 1024 ** 3,
        all_existing_other_job_output_bindings_unchanged=True,
        full_independent_replay_routes_require_completion_binding_for_these_new_outputs=True,
        original_failed_attempts_preserved=True, additional_training_seed_slots=0,
        actual_full_training_completed=False, strict_TAB_result=False,
        author_equivalence_certified=False, all_original_experiments_complete=False))
    print(dict(TimeGAN_configuration=TG_CONFIG, RCGAN_configuration=RC_CONFIG,
        original_scope=tg['original_scope'], registered_only=True))


if __name__ == '__main__':
    main()
