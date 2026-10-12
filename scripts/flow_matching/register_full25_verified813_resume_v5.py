"""Freeze verified813 CPU evidence and every original full image training job."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching import run_full_image_training_recovery_queue_v5 as queue
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive

CONFIG='configs/reproducibility/fm_full_image_training_verified813_resume.2026-10-13_v5.json'
RESULT='projects/flow_matching_research/results/2026-10-13/full813_verified_CPU_and_full25_GPU_resume_v5'


def register():
    import psutil
    old=queue.original;root=old.ROOT
    previous_path='configs/reproducibility/fm_full_image_training_verified401_resume.2026-10-12_v4.json'
    previous=old.read(root/previous_path)
    state=old.read(root/previous['state_root']/'status.json')
    if state['state']!='failed_full_training_attempt_preserved' or state['active_pid'] is not None:
        raise ValueError('The original native training attempt must already be terminal')
    for pid,ct in ((state['pid'],state['create_time']),(state['last_terminal']['worker_pid'],state['last_terminal']['worker_create_time'])):
        try:
            if psutil.Process(pid).create_time()==ct:raise ValueError('Never replace live scientific training')
        except psutil.NoSuchProcess:pass
    archive_path='configs/reproducibility/fm_full813_checkpoint_immutable_archive.2026-10-13_v1.json'
    archive=old.read(root/archive_path)
    proof_path='projects/flow_matching_research/results/2026-10-13/full813_resource_incident_and_CPU_audit_v1/native_CPU_state/full813_native_model_optimizer_CPU_rng_restore.json'
    terminal_path='experiments/runs/fm_full813_image_checkpoint_CPU_audit_v1/receipt.json'
    cpu=old.read(root/proof_path);terminal=old.read(root/terminal_path)
    if (terminal['exit_code']!=0 or not terminal['passed'] or terminal['resource_guard_aborted']
            or terminal['report_sha256']!=old.sha(root/proof_path) or not cpu['passed']
            or cpu['completed_original_epochs']!=813 or cpu['verified_original_optimizer_updates']!=92682):
        raise ValueError('Actual successful complete813 native CPU restoration required')
    old.verify_receipts([archive['source_checkpoint'],archive['immutable_archive']])
    job_path=previous['ordered_original_jobs'][0];job=old.read(root/job_path)
    progress=old.read(root/job['output_directory']/'progress.json')
    if progress['completed_epochs']!=813 or progress['global_steps']!=92682:
        raise ValueError('Canonical latest checkpoint must still be the full813 state')
    paths=[previous_path,archive_path,proof_path,terminal_path,
        'configs/runtime/fm_full813_image_checkpoint_CPU_audit.v1.json',
        'scripts/flow_matching/run_full_image_training_execution_recovery_v5.py',
        'scripts/flow_matching/run_full_image_training_recovery_queue_v5.py',
        'scripts/flow_matching/register_full25_verified813_resume_v5.py',
        'tests/test_fm_full25_verified813_resume_v5.py']
    pins=[dict(path=p,sha256=old.sha(root/p)) for p in paths]
    config=deepcopy(previous);config.pop('saved401_resume')
    config.update(configuration_path=CONFIG,previous_configuration=previous_path,
        state_root='experiments/runs/fm_full25_image_training_verified813_resume_v5',
        saved813_resume=dict(job_configuration=job_path,completed_epochs=813,remaining_epochs=1187,
            CPU_proof=dict(path=proof_path,sha256=old.sha(root/proof_path)),
            CPU_terminal=dict(path=terminal_path,sha256=old.sha(root/terminal_path)),archive=archive['immutable_archive']))
    config['settings']['minimum_commit_headroom_bytes']=40*1024**3
    config['source_receipts']+=pins
    attempt=Path(progress['attempt_directory'])
    config['original_failure_receipts']+=[dict(path=str(attempt/n),sha256=old.sha(attempt/n)) for n in ('terminal.json','stderr.log')]
    config['boundary']+=(' Latest canonical813 state is fully restored on CPU before dispatch. '
        'Only admission increases28to40GiB after the actual commit-guard abort; emergency16GiB, GPU guards, '
        'all source math, full32image batch, BF16/TF32, full45M model and all25 original2000epoch tasks retained. '
        'The first GPU worker must restore every model/optimizer and all original CUDA/CPU/order RNG bitwise '
        'before the remaining1187 epochs. CPU verification is no GPU restoration or benchmark certificate.')
    queue.verify_registration(config)
    if (root/CONFIG).exists() or (root/RESULT).exists():raise FileExistsError('Preserve every prior registration and result')
    exclusive(root/CONFIG,config)
    exclusive(root/RESULT/'actual_full813_CPU_completion_and_full25_resume_registration.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),actual_CPU_terminal=terminal,
        immutable_checkpoint=archive['immutable_archive'],actual_CPU_proof=config['saved813_resume']['CPU_proof'],
        original_failed_terminal=old.read(attempt/'terminal.json'),remaining_first_job_epochs=1187,
        original_full_image_jobs=25,original_FM_jobs=20,original_WT_jobs=5,epochs_per_original_job=2000,
        minimum_commit_headroom_bytes=40*1024**3,emergency_commit_headroom_bytes=16*1024**3,
        source_math_and_all_original_guard_values_except_stronger_admission_unchanged=True,
        actual_full_native_GPU_resume_passed=False,actual_remaining_training_complete=False,
        original_scope=config['original_scope'],benchmark_performance=False,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=CONFIG,verified_CPU_epochs=813,remaining_original_epochs=1187,
        original_image_jobs=25,minimum_commit_headroom_bytes=40*1024**3)))


if __name__=='__main__':register()
