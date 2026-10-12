"""Freeze all25 original image jobs and the verified immutable401epoch input."""
import copy
import json
from pathlib import Path

import psutil

from scripts.flow_matching import run_full_image_training_recovery_queue_v4 as queue
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    old=queue.original
    root=old.ROOT;path='configs/reproducibility/fm_full_image_training_execution_recovery.2026-10-11_v3.json'
    config=copy.deepcopy(old.read(root/path));old.verify_registration(config)
    prior=old.read(root/config['state_root']/'status.json')
    if prior['state']!='failed_full_training_attempt_preserved' or prior['active_pid'] is not None:
        raise ValueError('Actual preserved terminal training failure required before a new attempt')
    try:
        actor=psutil.Process(prior['pid'])
        if actor.create_time()==prior['create_time']:
            raise ValueError('Previous original full training coordinator must be terminal before restart')
    except psutil.NoSuchProcess:pass
    archive_path='configs/reproducibility/fm_full401_checkpoint_immutable_archive.2026-10-12_v1.json'
    archive=old.read(root/archive_path);old.verify_receipts([archive['immutable_full_state_archive'],archive['full401_CPU_proof']])
    latest='F:/aicoding/IIA_Data/public_datasets/flow_matching/experiments/flow_mismatching_full_training_v1/jobs/flow_mismatching_full__mvtec__conditional__seed1103/attempts/07d43be897da47659d3fb57f1c834d33'
    paths=[path,archive_path,'scripts/flow_matching/run_full_image_training_execution_recovery_v4.py',
        'scripts/flow_matching/run_full_image_training_recovery_queue_v4.py','scripts/flow_matching/register_full25_verified401_resume_v4.py',
        'tests/test_fm_full25_verified401_resume_v4.py','scripts/flow_matching/cached_resource_snapshot_v1.py']
    config.update(configuration_path='configs/reproducibility/fm_full_image_training_verified401_resume.2026-10-12_v4.json',
        previous_configuration=path,state_root='experiments/runs/fm_full25_image_training_verified401_resume_v4',
        saved401_resume=dict(job_configuration=config['ordered_original_jobs'][0],completed_epochs=401,remaining_epochs=1599,
            CPU_proof=archive['full401_CPU_proof'],archive=archive['immutable_full_state_archive']))
    config['source_receipts']+=[dict(path=p,sha256=old.sha(root/p)) for p in paths]
    config['original_failure_receipts']+=[dict(path=str(Path(latest)/name),sha256=old.sha(Path(latest)/name)) for name in ('terminal.json','stderr.log')]
    queue.verify_registration(config)
    exclusive(root/config['configuration_path'],config)
    print(json.dumps(dict(original_full_jobs=25,epochs_per_job=2000,verified_saved_epochs=401,
        remaining_first_job_epochs=1599,original_GPU_mutex_and_all_limits_unchanged=True)))


if __name__=='__main__':main()
