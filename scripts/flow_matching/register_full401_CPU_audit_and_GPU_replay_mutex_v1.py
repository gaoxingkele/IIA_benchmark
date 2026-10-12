"""Freeze full401 CPU state audit and shared-GPU correction for all49 RNG gates."""
import copy
import json
from pathlib import Path

import psutil

from scripts.flow_matching import audit_full401_image_checkpoint_CPU_v1 as audit
from scripts.flow_matching import run_shcl_training_rng_queue_v3 as gpu_queue
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def pins(paths): return [dict(path=path, sha256=audit.sha(audit.ROOT/path)) for path in paths]


def main():
    root = audit.ROOT
    full = 'configs/reproducibility/fm_full_image_training_execution_recovery.2026-10-11_v3.json'
    original = audit.read(root/full)
    job_path = original['ordered_original_jobs'][0]; job = audit.read(root/job_path)
    checkpoint = (root/job['output_directory']/'resume.pt').resolve()
    progress = audit.read(root/job['output_directory']/'progress.json')
    if progress['completed_epochs'] != 401 or progress['global_steps'] != 45714:
        raise ValueError('Actual complete401epoch original checkpoint and sampler update history required')
    reference_path = 'configs/reproducibility/fm_rcgan_seed1104_full416_state_recovery.2026-10-11_v2.json'
    reference = audit.read(root/reference_path)
    paths = [full, job_path, reference_path,
        'scripts/flow_matching/audit_full401_image_checkpoint_CPU_v1.py',
        'scripts/flow_matching/run_full401_image_checkpoint_CPU_audit_v1.py',
        'scripts/flow_matching/register_full401_CPU_audit_and_GPU_replay_mutex_v1.py',
        'tests/test_fm_full401_CPU_audit_and_GPU_replay_mutex_v1.py',
        'scripts/flow_matching/run_full_image_training_recovery_queue_v3.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py', 'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    config = dict(original_full25_configuration=full, job_configuration=job_path,
        completed_epochs=401, required_original_epochs=2000, original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        checkpoint_receipt=dict(path=str(checkpoint),sha256=audit.sha(checkpoint)),
        reference_readonly_resource_configuration=reference_path, settings=reference['settings'],
        resource_lock='experiments/runs/fm_readonly_checkpoint_audit_cpu.lock', python=original['python'],
        state_root='experiments/runs/fm_full401_image_checkpoint_CPU_audit_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/full401_CPU_state_and_GPU_replay_mutex_v1/native_CPU_state',
        source_receipts=pins(paths),
        boundary='Full45M native model/StableAdamW and all401 actual epoch histories/receipts restored on CPU. '
            'Verify full CPU/Python/NumPy/order RNG; retain original CUDA generator bytes. No CUDA context or '
            'GPU state restoration, training, inference, or performance credit. Entire20FM/5WT full2000epoch '
            'cohort and full native GPU restore/continuation obligations remain.')
    audit.verify(config)
    exclusive(root/'configs/runtime/fm_full401_image_checkpoint_CPU_audit.v1.json',config)

    old_path='configs/runtime/fm_shcl_all49_training_rng_replay.v2.json'
    old=audit.read(root/old_path)
    state=audit.read(root/old['state_root']/'status.json')
    actor=psutil.Process(state['pid'])
    identity=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline())
    if actor.create_time()!=state['create_time'] or actor.cmdline()!=state['command']:
        raise ValueError('Exact live old idle replay owner required')
    corrected=copy.deepcopy(old)
    corrected.update(runtime_configuration='configs/runtime/fm_shcl_all49_training_rng_replay.v3.json',
        GPU_mutex_reference_configuration=full,GPU_resource_lock=original['resource_lock'],
        state_root='experiments/runs/fm_shcl_all49_training_rng_replay_v3',
        full_training_output_root=old['full_training_output_root'].replace('_v2','_v3'),
        handoff=dict(previous_owner=identity,previous_state_path=old['state_root']+'/status.json',
            previous_configuration_sha256=audit.sha(root/old_path)))
    extra=[old_path,full,'scripts/flow_matching/combined_readonly_gpu_slot_v1.py',
        'scripts/flow_matching/run_shcl_training_rng_queue_v3.py','scripts/flow_matching/register_full401_CPU_audit_and_GPU_replay_mutex_v1.py',
        'tests/test_fm_combined_readonly_gpu_slot_v1.py','tests/test_fm_full401_CPU_audit_and_GPU_replay_mutex_v1.py']
    fixture=(root/old['fixture_output_root']/'full_checkpoint_rng_restore.json').relative_to(root).as_posix()
    corrected['source_receipts']+=pins(extra+[fixture])
    gpu_queue.previous.original.job_api.verify(corrected)
    gpu_queue.verify_gpu_mutex(corrected)
    if not gpu_queue.safe_phase(corrected,state,[dict(name=p.name(),exe=p.exe()) for p in actor.children(recursive=True)]):
        raise ValueError('Only genuinely idle source-checkpoint waiting controller may be handed off')
    exclusive(root/corrected['runtime_configuration'],corrected)
    print(json.dumps(dict(full_saved_CPU_checkpoint_epochs=401,remaining_original_training_epochs=1599,
        complete_original_image_jobs=25,original_SHCL_training_RNG_jobs=49,shared_GPU_mutex_added=True)))


if __name__=='__main__': main()
