"""Audit full trained Spectral checkpoints without promoting aborted sampling.

This does not run a smaller model, change author sampling, or claim a result.
All five original trained checkpoints and terminal resource receipts are checked
before deciding which state can actually be restored for a full recovery.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha, write


def checkpoint_budget(checkpoint, updates, torch):
    if set(checkpoint) != {'step', 'model', 'ema', 'opt'} or checkpoint['step'] != updates:
        raise ValueError('Unchanged complete native checkpoint and final update required')
    for field in ('model', 'ema'):
        tensors=checkpoint[field]
        if not tensors or any(not torch.is_tensor(v) or not torch.isfinite(v).all().item()
                              for v in tensors.values()):
            raise ValueError('All complete finite original model and EMA tensors required')
    optimizer=checkpoint['opt']
    if not optimizer.get('param_groups') or not optimizer.get('state'):
        raise ValueError('Complete original optimizer state required')
    parameters=[p for group in optimizer['param_groups'] for p in group['params']]
    if len(parameters)!=len(set(parameters)) or set(parameters)!=set(optimizer['state']):
        raise ValueError('Every original optimizer parameter needs exactly one complete state')
    states=optimizer['state'].values()
    steps=[float(s['step']) for s in states if 'step' in s]
    if not steps or len(steps)!=len(optimizer['state']) or any(s!=updates for s in steps):
        raise ValueError('Every original optimizer parameter must reach the full update budget')
    for state in optimizer['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Every native optimizer tensor must be finite')
    return dict(full_checkpoint_training_step=checkpoint['step'],
        model_state_tensors=len(checkpoint['model']),ema_state_tensors=len(checkpoint['ema']),
        optimizer_parameter_states=len(steps),all_native_model_ema_optimizer_tensors_finite=True,
        complete_optimizer_update_budget_verified=True,
        RNG_state_saved=False,scheduler_state_saved=False,
        exact_post_training_sampling_RNG_resume_certified=False)


def audit(config):
    for source in config['source_receipts']:
        if sha(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen original failure/checkpoint evidence changed: '+source['path'])
    queue=read(ROOT/config['queue'])
    if len(queue['jobs'])!=70 or len({j['id'] for j in queue['jobs']})!=70:
        raise ValueError('All original fourteen paired five-seed generation cohorts required')
    jobs=[j for j in queue['jobs'] if j['id'] in config['failed_job_ids']]
    if (len(jobs)!=5 or sorted(j['seed'] for j in jobs)!=[42,43,44,45,46]
            or any(j['model']!='spectral_flow' or j['dataset']!='stocks'
                   or j['training_updates']!=10000 or j['generation_batch_size']!=2001
                   or j['generation_field_chunk_size'] is not None or j['data_shape']!=[3662,24,6]
                   for j in jobs)):
        raise ValueError('Every original full Stocks Spectral cohort job must remain unchanged')
    import torch
    rows=[]
    for job in jobs:
        output=ROOT/job['output_directory']
        receipt_path=ROOT/queue['state_root']/'receipts'/(job['id']+'.json')
        receipt=read(receipt_path)
        progress=read(output/'progress.json')
        if (receipt['id']!=job['id'] or receipt['exitcode']==0 or receipt['full_result_validated']
                or receipt['resource_guard_aborted'] is not True
                or progress!={'phase':'sampling','optimizer_updates':10000}
                or (output/'result.json').exists()):
            raise ValueError('Actual terminal full-training/sampling-resource failure required')
        checkpoint=Path(job['artifact_directory'])/'author_output/checkpoints/checkpoint-10.pt'
        native=torch.load(checkpoint,map_location='cpu',weights_only=True)
        budget=checkpoint_budget(native,job['training_updates'],torch)
        if sha(ROOT/job['frozen_samples'])!=job['data_sha256'] or sha(ROOT/job['reference_path'])!=job['reference_sha256']:
            raise ValueError('Original complete paired training/reference data changed')
        fake=Path(job['artifact_directory'])/'fake_data.npy'
        if fake.exists():
            raise ValueError('Sampling progressed beyond the frozen failure; review the new evidence')
        rows.append(dict(canonical_id=job['id'],seed=job['seed'],
            original_model_config=job['model_config'],original_author_yaml=job['author_yaml'],
            full_training_checkpoint_path=str(checkpoint),full_training_checkpoint_sha256=sha(checkpoint),
            original_training_updates=job['training_updates'],full_generation_batch_size=2001,
            original_reference_shape=job['data_shape'],original_unmodified_field_chunk_size=None,
            actual_original_progress=progress,actual_terminal_resource_receipt=receipt,
            actual_terminal_resource_receipt_sha256=sha(receipt_path),
            complete_generated_array_available=False,complete_four_evaluator_metrics_available=False,
            original_post_training_noise_state_available=False,
            full_original_training_and_sampling_required_for_exact_recovery=True,
            **budget))
    if torch.cuda.is_initialized():
        raise ValueError('Checkpoint inspection must not initialize a CUDA context')
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        full_original_generation_queue=config['queue'],full_original_registered_jobs=70,
        full_original_registered_cohorts=14,actual_failed_full_training_checkpoints_verified=len(rows),
        actual_finished_generation_experiments=0,records=rows,
        cuda_context_initialized=False,model_reinference_executed=False,
        author_source_or_scientific_sampling_budget_changed=False,
        additional_training_seed_slots=0,author_equivalence_certified=False,
        all_original_experiments_complete=False,
        boundary='All five full original Stocks Spectral training checkpoints contain10000updates plus finite model/EMA/optimizer states; full2001batch sampling resource-aborted before complete generated arrays or any of four full evaluators. No RNG or scheduler state was saved, so exact post-training sampling continuation is unproven. Preserve every failed attempt. Full sampling and evaluators remain required; full original training must be repeated unless the original RNG trajectory can be independently restored. This CPU-only checkpoint audit executes no model forwards and supplies no performance metric.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=read(ROOT/args.config)
    target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable actual full checkpoint inspection')
    proof=audit(config)
    target.mkdir(parents=True)
    write(target/'full_native_checkpoint_inspection.json',proof)
    write(target/'validation.json',dict(passed=True,
        source_receipts=config['source_receipts']+[dict(path=args.config,sha256=sha(ROOT/args.config))],
        outputs={'full_native_checkpoint_inspection.json':sha(target/'full_native_checkpoint_inspection.json')}))
    print(dict(full_original_checkpoints=proof['actual_failed_full_training_checkpoints_verified'],
        generated_experiments_complete=0,exact_sampling_RNG_resume_certified=False))


if __name__=='__main__':
    main()
