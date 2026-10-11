"""Verify full original generation arrays/checkpoint and evaluator statistics.

Learned evaluators are not independently retrained by this CPU-only audit.
Evaluator repetitions must never be counted as independent trained models.
"""
import argparse
from datetime import datetime,timezone
import math
from pathlib import Path
import statistics

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write

METRICS=('context_fid','correlational','discriminative','predictive')


def evaluator_statistics(metrics):
    from scipy.stats import t
    if set(metrics)!=set(METRICS):
        raise ValueError('Every original full native evaluator required')
    rows={}
    for key,item in metrics.items():
        raw=item['raw_repeats']
        if len(raw)!=5 or not all(math.isfinite(v) and v>=0 for v in raw):
            raise ValueError('Every complete finite five-repeat evaluator value required')
        mean=statistics.mean(raw)
        std=statistics.stdev(raw)
        se=std/math.sqrt(5)
        margin=float(t.ppf(.975,4))*se
        independently=dict(mean=mean,std=std,se=se,ci95_low=mean-margin,ci95_high=mean+margin,
            author_display_halfwidth=margin)
        for name,value in independently.items():
            if not math.isclose(item[name],value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Original complete evaluator statistic differs: '+key+'/'+name)
        rows[key]=dict(raw_repeats=raw,**independently,evaluator_repeats=5,
            independent_training_models=1,planned_original_training_models=5,
            between_training_seed_STD=None,between_training_seed_CI95=None,
            uncertainty_scope='five evaluator runs for one complete native trained model')
    return rows


def audit(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen completed native generation source/artifact changed: '+pin['path'])
    from scripts.flow_matching.run_spectral_original_job import verify,complete,fingerprint
    import numpy as np
    import torch
    from iia_benchmark.models.fm_spectral_author import build_author_model
    queue=read(ROOT/config['queue'])
    verify(queue)
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if not complete(job):
        raise ValueError('Entire original native training/sampling/four-evaluator result must verify')
    result=read(ROOT/job['output_directory']/'result.json')
    receipt=read(ROOT/config['terminal_receipt'])
    if (receipt['id']!=job['id'] or receipt['exitcode']!=0 or not receipt['full_result_validated']
            or receipt['resource_guard_aborted'] or result['job']!=job
            or result['job_sha256']!=fingerprint(job) or result['benchmark_smoke']
            or result['optimizer_updates']!=job['training_updates']
            or result['gradient_accumulate_every']!=job['gradient_accumulate_every']
            or result['training_minibatches']!=job['training_updates']*job['gradient_accumulate_every']
            or result['evaluation_repeats']!=5):
        raise ValueError('Actual terminal success and full original scientific budgets required')
    configuration=read(ROOT/job['model_config'])['parameters']
    model=build_author_model(ROOT/configuration['source_root'],configuration['model_target'],configuration['parameters'])
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=job['expected_parameters'] or parameters!=result['parameter_count']:
        raise ValueError('Full actual original author architecture/learned positions required')
    artifact={a['role']:a for a in result['artifacts']}
    checkpoint=torch.load(artifact['checkpoint']['path'],map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True)
    if any(not torch.equal(value,checkpoint['model'][key]) for key,value in model.state_dict().items()):
        raise ValueError('Every complete native model tensor must restore bitwise')
    for field in ['model','ema']:
        if any(not torch.is_tensor(v) or not torch.isfinite(v).all().item() for v in checkpoint[field].values()):
            raise ValueError('Every complete trained model/EMA tensor must be finite')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all().item()
           for state in checkpoint['opt']['state'].values() for v in state.values()):
        raise ValueError('Every original optimizer state tensor must be finite')
    generated=np.load(artifact['generated']['path'],allow_pickle=False)
    reference=np.load(artifact['reference']['path'],allow_pickle=False)
    original_reference=np.load(ROOT/job['reference_path'],allow_pickle=False)
    full_generated=job['generation_batch_size']*(job['data_shape'][0]//job['generation_batch_size']+1)
    if (list(generated.shape)!=[full_generated,*job['data_shape'][1:]]
            or list(reference.shape)!=job['data_shape'] or not np.array_equal(reference,original_reference)
            or result['generated_samples']!=full_generated or result['evaluated_reference_samples']!=len(reference)
            or not np.isfinite(generated).all() or not np.isfinite(reference).all()):
        raise ValueError('Every complete original reference/generated array and row required')
    metrics=evaluator_statistics(result['metrics'])
    if torch.cuda.is_initialized():
        raise ValueError('Read-only completed artifact audit must remain CPU-only')
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,canonical_id=job['id'],seed=job['seed'],
        original_full_registered_generation_jobs=len(queue['jobs']),actual_full_native_model_count=1,
        planned_original_training_seed_count=5,actual_model_parameters=parameters,
        paper_reported_parameters=job['paper_reported_parameters'],
        original_optimizer_updates=job['training_updates'],original_gradient_accumulation=job['gradient_accumulate_every'],
        original_training_minibatches=result['training_minibatches'],complete_reference_shape=list(reference.shape),
        complete_generated_shape=list(generated.shape),evaluated_generated_rows=len(reference),
        actual_full_original_checkpoint_tensors_restored_bitwise=True,
        complete_original_reference_array_bitwise_verified=True,all_original_generated_arrays_finite=True,
        all_four_five_repeat_evaluator_statistics_independently_recomputed=True,
        learned_evaluators_independently_retrained=False,generative_model_reinferred_by_this_audit=False,
        metrics=metrics,original_phase_seconds={k:result[k] for k in ['training_seconds','sampling_seconds','evaluation_seconds']},
        original_result_sha256=sha(ROOT/job['output_directory']/'result.json'),
        actual_terminal_receipt_sha256=sha(ROOT/config['terminal_receipt']),artifacts=result['artifacts'],
        cuda_context_initialized=False,additional_training_seed_slots=0,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary='Actual full original native Diffusion-TS Stocks training/sampling and all four five-repeat evaluators completed. All complete model/EMA/optimizer tensors and original/generated arrays independently inspected in native environment, and every evaluator statistic independently recomputed. Five evaluator repetitions describe one trained model, not five training seeds. This audit retrains no learned evaluator and re-generates no model output. Original distribution reconstruction/all-source scaling and seeded-evaluator runtime boundaries remain; no held-out future-generalization, author or TAB equivalence claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=read(ROOT/args.config)
    target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable full generation review')
    proof=audit(config)
    target.mkdir(parents=True)
    write(target/'independent_full_native_artifact_audit.json',proof)
    write(target/'validation.json',dict(passed=True,
        source_receipts=config['source_receipts']+[dict(path=args.config,sha256=sha(ROOT/args.config))],
        outputs={'independent_full_native_artifact_audit.json':sha(target/'independent_full_native_artifact_audit.json')}))
    print(dict(passed=True,training_models=1,required_training_models=5,
        full_generated_shape=proof['complete_generated_shape'],metrics={k:v['mean'] for k,v in proof['metrics'].items()}))


if __name__=='__main__':
    main()
