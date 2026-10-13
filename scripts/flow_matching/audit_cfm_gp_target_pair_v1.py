"""Compare already complete paired GP target variants on every original grid.

Bootstrap whole held-out trajectories, conditional on these trained models.
Never infer, train, tune on test labels or substitute this snapshot for five seeds.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT,read,sha=cold.ROOT,cold.original.read,cold.streaming_sha


def paired_summary(literal,conditional,ids,seed,repeats):
    import numpy as np
    a,b=np.asarray(literal,dtype=np.float64),np.asarray(conditional,dtype=np.float64)
    if (a.ndim!=1 or b.shape!=a.shape or len(a)!=len(ids) or len(set(ids))!=len(ids)
            or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any() or repeats<100):
        raise ValueError('Distinct complete trajectory groups and finite errors required')
    delta=b-a;rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(ids),size=(repeats,len(ids)))
    means=delta[sampled].mean(axis=1)
    return dict(held_out_trajectory_groups=len(ids),literal_MSE=float(a.mean()),conditional_MSE=float(b.mean()),
        paired_MSE_difference_conditional_minus_literal=float(delta.mean()),
        paired_trajectory_bootstrap_ci95=[float(x) for x in np.quantile(means,[.025,.975])],
        conditional_lower_error_trajectory_fraction=float(np.mean(delta<0)),
        bootstrap_seed=seed,bootstrap_replicates=repeats,bootstrap_unit='whole held-out trajectory',
        training_seed_uncertainty_estimated=False,confidence_boundary='Conditional on this trained seed pair; not a five-training-seed confidence interval.')


def audit(config):
    import numpy as np
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen paired snapshot changed: '+pin['path'])
    runtime=read(ROOT/config['replay_runtime']);snapshot=read(ROOT/config['closed_record_snapshot'])
    byid={r['original_canonical_id']:r for r in snapshot['records']}
    records=[byid[i] for i in config['paired_canonical_ids']]
    jobs=[next(r['job'] for r in runtime['routes'] if r['original_canonical_id']==i) for i in config['paired_canonical_ids']]
    if (any(j['epochs']!=400 or j['expected_optimizer_updates']!=250000 or j['test_trajectories']!=80 for j in jobs)
            or jobs[0]['seed']!=jobs[1]['seed'] or jobs[0]['data_sha256']!=jobs[1]['data_sha256']
            or jobs[0]['model_overrides']!=jobs[1]['model_overrides']):raise ValueError('Exact full original paired data, seed and budgets required')
    parameters=[read(ROOT/j['model_config'])['parameters'] for j in jobs]
    if [k for k in parameters[0] if parameters[0][k]!=parameters[1][k]]!=['target_mode']:
        raise ValueError('Source model recipes may differ only in original target construction')
    if [p['target_mode'] for p in parameters]!=['paper_literal','gaussian_conditional']:
        raise ValueError('Frozen literal/conditional comparison ordering required')
    verify=FunctionType(cold.original.verify_record.__code__,dict(cold.original.__dict__,sha=sha))
    errors=[];identities=None;metrics=[]
    for record in records:
        cold.validate_closed_primary_record(runtime,record)
        nativepath=ROOT/runtime['child_config_root']/(record['original_canonical_id']+'.native.json')
        native=read(nativepath);canonical=next(r for r in native['full_results'] if r['original_canonical_id']==record['original_canonical_id'])
        assert verify(runtime,canonical,record)==record
        with np.load(record['replay_arrays'],allow_pickle=False) as arrays:
            if arrays['predictions'].shape!=(80,500,1):raise ValueError('All80 complete500-point original grids required')
            if identities is None:identities=arrays['test_ids'].copy()
            elif not np.array_equal(identities,arrays['test_ids']):raise ValueError('Paired complete trajectory identities differ')
            errors.append(np.square(arrays['predictions'].astype(np.float64)-arrays['truth']).mean(axis=(1,2)))
            metrics.append(record['replayed_metrics'])
    summary=paired_summary(*errors,identities.tolist(),config['bootstrap_seed'],config['bootstrap_replicates'])
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),paired_canonical_ids=config['paired_canonical_ids'],
        original_scope=runtime['original_scope'],original_registered_CFM_slots=len(runtime['routes']),
        training_seed=jobs[0]['seed'],training_epoch_budget=400,optimizer_updates_per_model=250000,
        source_model_parameter_difference='target_mode only',same_full_original_data_sha256=jobs[0]['data_sha256'],
        both_native_and_independent_guarded_proof_chains_verified=True,checked_full_prediction_coordinates=80000,
        original_model_metrics=metrics,**summary,per_trajectory_ids=identities.tolist(),
        per_trajectory_literal_MSE=errors[0].tolist(),per_trajectory_conditional_MSE=errors[1].tolist(),
        required_original_training_seeds=[1103,1104,1105,1106,1107],completed_snapshot_training_seed_pairs=1,
        full_five_seed_method_comparison_complete=False,test_based_recipe_selection=False,
        additional_model_inference=0,additional_optimizer_updates=0,author_equivalence_certified=False,
        strict_TAB_result=False,all_original_experiments_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(ROOT/args.config);report=audit(config)
    with (ROOT/config['output_report']).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print({k:report[k] for k in ['literal_MSE','conditional_MSE','paired_MSE_difference_conditional_minus_literal','paired_trajectory_bootstrap_ci95','conditional_lower_error_trajectory_fraction']})


if __name__=='__main__':main()
