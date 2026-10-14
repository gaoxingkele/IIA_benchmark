"""Compare all original conditional GP400/600 seed pairs after full reviews.

All five joint data/training seeds and all80 trajectories are required. Reuse
the preregistered hierarchical paired bootstrap; never train, infer or tune.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
from datetime import datetime, timezone
from types import FunctionType
from scripts.flow_matching import audit_cfm_gp_five_seed_targets_v1 as previous

ROOT, read, sha, SEEDS = previous.ROOT, previous.read, previous.sha, previous.SEEDS


def identities():
    return [['cfm_ts_original__appendix_d__gp_gaussian_conditional_e'+str(epochs)+'__sine__seed'+str(seed)
        for epochs in [400,600]] for seed in SEEDS]


def select_complete_pairs(records, pairs):
    return FunctionType(previous.select_complete_pairs.__code__, dict(previous.__dict__, identities=identities))(records,pairs)


def validate_pair(jobs, parameters, seed):
    ignored={'id','epochs','expected_optimizer_updates','model_config','model_config_sha256','output_directory','artifact_directory'}
    if (len(jobs)!=2 or [j['epochs'] for j in jobs]!=[400,600]
            or [j['expected_optimizer_updates'] for j in jobs]!=[250000,375000]
            or any(j['seed']!=seed or j['test_trajectories']!=80 or j['test_points_per_trajectory']!=500
                or j['observed_dimensions']!=1 or j['variant']!='gp_gaussian_conditional' for j in jobs)
            or {k:v for k,v in jobs[0].items() if k not in ignored}!={k:v for k,v in jobs[1].items() if k not in ignored}
            or {k for k in parameters[0].keys()|parameters[1].keys() if parameters[0].get(k)!=parameters[1].get(k)}!={'epochs'}
            or [p['epochs'] for p in parameters]!=[400,600]
            or any(p['target_mode']!='gaussian_conditional' for p in parameters)):
        raise ValueError('Only the already registered full400/600 epoch/update budget may differ within each original data/model seed pair')


def verify_config(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full budget-pair analysis changed: '+pin['path'])
    runtime=read(ROOT/config['primary_runtime'])
    if (len(runtime['routes'])!=285 or runtime['original_scope']!=config['original_scope']
            or config['paired_canonical_ids']!=identities() or config['bootstrap_replicates']!=10000):
        raise ValueError('All original285 slots,45/681/273 scope and five fixed seed pairs required')
    byid={r['original_canonical_id']:r['job'] for r in runtime['routes']}
    for seed,pair in zip(SEEDS,identities()):
        jobs=[byid[i] for i in pair]
        validate_pair(jobs,[read(ROOT/j['model_config'])['parameters'] for j in jobs],seed)
    return runtime


def prefix_evidence(short, long):
    import numpy as np
    a,b=np.asarray(short,dtype=np.float64),np.asarray(long,dtype=np.float64)
    if a.shape!=(400,) or b.shape!=(600,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Complete finite400/600 epoch histories required')
    different=np.flatnonzero(a!=b[:400])
    return dict(first400_epoch_losses_bitwise_identical=bool(len(different)==0),
        first_differing_epoch_1based=int(different[0]+1) if len(different) else None,
        maximum_first400_epoch_loss_difference=float(np.max(np.abs(a-b[:400]))),
        matching_full_optimizer_state_or_intermediate_weights_claimed=False)


def audit_complete(config,runtime,snapshot):
    import numpy as np
    selected,missing=select_complete_pairs(snapshot['records'],config['paired_canonical_ids'])
    if missing:
        raise ValueError('Never publish a successful-subset five-seed budget result')
    errors=[[],[]]; pairs=[]
    for seed,pair in zip(SEEDS,selected):
        original_ids=original_truth=original_times=None;histories=[]
        for position,record in enumerate(pair):
            previous.cold.validate_closed_primary_record(runtime,record)
            if (record['epochs']!=[400,600][position] or record['optimizer_updates']!=[250000,375000][position]
                    or record['seed']!=seed or record['full_replay_shape']!=[80,500,1]):
                raise ValueError('Every complete original training budget and full held-out grid required')
            result=read(ROOT/record['result_path']);histories.append(result['training_losses'])
            with np.load(record['replay_arrays'],allow_pickle=False) as arrays:
                prediction,truth=arrays['predictions'],arrays['truth']
                if (prediction.shape!=(80,500,1) or truth.shape!=prediction.shape
                        or not np.isfinite(prediction).all() or not np.isfinite(truth).all()
                        or len(set(arrays['test_ids'].tolist()))!=80):
                    raise ValueError('Every finite paired original coordinate and whole trajectory required')
                if position==0:
                    original_ids=arrays['test_ids'].copy();original_truth=truth.copy();original_times=arrays['times'].copy()
                elif (not np.array_equal(original_ids,arrays['test_ids']) or not np.array_equal(original_truth,truth)
                        or not np.array_equal(original_times,arrays['times'])):
                    raise ValueError('Identical paired original held-out trajectories, truth and times required')
                value=np.square(prediction.astype(np.float64)-truth).mean(axis=(1,2))
                if not np.isclose(value.mean(),record['replayed_metrics']['MSE'],rtol=1e-12,atol=1e-12):
                    raise ValueError('Full saved MSE must equal all80 original trajectory errors')
                errors[position].append(value)
        pairs.append(dict(seed=seed,records=pair,trajectory_ids=original_ids.tolist(),
            actual_first400_training_prefix=prefix_evidence(*histories)))
    summary=previous.grouped_summary(*errors,SEEDS,config['bootstrap_seed'],config['bootstrap_replicates'])
    summary['conditional_GP400']=summary.pop('paper_literal')
    summary['conditional_GP600']=summary.pop('gaussian_conditional')
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=config['original_scope'],
        original_registered_CFM_models=285,complete_model_count=10,epochs_by_pair=[400,600],
        optimizer_updates_by_pair=[250000,375000],all_complete_prediction_coordinates_recomputed=400000,
        actual_native_and_independent_guarded_proofs_verified=True,pairs=pairs,**summary,
        paired_difference_direction='conditional GP600 minus conditional GP400',
        all_five_original_joint_seed_budget_pairs_complete=True,additional_optimizer_updates=0,
        additional_model_inference=0,test_based_recipe_selection=False,author_equivalence_certified=False,
        strict_TAB_result=False,all_original_experiments_complete=False,
        uncertainty_boundary='Five paired seeds jointly vary data/split and training RNG. Outer seed-pair/inner whole-trajectory bootstrap is descriptive at n=5. Loss-prefix equality does not certify identical intermediate weights/optimizer state or isolate a training-RNG-only effect.')


def main():
    values=dict(previous.__dict__,identities=identities,select_complete_pairs=select_complete_pairs,
        verify_config=verify_config,audit_complete=audit_complete)
    FunctionType(previous.main.__code__,values)()


if __name__=='__main__':main()
