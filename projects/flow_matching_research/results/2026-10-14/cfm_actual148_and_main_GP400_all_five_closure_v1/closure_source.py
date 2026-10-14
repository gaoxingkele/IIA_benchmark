from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays
from scripts.flow_matching.audit_cfm_gp_conditional_five_seed_budgets_v1 import prefix_evidence

root,read,sha=cold.ROOT,cold.original.read,cold.streaming_sha
runtimepaths=['configs/runtime/fm_cfm_completion_replay_queue.v4.json',
    'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json','configs/runtime/fm_cfm_saved_score_full_replay.v1.json']
runtimes=[read(root/p) for p in runtimepaths]
records={};owners={};snapshots=[]
for runtime in runtimes:
    path=root/runtime['state_root']/'verified_full_model_records.json';raw=path.read_bytes()
    assert len(runtime['routes'])==285
    for record in json.loads(raw)['records']:
        identity=record['original_canonical_id']
        if identity in records:assert records[identity]==record
        else:records[identity]=record;owners[identity]=runtime
    snapshots.append(raw)
assert len(records)>=148
relative='projects/flow_matching_research/results/2026-10-14/cfm_actual148_and_main_GP400_all_five_closure_v1'
folder=root/relative;folder.mkdir(exist_ok=False)
for number,raw in enumerate(snapshots): (folder/f'frozen_full_model_snapshot_{number}.json').write_bytes(raw)
newids=['cfm_ts_original__appendix_d__gp_gaussian_conditional_e600__sine__seed1103',
    'cfm_ts_original__main_text__gp_gaussian_conditional_e400__sine__seed1106']
for identity in newids:
    record=records[identity];runtime=owners[identity]
    cold.validate_closed_primary_record(runtime,record)
    case=folder/identity;case.mkdir()
    nativepath=root/runtime['child_config_root']/(identity+'.native.json')
    reviewpath=root/runtime['child_config_root']/(identity+'.review.json')
    native,review=read(nativepath),read(reviewpath)
    sources=dict(original_result=root/record['result_path'],native_config=nativepath,
        independent_config=reviewpath,native_report=root/native['output_directory']/'checkpoint_replay_audit.json',
        independent_proof=root/review['output_proof'],native_guarded_receipt=root/native['state_root']/'receipt.json',
        independent_guarded_receipt=root/review['state_root']/'receipt.json')
    source_receipt=(root/'experiments/runs/fm_cfm_saved_score_queued_handoff_v2/receipt.json'
        if identity==newids[1] else root/'experiments/runs/fm_cfm_ts_original_v1/receipts'/(identity+'.json'))
    receipt=read(source_receipt)
    assert receipt['exit_code']==0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
    sources['original_guarded_scoring_receipt']=source_receipt
    for name,path in sources.items(): (case/(name+'.json')).write_bytes(path.read_bytes())
    result=read(root/record['result_path']);job=result['job']
    predictionpath=next(a['path'] for a in result['artifacts'] if Path(a['path']).name=='predictions.npz')
    with np.load(predictionpath,allow_pickle=False) as arrays,np.load(root/job['data_path'],allow_pickle=False) as data:
        metrics=audit_arrays(job,result,arrays,data)
        assert not set(data['train_ids'])&set(data['test_ids'])
    (case/'all_original_source_coordinates_recomputed.json').write_text(json.dumps(metrics,indent=2)+'\n')
    if identity==newids[1]:
        config=read(root/'configs/runtime/fm_cfm_saved_score_queued_handoff.v2.json')
        assert result['additional_optimizer_updates']==0 and result['restored_original_weights_bitwise_identical']
        assert result['training_seconds'] is None and receipt['additional_optimizer_updates']==0
        assert sha(config['checkpoint_path'])==config['checkpoint_sha256']==result['recovery_original_checkpoint_sha256']
        assert result['epochs_completed']==400 and result['optimizer_updates']==62400

errors=[];mainrecords=[];influence=[]
for seed in range(1103,1108):
    identity='cfm_ts_original__main_text__gp_gaussian_conditional_e400__sine__seed'+str(seed)
    record=records[identity];runtime=owners[identity];cold.validate_closed_primary_record(runtime,record)
    assert record['epochs']==400 and record['optimizer_updates']==62400 and record['full_replay_shape']==[20,500,1]
    with np.load(record['replay_arrays'],allow_pickle=False) as a:
        assert a['predictions'].shape==a['truth'].shape==(20,500,1)
        value=np.square(a['predictions'].astype(np.float64)-a['truth']).mean(axis=(1,2))
        assert np.isfinite(value).all() and len(set(a['test_ids'].tolist()))==20
        assert np.isclose(value.mean(),record['replayed_metrics']['MSE'],rtol=1e-12,atol=1e-12)
        top=int(np.argmax(value))
        influence.append(dict(seed=seed,largest_error_trajectory_id=int(a['test_ids'][top]),
            largest_trajectory_MSE=float(value[top]),largest_trajectory_fraction_of_seed_squared_error=float(value[top]/value.sum()),
            median_trajectory_MSE=float(np.median(value)),no_trajectory_removed=True))
        errors.append(value)
    mainrecords.append(record)
errors=np.asarray(errors);means=errors.mean(axis=1);rng=np.random.default_rng(20261014);bootstrap=[]
for start in range(0,10000,256):
    count=min(256,10000-start);outer=rng.integers(0,5,size=(count,5));inner=rng.integers(0,20,size=(count,5,20))
    bootstrap.extend(errors[outer[:,:,None],inner].mean(axis=(1,2)).tolist())
main=dict(captured_utc=datetime.now(timezone.utc).isoformat(),complete_original_models=5,
    joint_data_and_training_seeds=list(range(1103,1108)),epochs_per_model=400,optimizer_updates_per_model=62400,
    full_original_shape_per_model=[20,500,1],all_original_coordinates_recomputed=50000,
    per_joint_seed_MSE=means.tolist(),mean_MSE=float(means.mean()),joint_seed_sample_std=float(means.std(ddof=1)),
    joint_seed_SE=float(means.std(ddof=1)/np.sqrt(5)),
    hierarchical_seed_whole_trajectory_bootstrap_mean_CI95=np.quantile(bootstrap,[.025,.975]).tolist(),
    bootstrap_seed=20261014,bootstrap_replicates=10000,outer_unit='joint original data/split and model seed',
    inner_unit='whole original held-out trajectory within each joint seed',trajectory_error_concentration=influence,
    original_model_and_data_and_guarded_native_independent_proofs_verified=True,records=mainrecords,
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
    boundary='All five Main100-total/20-test/62400-update models, including restored seed1106. Separate from Appendix400-total/80-test/250000-update cohort. Only five outer groups; descriptive uncertainty, no outlier removal, recipe selection or causal bug attribution.')
(folder/'main_GP400_complete_all_five_original_models.json').write_text(json.dumps(main,indent=2)+'\n')

shortid='cfm_ts_original__appendix_d__gp_gaussian_conditional_e400__sine__seed1103'
pair=[records[shortid],records[newids[0]]];pairerrors=[];histories=[];reference=None
for position,record in enumerate(pair):
    cold.validate_closed_primary_record(owners[record['original_canonical_id']],record)
    assert record['epochs']==[400,600][position] and record['optimizer_updates']==[250000,375000][position]
    histories.append(read(root/record['result_path'])['training_losses'])
    with np.load(record['replay_arrays'],allow_pickle=False) as a:
        assert a['predictions'].shape==a['truth'].shape==(80,500,1)
        if reference is None:reference={k:a[k].copy() for k in ['truth','times','test_ids']}
        else:assert all(np.array_equal(reference[k],a[k]) for k in reference)
        pairerrors.append(np.square(a['predictions'].astype(np.float64)-a['truth']).mean(axis=(1,2)))
delta=pairerrors[1]-pairerrors[0];rng=np.random.default_rng(20261014)
samples=delta[rng.integers(0,80,size=(5000,80))].mean(axis=1)
comparison=dict(seed=1103,records=pair,MSE_GP400=float(pairerrors[0].mean()),MSE_GP600=float(pairerrors[1].mean()),
    paired_MSE_difference_600_minus_400=float(delta.mean()),
    paired_whole_trajectory_bootstrap_CI95=np.quantile(samples,[.025,.975]).tolist(),
    bootstrap_seed=20261014,bootstrap_replicates=5000,paired_whole_trajectories=80,
    all_original_pair_coordinates_recomputed=80000,training_prefix=prefix_evidence(*histories),
    complete_five_outer_seed_budget_comparison=False,additional_optimizer_updates=0,additional_model_inference=0,
    author_equivalence_certified=False,strict_TAB_result=False,
    boundary='One already registered seed pair only;80 trajectories do not become80 independent training seeds. Full five-seed comparison remains pending; no recipe selection.')
(folder/'Appendix_conditional_GP400_GP600_seed1103_complete_pair.json').write_text(json.dumps(comparison,indent=2)+'\n')
for name,path in [('completed_source600_review_window.json','experiments/runs/fm_cfm_recurring_ready_review_windows_v3/window_3_receipt.json'),
    ('completed_restored400_review_window.json','experiments/runs/fm_cfm_saved_score_queued_handoff_v2/post_score_full_review_window_receipt.json'),
    ('completed_restored400_controller.json','experiments/runs/fm_cfm_saved_score_queued_handoff_v2/status.json')]:
    (folder/name).write_bytes((root/path).read_bytes())
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),full_unique_canonical_models=len(records),
    required_original_models=285,deduplicated_full_records=list(records.values()),new_fully_closed_original_model_ids=newids,
    restored_original_GP400_weights_without_training=True,additional_optimizer_updates=0,additional_model_inference=0,
    actual_main_five_seed_GP400_cohort_complete=True,full_five_Appendix_GP400_GP600_cohort_comparison_complete=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),all_original_experiments_complete=False)
(folder/'actual148_full_unique_canonical_model_closure.json').write_text(json.dumps(closure,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
(folder/'README.md').write_text('148/285 unique original CFM models have guarded native plus independent full-grid certificates. New closures: original Appendix conditional GP600/Sine1103 (600 epochs/375000 updates/80x500), and restored Main conditional GP400/Sine1106 (original400 epochs/62400 updates/20x500; zero new fitting/updates). All five Main GP400 models now qualify; report joint seed and whole-trajectory uncertainty without removing high-error trajectories. One Appendix GP400/600 seed pair is separately analyzed; the full five-seed budget comparison still waits for the other600 models. Native replays of both new models were bitwise identical. Preserve original failed metadata evidence, original solver/grid/resource guards and all45/681/273 scope. Neither local reproducibility nor a close paper-number comparison certifies original-author equivalence or strict TAB.\n')
print(json.dumps(dict(unique=len(records),main_five_MSE_mean=main['mean_MSE'],main_five_STD=main['joint_seed_sample_std'],
    main_five_CI95=main['hierarchical_seed_whole_trajectory_bootstrap_mean_CI95'],
    single_seed_400_to_600=comparison['paired_MSE_difference_600_minus_400'],
    restored_seed1106_top_trajectory=influence[3]),indent=2))
