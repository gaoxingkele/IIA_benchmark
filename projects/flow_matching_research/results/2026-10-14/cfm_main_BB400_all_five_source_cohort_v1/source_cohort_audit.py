from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, independent_stats
from scripts.flow_matching.audit_cfm_gp_seed_dispersion_v1 import dispersion
from scripts.flow_matching.run_cfm_completion_replay_queue_v1 import check_binding

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
original_path = root / 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
recovery_path = root / 'configs/experiments/fm_cfm_metadata_entry_full_retry_queue.v2.json'
protocol_path = root / 'configs/reproducibility/cfm_ts_original_protocol.v1.json'
original, recovery, protocol = read(original_path), read(recovery_path), read(protocol_path)
canon = {j['id']: j for j in original['jobs']}
retries = {j['recovery_original_job_id']: j for j in recovery['jobs'] if j['id'] in recovery['recovery_job_ids']}
ids = ['cfm_ts_original__main_text__bb_paper_literal_e400__sine__seed' + str(seed) for seed in range(1103, 1108)]
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_main_BB400_all_five_source_cohort_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
rows, pins = [], [original_path, recovery_path, protocol_path, root / protocol['paper_pdf'],
                 root / 'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
                 root / 'scripts/flow_matching/audit_cfm_gp_seed_dispersion_v1.py',
                 root / 'scripts/flow_matching/run_cfm_completion_replay_queue_v1.py']
for identity in ids:
    job = retries.get(identity, canon[identity])
    check_binding(canon[identity], job)
    assert job['epochs'] == 400 and job['expected_optimizer_updates'] == 6400 and job['test_trajectories'] == 20
    queue = recovery if job['id'] != identity else original
    resultpath = root / job['output_directory'] / 'result.json'
    receiptpath = root / queue['state_root'] / 'receipts' / (job['id'] + '.json')
    result, receipt = read(resultpath), read(receiptpath)
    assert receipt['exit_code'] == 0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
    assert sha(root / job['data_path']) == job['data_sha256'] and sha(root / job['model_config']) == job['model_config_sha256']
    for artifact in result['artifacts']: assert sha(artifact['path']) == artifact['sha256']
    score = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
    with np.load(score, allow_pickle=False) as arrays, np.load(root / job['data_path'], allow_pickle=False) as data:
        metrics = audit_arrays(job, result, arrays, data)
        assert not set(data['train_ids']) & set(data['test_ids'])
        detail = dispersion(arrays['per_trajectory_MSE'], arrays['test_ids'], arrays['initial_states'],
                            arrays['times'], arrays['predictions'], arrays['truth'])
        limits = dict(min=float(data['observed_states'][data['train_ids']].min()),
                      max=float(data['observed_states'][data['train_ids']].max()))
        for item in detail['largest_error_trajectories']:
            item['initial_inside_observed_training_state_range'] = all(limits['min'] <= x <= limits['max'] for x in item['initial_state'])
        groups = dict(training_trajectory_groups=len(data['train_ids']), held_out_trajectory_groups=len(data['test_ids']), overlap=0)
    rows.append(dict(canonical_id=identity, execution_id=job['id'], seed=job['seed'],
        full_epochs=400, optimizer_updates=6400, full_prediction_shape=[20, 500, 1],
        original_source_successful_terminal_receipt_verified=True,
        whole_original_arrays_metrics_and_artifact_hashes_verified=True,
        metrics=metrics, split=groups, full_error_dispersion=detail, observed_training_state_range=limits,
        result_path=resultpath.relative_to(root).as_posix(), result_sha256=sha(resultpath),
        checkpoint_and_prediction_artifact_hashes=result['artifacts'],
        guarded_native_plus_independent_proof_awarded_by_this_readonly_audit=False))
    pins += [resultpath, receiptpath, root / job['data_path'], root / job['model_config']]
    pins += [Path(a['path']) for a in result['artifacts']]
summary = {metric: independent_stats([row['metrics'][metric] for row in rows]) for metric in ['MSE', 'MAE', 'RMSE']}
claim = next(c for c in protocol['paper_claims'] if c['method'] == 'bb' and c['dataset'] == 'sine')
largest = sorted(rows, key=lambda row: row['metrics']['MSE'], reverse=True)[:2]
paper_review = dict(paper_path=protocol['paper_pdf'], paper_sha256=sha(root / protocol['paper_pdf']),
    visual_source_pages_reviewed=[3, 4], table_label='Table 1', method='CFM-BB', task='Sine',
    reported_MSE_mean=claim['mean'], reported_MSE_std=claim['std'], reported_runs=claim['runs'],
    stated_training_epochs_range=[400, 600], table_row_specific_epoch_budget_identified=False,
    main_text_trajectory_count=100, main_text_batch_size=258, main_text_train_test_fraction=[.8, .2],
    mean_difference_local400_minus_paper=float(summary['MSE']['mean'] - claim['mean']),
    numeric_proximity_does_not_certify_author_equivalence=True)
report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), required_canonical_ids=ids,
    full_source_training_and_scoring_seed_count=5, required_original_seed_count=5,
    full_original_source_training_cohort_complete=True, prediction_coordinates_recomputed=50000,
    metrics_by_original_joint_data_and_training_seed=rows, joint_data_and_training_seed_metric_summary=summary,
    two_largest_error_seed_ids=[row['seed'] for row in largest],
    two_largest_error_seed_fraction_of_equal_weight_cohort_MSE=float(sum(row['metrics']['MSE'] for row in largest) / sum(row['metrics']['MSE'] for row in rows)),
    statistical_interval='Two-sided Student-t df4 interval of five joint data-generation/split and training-seed values, descriptive at n5; not isolated training RNG variation.',
    paper_table_and_training_budget_review=paper_review,
    no_failed_or_missing_seed_removed=True, no_trajectory_removed=True,
    additional_optimizer_updates=0, additional_model_inference=0,
    all_five_guarded_model_replays_and_independent_proofs_certified=False,
    source_training_cohort_complete_is_not_final_model_replay_completion=True,
    causal_mechanism_of_error_dispersion_certified=False, test_based_recipe_selection=False,
    author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False)
pins = list(dict.fromkeys(pins))
config = dict(required_canonical_ids=ids, exact_source_execution_ids=[row['execution_id'] for row in rows],
    citation=protocol['citation'], output_report=relative + '/all_five_full_BB400_original_source_metrics_and_dispersion.json',
    source_receipts=[dict(path=p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p), sha256=sha(p)) for p in pins],
    boundary='All five original main-text BB400 full training/scoring sources, preserved aliases for two unsaved failures, every20x500 grid and all data groups. No extra seed, subset mean, model inference or relaxed guarded replay gate. Local vs Table1 comparison remains unproved author equivalence.')
with (root / 'configs/reproducibility/fm_cfm_main_BB400_all_five_source_metrics.2026-10-14_v1.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
(root / config['output_report']).write_text(json.dumps(report, indent=2) + '\n')
(folder / 'paper_table_and_budget_visual_review.json').write_text(json.dumps(paper_review, indent=2) + '\n')
(folder / 'source_cohort_audit.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps({'all_five_source_metric_summary': summary, 'largest_seed_fraction': report['two_largest_error_seed_fraction_of_equal_weight_cohort_MSE'],
    'seed_concentration': [dict(seed=row['seed'], full_MSE=row['metrics']['MSE'], top1_fraction=row['full_error_dispersion']['top1_squared_error_fraction'],
                                median_trajectory_MSE=row['full_error_dispersion']['trajectory_MSE_median']) for row in rows], 'paper_review':paper_review}, indent=2))
