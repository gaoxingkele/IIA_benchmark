"""Frozen five-seed uncertainty and paired native-step sensitivity, never best-step selection."""
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import validate_queue

METRICS = ('image_AUROC', 'image_AP', 'image_F1_max', 'pixel_AUROC', 'pixel_AP', 'pixel_F1_max', 'pixel_AUPRO')
SEEDS = list(range(1103, 1108))
STEPS = [1, 10, 20, 50, 100, 200]
UNCERTAINTY = dict(statistical_unit='independent_original_training_seed', required_seeds=SEEDS,
    confidence=.95, interval='two_sided_Student_t_mean_unclipped', sample_std_ddof=1,
    category_aggregation='unweighted_mean_all15_original_categories', paired_reference_steps=[1, 50],
    category_variation='descriptive_heterogeneity_not_independent_training_replicates',
    select_best_step_using_test_labels=False, uncertainty_is_paper_significance_test=False)


def validate_configuration(config):
    verify_receipts(config['source_receipts'])
    queue = read(ROOT / config['score_queue']); verify_receipts(queue['source_receipts']); validate_queue(queue)
    if (config['uncertainty'] != UNCERTAINTY or config['metric_names'] != list(METRICS)
            or config['required_case_seed_evaluations'] != 30
            or queue['solver_facts']['native_step_scan'] != STEPS
            or config['original_scope'] != queue['original_scope']
            or config['author_equivalence_certified'] or config['strict_TAB_result']
            or config['all_original_experiments_complete']):
        raise ValueError('All30 full WT cases, five original seeds and fixed descriptive uncertainty required')
    return queue


def score_locations_ready(config, queue):
    # Only eligibility: heavy verification belongs to the separately guarded worker.
    return all((ROOT / case['output_directory'] / 'fully_verified_score_result.json').is_file() for case in queue['cases'])


def complete_summary(config):
    directory = ROOT / config['output_directory']
    path = directory / 'fully_verified_cohort_summary.json'
    if not path.exists():
        return False
    location = read(path)
    verify_receipts([location['candidate'], location['independent_review']])
    report = read(ROOT / location['independent_review']['path'])
    candidate = read(ROOT / location['candidate']['path'])
    if (not location['passed'] or not report['passed']
            or report['candidate_receipt'] != location['candidate']
            or candidate['config'] != config['configuration_path']
            or candidate['config_sha256'] != sha(ROOT / config['configuration_path'])
            or candidate['score_queue_sha256'] != sha(ROOT / config['score_queue'])
            or candidate['full_verified_case_seed_evaluations'] != 30
            or candidate['independent_original_training_seeds'] != SEEDS
            or candidate['native_step_grid'] != STEPS or candidate['uncertainty'] != UNCERTAINTY
            or candidate['additional_training_seed_slots'] != 0
            or candidate['author_equivalence_certified'] or candidate['strict_TAB_result']
            or candidate['all_original_experiments_complete']
            or len(candidate['full_source_case_records']) != 30
            or len(candidate['aggregate_metrics']) != 42 or len(candidate['per_category_metrics']) != 630
            or len(candidate['paired_step_differences']) != 84
            or report['full_verified_case_seed_evaluations'] != 30
            or report['independent_original_training_seeds'] != SEEDS
            or report['native_step_grid'] != STEPS or report['aggregate_metric_rows'] != 42
            or report['per_category_metric_rows'] != 630 or report['paired_metric_rows'] != 84
            or not report['all_source_checkpoint_data_and_full_score_proofs_verified']
            or not report['all_seed_uncertainty_and_pairing_independently_verified']
            or report['score_queue_sha256'] != sha(ROOT / config['score_queue'])
            or report['author_equivalence_certified'] or report['strict_TAB_result']
            or report['all_original_experiments_complete']):
        raise ValueError('Complete independently reviewed WT cohort summary required')
    queue = read(ROOT / config['score_queue'])
    from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import complete
    if {row['case_id'] for row in candidate['full_source_case_records']} != {case['id'] for case in queue['cases']}:
        raise ValueError('All30 original source score cases must remain bound')
    for record in candidate['full_source_case_records']:
        verify_receipts([record['score_completion_receipt'], record['full_score_report']])
    if not all(complete(case) for case in queue['cases']):
        raise ValueError('Every actual full model/data/score proof must remain complete')
    return True
