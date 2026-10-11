"""Freeze the completed original five-seed BRITS cohort and v40-to-v41 delta."""
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new


def main():
    previous = 'configs/reproducibility/fm_brits_four_seed_publication.2026-10-11_v39_v2.json'
    old = read(ROOT / previous)
    verify_receipts(old['source_receipts'])
    replay_target = 'projects/flow_matching_research/results/2026-10-11/brits_electricity_seed30_full_replay_v6'
    verification = read(ROOT / replay_target / 'independent_full_replay_verification.json')
    if (verification['id'] != 'saits_paper_brits_electricity_author_missing0.1_seed30'
            or not verification['actual_full_checkpoint_reinference_completed']
            or not verification['full_original_truth_masks_units_groups_equal']
            or verification['months'] != 10 or verification['held_out_coordinates'] != 1020954):
        raise ValueError('Independent complete original seed30 verification required')
    target = 'projects/flow_matching_research/results/2026-10-11/brits_five_seed_metric_increment_v41'
    config = dict(
        base_publication='projects/flow_matching_research/results/2026-10-11/all_algorithm_dataset_metrics_v40',
        output_directory='projects/flow_matching_research/results/2026-10-11/all_algorithm_dataset_metrics_v41',
        full_model_proofs=old['full_model_proofs'] + [replay_target + '/checkpoint_replay_audit.json'],
        full_model_guard_receipts=old['full_model_guard_receipts'] + [replay_target + '/replay_resource_receipt.json'],
        original_training_seeds=[26, 27, 28, 29, 30], original_scope=old['original_scope'],
        additional_training_seed_slots=0, cohort_summary_path=target + '/cohort_metric_summary.json',
        independent_review_path=target + '/independent_full_publication_review.json')
    paths = [
        'scripts/flow_matching/register_brits_five_seed_publication_v41.py',
        'scripts/flow_matching/publish_brits_five_seed_listing_v41.py',
        'scripts/flow_matching/review_brits_five_seed_publication_v41.py',
        'tests/test_fm_brits_five_seed_publication_v41.py',
        'scripts/flow_matching/publish_brits_two_seed_listing_v27.py',
        'scripts/flow_matching/review_brits_two_seed_publication_v27.py',
        'scripts/flow_matching/verify_brits_electricity_full_replay_v3.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'configs/reproducibility/fm_brits_electricity_seed30_full_replay_verification.2026-10-11_v6.json',
        replay_target + '/independent_full_replay_verification.json', previous,
        config['base_publication'] + '/all_algorithm_dataset_metrics.csv',
        config['base_publication'] + '/publication_validation.json',
        *config['full_model_proofs'], *config['full_model_guard_receipts']]
    config['source_receipts'] = [dict(path=p, sha256=sha(ROOT / p)) for p in paths]
    path = 'configs/reproducibility/fm_brits_five_seed_publication.2026-10-11_v41.json'
    write_new(ROOT / path, config)
    print(dict(frozen_original_training_seeds=config['original_training_seeds'],
               registered_slots=8618, allowed_metric_changes=3))


if __name__ == '__main__':
    main()
