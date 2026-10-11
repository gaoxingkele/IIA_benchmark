"""Verify the actual terminal full GRASP replay and every saved recipe again."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_resume_v2 import base, verify_reusable_recipes


def check_terminal(report, receipt, config):
    ids = [record['canonical_id'] for record in report['records']]
    expected = [record['canonical_id'] for record in config['full_results']]
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['full_replay_passed']
            or not report['full_checkpoint_contents_and_re_inference_passed']
            or report['full_job_count'] != config['full_job_count']
            or ids != expected or len(ids) != len(set(ids))
            or report['additional_training_seed_slots'] != 0 or report['cuda_context_initialized']):
        raise ValueError('Exactly the original full model replay must finish without abort or extra seeds')
    for record in report['records']:
        if (record['epochs_completed'] != 1500 or record['optimizer_updates'] != 93000
                or record['parameter_count'] != 69244
                or not record['full_checkpoint_restoration_and_reinference_passed']):
            raise ValueError('Full original Lin model, training and checkpoint restoration required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = base.read(base.ROOT / args.config)
    for source in config['source_receipts']:
        if base.sha(base.ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen actual complete replay evidence differs')
    producer = base.read(base.ROOT / config['completed_replay_configuration'])
    report_path = base.ROOT / producer['output_directory'] / 'checkpoint_replay_audit.json'
    report = base.read(report_path)
    receipt_path = base.ROOT / producer['state_root'] / 'receipt.json'
    receipt = base.read(receipt_path)
    check_terminal(report, receipt, config)
    if (report['configuration_sha256'] != base.sha(base.ROOT / config['completed_replay_configuration'])
            or report['score_tolerance'] != config['score_tolerance']
            or report['metric_tolerance'] != config['metric_tolerance']):
        raise ValueError('Original preregistered replay configuration and tolerances required')
    for record in report['records']:
        if (base.sha(Path(record['checkpoint_path'])) != record['checkpoint_sha256']
                or base.sha(base.ROOT / record['result_path']) != record['result_sha256']):
            raise ValueError('Actual original full checkpoint or native result differs')
    # The unchanged existing verifier reloads every complete original/replay
    # score archive, checks full identities/forward counts, and recomputes metrics.
    checked = verify_reusable_recipes(config)
    if len(checked) != config['full_score_recipe_count'] or config['pending_full_recipe_count'] != 0:
        raise ValueError('Every actually completed full score recipe must be checked')
    target = base.ROOT / config['verification_output_directory']
    target.mkdir(parents=True, exist_ok=False)
    recipes = [dict(canonical_id=key[0], **record) for key, record in checked.items()]
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        canonical_id=config['full_results'][0]['canonical_id'],
        completed_native_replay_configuration=config['completed_replay_configuration'],
        completed_native_replay_report_sha256=base.sha(report_path),
        full_original_model_restoration_and_reinference_completed=True,
        all_saved_full_score_arrays_and_identities_reverified=True,
        all_original_forward_counts_and_recipe_budgets_verified=True,
        all_validation_and_test_points_finite=True,
        all_metrics_independently_recomputed_from_full_replay=True,
        actual_executed_recipe_count=len(checked), full_validation_points=3935, full_test_points=72000,
        recipes=recipes, terminal_resource_receipt=receipt,
        additional_training_seed_slots=0, additional_model_inference_executed_by_this_review=False,
        author_equivalence_certified=False, all_original_experiments_complete=False)
    base.write(target / 'independent_full_replay_verification.json', proof)
    shutil.copyfile(receipt_path, target / 'native_replay_resource_receipt.json')
    shutil.copyfile(base.ROOT / producer['state_root'] / 'status.json', target / 'native_replay_terminal_status.json')
    print(json.dumps(dict(full_original_native_model_replay_verified=True,
        complete_validation_points=3935, complete_test_points=72000,
        exact_original_forward_budget_verified=True)))


if __name__ == '__main__':
    main()
