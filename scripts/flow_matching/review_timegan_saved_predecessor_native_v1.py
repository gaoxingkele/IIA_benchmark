"""Independently bind the actual native replay proof to every saved full array."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.flow_matching.audit_timegan_saved_predecessor_native_v1 import original, verify


def review(config):
    import numpy as np
    queue, model, job, checkpoints, records = verify(config)
    receipt_path = original.ROOT / config['state_root'] / 'receipt.json'
    receipt = original.read(receipt_path)
    proof_path = original.ROOT / config['output_root'] / job['id'] / 'saved_predecessor_native_replay.json'
    proof = original.read(proof_path)
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] is not False
            or receipt['full_saved_predecessor_native_replay_passed'] is not True
            or receipt['report_sha256'] != original.sha(proof_path)
            or proof['canonical_job'] != job or proof['full_original_data_shape'] != [3661,24,6]
            or proof['passed'] is not True or proof['actual_saved_checkpoint_restores'] != 13
            or proof['actual_saved_trained_evaluator_replays'] != 10
            or proof['original_source_protocol_iterations_reconstructed'] != 20000
            or proof['training_executed'] is not False or proof['numerical_optimizer_updates_executed'] != 0
            or proof['experiment_complete'] is not False or proof['new_benchmark_results'] != 0
            or proof['final_full45_checkpoint_restores_still_required'] != 1035
            or proof['final_full45_trained_evaluator_replays_still_required'] != 900):
        raise ValueError('Actual successful additional saved-stage proof and all final obligations required')
    stage_checks = []
    for row, checkpoint in zip(proof['stage_checkpoint_restores'], checkpoints[:3]):
        if row['tag'] != checkpoint['tag'] or row['all_native_variables_restored_bitwise'] is not True:
            raise ValueError('Every complete actual native saved generator checkpoint required')
        stage_checks.append(dict(tag=row['tag'], verified_native_variable_count=row['full_native_variable_count']))
    if len(stage_checks) != 3:
        raise ValueError('Exactly all three native generator stages required')
    with np.load(str(original.ROOT / job['output_directory'] / 'full_native_generation_feed.npz'), allow_pickle=False) as feed:
        actual = hashlib.sha256(feed['normalized_generated'].tobytes()).hexdigest()
        if actual != proof['generation_array_sha256']:
            raise ValueError('Actual entire normalized generation must match native re-inference digest')
    rows = []
    if len(proof['evaluators']) != 10:
        raise ValueError('All ten saved complete native evaluators required')
    for index, (row, record) in enumerate(zip(proof['evaluators'], records)):
        if (row['kind'] != 'discriminative' or row['repeat'] != index
                or row['original_graph_seed'] != record['graph_seed'] or row['score'] != record['score']
                or row['complete_test_windows'] != 1466 or row['source_protocol_iterations_reconstructed'] != 2000
                or row['numerical_optimizer_updates_executed'] != 0
                or any(row[k] is not True for k in ('all_native_variables_restored_bitwise',
                    'full_native_predictions_bitwise_verified', 'full_native_score_bitwise_verified',
                    'full_post_evaluator_NumPy_state_bitwise_verified', 'no_model_or_optimizer_variables_changed'))):
            raise ValueError('Every exact original complete prediction, score, RNG and untouched optimizer proof required')
        archive_path = original.ROOT / job['output_directory'] / ('discriminative_%02d_full_test.npz' % index)
        with np.load(str(archive_path), allow_pickle=False) as archive:
            arrays = [archive[k] for k in ('real_probabilities','synthetic_probabilities')]
            if any(v.shape != (733,1) or not np.isfinite(v).all() for v in arrays):
                raise ValueError('All full finite original test probabilities required')
            digests = [hashlib.sha256(v.tobytes()).hexdigest() for v in arrays]
        if digests != row['full_prediction_array_sha256']:
            raise ValueError('Every actual entire native prediction must agree with saved-array hash')
        rows.append(dict(repeat=index, original_seed=record['graph_seed'], complete_test_points=1466,
            full_native_variable_count=row['full_native_variable_count'], full_saved_prediction_sha256=digests))
    return dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job=job,
        actual_native_receipt_sha256=original.sha(receipt_path), native_replay_proof_sha256=original.sha(proof_path),
        independently_bound_full_generation_digest=actual, independently_bound_generator_stages=stage_checks,
        independently_bound_all10_complete_prediction_arrays=rows,
        full_predictive_training_and_all45_all20_final_replay_still_required=True,
        additional_optimizer_training_executed=False, new_benchmark_results=0, experiment_complete=False,
        all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = original.read(original.ROOT / args.config)
    result = review(config)
    original.exclusive(original.ROOT / args.output, result)
    print(json.dumps(dict(passed=True, full_native_checkpoints=13, full_saved_evaluators=10,
        full_original_test_points=14660, experiment_complete=False)))


if __name__ == '__main__':
    main()
