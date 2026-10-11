"""Verify exact whole-test replay coverage of every completed canonical GiFlow job."""
import argparse
import math
from pathlib import Path

import numpy as np

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.audit_giflow_full_arrays_v2 import batch_errors
from scripts.flow_matching.giflow_native_protocol import verify as verify_queue
from scripts.flow_matching.summarize_giflow_canonical_v1 import observed_result
from scripts.flow_matching.verify_giflow_corrected_full_replay_v1 import verify_batch


def exact_scope(completed, proofs):
    expected = {job['id']: job for job in completed}
    if len(expected) != len(completed):
        raise ValueError('Canonical completed job identities must be unique')
    actual = {}
    for proof in proofs:
        identity = proof['job_id']
        if identity in actual:
            raise ValueError('A repeated model replay cannot add a training seed')
        if identity not in expected:
            raise ValueError('A replay outside the original completed scope cannot enter')
        job = expected[identity]
        if (proof['seed'] != job['seed'] or proof['original_batch_size'] != job['arguments']['batch_size']
                or proof['original_Euler_steps'] != 20 or proof['full_test_windows'] != 1747
                or proof['full_test_batches'] != 14 or proof['observed_forward_calls'] != 280
                or proof['observed_window_evaluations'] != 34940
                or len(proof['full_batch_receipts']) != 14
                or any(not proof[key] for key in ['full_checkpoint_contents_and_reinference_passed',
                    'full_data_masks_graph_scaler_and_splits_bitwise', 'actual_tested_weights_restored_bitwise'])
                or proof.get('additional_training_seed_slots', 0) != 0
                or proof.get('cuda_context_initialized', False) or proof['author_equivalence_certified']):
            raise ValueError('Every full original CPU model replay and inference budget required')
        actual[identity] = proof
    if set(actual) != set(expected):
        raise ValueError('Every completed original canonical model must be covered')
    return actual


def verify(config):
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen full native replay union evidence changed: ' + receipt['path'])
    queue = read(ROOT / config['queue'])
    verify_queue(queue)
    completed, results = [], {}
    for job in queue['jobs']:
        result = observed_result(job, config['queue'])
        if result is not None:
            completed.append(job)
            results[job['id']] = result
    proofs = [read(ROOT / entry['proof']) for entry in config['models']]
    actual = exact_scope(completed, proofs)
    if len(completed) != 3 or len(queue['jobs']) != 140:
        raise ValueError('Frozen three completed models within the unchanged140 original slots required')
    entries = {entry['job_id']: entry for entry in config['models']}
    records = []
    for job in completed:
        result, proof, entry = results[job['id']], actual[job['id']], entries[job['id']]
        native = read(ROOT / entry['replay_config'])
        if native['queue'] != config['queue'] or native['job_id'] != job['id']:
            raise ValueError('Each full model replay must bind the original scientific queue')
        if (native['prediction_tolerance'] != config['prediction_tolerance']
                or native['metric_tolerance'] != config['metric_tolerance']):
            raise ValueError('Original prediction and metric tolerances must remain frozen')
        for receipt in native['source_receipts']:
            if sha(ROOT / receipt['path']) != receipt['sha256']:
                raise ValueError('Original native source, model, data or training receipt changed')
        terminal = read(ROOT / entry['worker_receipt'])
        if terminal['exit_code'] != 0 or not terminal['full_model_replay_passed'] or terminal['resource_guard_aborted']:
            raise ValueError('Actual successful terminal full-model execution required')
        for artifact in result['artifacts']:
            if sha(ROOT / artifact['path']) != artifact['sha256']:
                raise ValueError('Actual complete checkpoint, data, history or prediction artifact changed')
        original = [item for item in result['artifacts'] if 'targets' in item]
        rows, original_values = [], []
        for index, (before, replay) in enumerate(zip(original, proof['full_batch_receipts'])):
            if (replay['batch'] != index or replay['original_sha256'] != before['sha256']
                    or sha(before['path']) != before['sha256'] or sha(replay['path']) != replay['sha256']):
                raise ValueError('Complete original/replayed batch ordering and SHA differ')
            with np.load(before['path'], allow_pickle=False) as saved, np.load(replay['path'], allow_pickle=False) as restored:
                row = verify_batch(dict(restored), dict(saved), 128 if index < 13 else 83,
                                   before['targets'], native['prediction_tolerance'])
                original_values.append(batch_errors(saved['prediction'], saved['target']))
            rows.append(dict(batch=index, original_path=before['path'], original_sha256=before['sha256'],
                             replay_path=replay['path'], replay_sha256=replay['sha256'], **row))
        targets = sum(row['targets'] for row in rows)
        if (len(rows) != 14 or sum(row['windows'] for row in rows) != 1747
                or targets != result['test_targets_with_window_repetitions']
                or targets != proof['full_masked_test_targets']):
            raise ValueError('All actual seed-specific masked coordinates must be covered')
        def aggregate(values):
            metrics = {key: float(np.mean([value[key] for value in values])) for key in
                       ['native_mean_batch_mae', 'native_mean_batch_mse', 'native_mean_batch_mape_percent']}
            metrics['sqrt_native_mean_batch_mse'] = math.sqrt(metrics['native_mean_batch_mse'])
            return metrics
        recalculated = aggregate([row['independently_recomputed_metrics'] for row in rows])
        original_metrics = aggregate(original_values)
        for key, value in recalculated.items():
            for expected in [result['metrics'][key], proof['original_metrics'][key], proof['replayed_metrics'][key], original_metrics[key]]:
                if not math.isclose(value, expected, **native['metric_tolerance']):
                    raise ValueError('Complete independent native unweighted batch metric differs')
        original_binding = ROOT / job['output_directory'] / 'split_mask_graph_scaler.npz'
        replay_binding = Path(native['artifact_root']) / 'split_mask_graph_scaler.npz'
        with np.load(original_binding, allow_pickle=False) as original_data, np.load(replay_binding, allow_pickle=False) as replay_data:
            if (set(original_data.files) != set(replay_data.files)
                    or any(not np.array_equal(original_data[key], replay_data[key]) for key in original_data.files)):
                raise ValueError('All original split, mask, graph and scaler identities must match')
            binding_arrays = len(original_data.files)
        records.append(dict(job_id=job['id'], seed=job['seed'], track=job['track'], dataset=job['dataset'],
            full_training_epochs=result['epochs_completed'], optimizer_updates=result['optimizer_updates'],
            configured_maximum_epochs=job['arguments']['training_epoch'], original_early_stopping_retained=True,
            original_training_result_sha256=sha(ROOT / job['output_directory'] / 'result.json'),
            full_test_windows=1747, full_test_batches=14, full_masked_targets=targets,
            original_Euler_steps=20, full_native_forward_calls=280, full_original_binding_arrays_verified=binding_arrays,
            original_binding_sha256=sha(original_binding), replay_binding_sha256=sha(replay_binding),
            actual_evaluated_checkpoint_sha256=sha(ROOT / job['output_directory'] / 'evaluated_model.pt'),
            independently_recomputed_original_metrics=original_metrics, independently_recomputed_replayed_metrics=recalculated,
            original_metrics=result['metrics'], test_used_in_validation=result['selection']['test_used_in_validation'],
            native_full_checkpoint_replay_verified=True, complete_original_arrays_and_metrics_verified=True, batches=rows))
    return dict(original_canonical_slots=140, completed_canonical_models=3,
        all_completed_canonical_models_replayed_and_independently_verified=True,
        full_scoring_batches=42, full_test_window_evaluations=5241, full_masked_target_evaluations=sum(r['full_masked_targets'] for r in records),
        full_native_forward_calls=840, records=records, additional_training_seed_slots=0,
        released_training_seeds_completed=2, released_training_seeds_required=5,
        corrected_training_seeds_completed=1, corrected_training_seeds_required=5,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        boundary='All three previously completed original canonical models;42 whole native batches independently checked. '
                 'Full original early-stopped training and20Euler-step model replays were preserved, not repeated. '
                 'Released test-in-validation and EMA-per-training-batch reset boundaries remain. '
                 'Native overlapping-window unweighted batch means retained; square-root batch-mean MSE is not global target-weighted RMSE. '
                 'No new training seed or author/TAB equivalence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    output = ROOT / config['output_proof']
    if output.exists():
        raise FileExistsError('Preserve every immutable full native replay union')
    proof = verify(config)
    proof.update(configuration=args.config, configuration_sha256=sha(ROOT / args.config))
    write(output, proof)
    print(dict(models=3, full_batches=42, original_slots=140, passed=True, new_training_seeds=0))


if __name__ == '__main__':
    main()
