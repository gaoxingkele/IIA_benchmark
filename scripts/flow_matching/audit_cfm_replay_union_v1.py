"""Verify exact coverage of preserved full-model replays and a new increment."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def exact_replay_union(snapshot, proofs):
    expected = {j['id']: j for j in snapshot['jobs'] if j['status'] == 'completed'}
    records = {}
    for proof in proofs:
        if (not proof['full_checkpoint_contents_and_re_inference_passed']
                or proof['full_job_count'] != len(proof['records'])
                or proof['cuda_context_initialized']
                or proof['additional_training_seed_slots'] != 0):
            raise ValueError('Passed complete CPU checkpoint proof required')
        for record in proof['records']:
            identity = record['original_canonical_id']
            if identity in records:
                raise ValueError('Duplicate canonical replay cannot count as a new seed')
            if identity not in expected:
                raise ValueError('Replay outside frozen completed canonical scope')
            job = expected[identity]
            if (record['result_path'] != job['result_path']
                    or record['result_sha256'] != job['result_sha256']
                    or not record['checkpoint_contents_and_full_budget_verified']
                    or not record['original_full_grid_replayed']):
                raise ValueError('Replay must bind the current full canonical result')
            records[identity] = record
    if set(records) != set(expected):
        raise ValueError('Every currently completed canonical model must be covered')
    return expected, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full-model union evidence changed')
    snapshot = read(ROOT / config['capture'] / 'execution_audit.json')
    proofs = [read(ROOT / p) for p in config['full_model_proofs']]
    if any(p['prediction_tolerance'] != config['prediction_tolerance'] for p in proofs):
        raise ValueError('Original replay comparison tolerance must stay frozen')
    expected, records = exact_replay_union(snapshot, proofs)
    import numpy as np
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, prediction_metrics
    verified = []
    for identity, replay in records.items():
        canonical = expected[identity]
        result = read(ROOT / replay['result_path'])
        job = result['job']
        if (sha(ROOT / replay['result_path']) != replay['result_sha256']
                or sha(replay['checkpoint_path']) != replay['checkpoint_sha256']
                or sha(replay['replay_arrays']) != replay['replay_arrays_sha256']
                or sha(ROOT / job['data_path']) != job['data_sha256']
                or replay['original_full_data_sha256'] != job['data_sha256']
                or sha(ROOT / job['model_config']) != job['model_config_sha256']
                or replay['epochs'] != job['epochs']
                or replay['optimizer_updates'] != job['expected_optimizer_updates']
                or replay['execution_id'] != job['id'] or replay['seed'] != job['seed']):
            raise ValueError('Preserved full checkpoint/data/result/budget binding differs')
        artifacts = {Path(a['path']).name: a for a in result['artifacts']}
        for artifact in artifacts.values():
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Preserved training artifact changed')
        if (Path(artifacts['model.pt']['path']).resolve() != Path(replay['checkpoint_path']).resolve()
                or artifacts['model.pt']['sha256'] != replay['checkpoint_sha256']):
            raise ValueError('Replay checkpoint must match actual completed training')
        with np.load(ROOT / job['data_path'], allow_pickle=False) as data, \
                np.load(artifacts['predictions.npz']['path'], allow_pickle=False) as original, \
                np.load(replay['replay_arrays'], allow_pickle=False) as reproduced:
            audit_arrays(job, result, original, data)
            for field in ('truth', 'times', 'test_ids'):
                if not np.array_equal(reproduced[field], original[field]):
                    raise ValueError('Full replay test identity differs: ' + field)
            shape = (job['test_trajectories'], job['test_points_per_trajectory'], job['observed_dimensions'])
            if tuple(replay['full_replay_shape']) != shape or reproduced['predictions'].shape != shape:
                raise ValueError('All original trajectory/grid/dimension points required')
            if not np.isfinite(reproduced['predictions']).all() or not np.allclose(
                    reproduced['predictions'], original['predictions'], **config['prediction_tolerance']):
                raise ValueError('Actual full replay predictions do not match saved predictions')
            if replay['bitwise_identical'] != bool(np.array_equal(reproduced['predictions'], original['predictions'])):
                raise ValueError('Bitwise-equivalence evidence differs')
            actual_difference = float(np.max(np.abs(reproduced['predictions'].astype(np.float64)
                                                      - original['predictions'].astype(np.float64))))
            if actual_difference != replay['max_absolute_difference']:
                raise ValueError('Actual maximum prediction difference differs')
            metrics, _ = prediction_metrics(reproduced['predictions'], reproduced['truth'])
            for key, value in metrics.items():
                recorded = replay['replayed_metrics'][key]
                if ((value is None) != (recorded is None)
                        or value is not None and not np.allclose(value, recorded, rtol=1e-12, atol=1e-12)):
                    raise ValueError('Independently recomputed full-replay metric differs')
        verified.append(dict(replay, canonical_status=canonical['status'],
                             source_full_model_replay_and_current_artifacts_verified=True))
    target = ROOT / config['output_proof']
    if target.exists():
        raise FileExistsError('Full-model union proofs are immutable')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=args.config,
                  configuration_sha256=sha(ROOT / args.config),
                  source_capture_sha256=sha(ROOT / config['capture'] / 'execution_audit.json'),
                  full_checkpoint_contents_and_re_inference_passed=True,
                  full_canonical_models_verified=len(verified),
                  original_canonical_slots=snapshot['registered_jobs'],
                  required_original_scope=config['original_scope'],
                  source_full_model_proofs=config['full_model_proofs'],
                  preserved_model_replays=config['preserved_model_replays'],
                  newly_replayed_models=len(verified)-config['preserved_model_replays'],
                  all_completed_canonical_models_covered=True,
                  all_original_test_points_covered=True,
                  bitwise_identical_models=sum(r['bitwise_identical'] for r in verified),
                  original_prediction_tolerance=config['prediction_tolerance'],
                  additional_canonical_training_seed_slots=0,
                  all_original_experiments_complete=False,
                  author_equivalence_certified=False, records=verified,
                  boundary='Exact union of actual preserved full-model replay proofs; each current checkpoint, prediction array, original data identity and frozen training budget is independently rebound and verified. No repeated historical inference, reduced grids, new seeds or cherry-picked attempts. The remaining original tasks and author equivalence remain unproven.')
    write(target, report)
    print({k: report[k] for k in ('full_canonical_models_verified', 'preserved_model_replays',
                                 'newly_replayed_models', 'bitwise_identical_models')})


if __name__ == '__main__':
    main()
