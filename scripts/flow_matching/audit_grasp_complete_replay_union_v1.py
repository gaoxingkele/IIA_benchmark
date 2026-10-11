"""Independently verify all thirteen original GRASP full-model replay artifacts."""
import argparse
import json
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def metrics(labels, scores, validation):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    labels = np.asarray(labels).astype(bool)
    threshold = float(np.quantile(validation, .99))
    predicted = scores > threshold
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    p, r, _ = precision_recall_curve(labels, scores)
    best = float(np.max(np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r)>0)))
    return dict(paper_retrospective=dict(AP=float(average_precision_score(labels, scores)),
                    ROC=float(roc_auc_score(labels, scores)), Best_F1=best),
                validation_only_control=dict(precision=precision, recall=recall, f1=f1, threshold=threshold))


def verify(config):
    import numpy as np
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen whole-model replay union input changed: ' + item['path'])
    q = read(ROOT / config['replay_config'])
    audit_root = ROOT / q['output_directory']
    validation = read(audit_root / 'validation.json')
    for receipt in validation['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Full native replay source receipt changed')
    for name, expected in validation['outputs'].items():
        if sha(audit_root / name) != expected:
            raise ValueError('Full native replay output changed')
    terminal = read(ROOT / q['state_root'] / 'receipt.json')
    status = read(ROOT / q['state_root'] / 'status.json')
    if (terminal['exit_code'] != 0 or not terminal['full_replay_passed'] or terminal['resource_guard_aborted']
            or status['state'] != 'completed' or status['active_pid'] is not None):
        raise ValueError('Authoritative successful full native worker exit required')
    audit = read(audit_root / 'checkpoint_replay_audit.json')
    if not audit['full_checkpoint_contents_and_re_inference_passed'] or audit['additional_training_seed_slots'] != 0:
        raise ValueError('Complete checkpoint restoration without invented training repeats required')
    records = list(audit['records'])
    for preserved in q['completed_full_replays_preserved']:
        proof = read(ROOT / preserved['path'])
        if sha(ROOT / preserved['path']) != preserved['sha256']:
            raise ValueError('Previously complete original native replay changed')
        original = read(ROOT / preserved['scientific_result']['result_path'])
        if sha(ROOT / preserved['scientific_result']['result_path']) != preserved['scientific_result']['result_sha256']:
            raise ValueError('Original preserved complete training result changed')
        records.append(dict(canonical_id=preserved['canonical_id'], recipes=proof['completed_recipes'],
                            original_scientific_result=original,
                            result_path=preserved['scientific_result']['result_path'],
                            result_sha256=preserved['scientific_result']['result_sha256']))
    identities = [r['canonical_id'] for r in records]
    if len(identities) != 13 or len(set(identities)) != 13 or sum(len(r['recipes']) for r in records) != 31:
        raise ValueError('Exactly all thirteen original trained models and thirty-one whole-data recipes required')
    verified = []
    for model in records:
        if sha(ROOT / model['result_path']) != model['result_sha256']:
            raise ValueError('Original full training result changed')
        original_result = read(ROOT / model['result_path'])
        for artifact in original_result['artifacts']:
            if sha(ROOT / artifact['path']) != artifact['sha256']:
                raise ValueError('Full checkpoint/training history/scoring artifact changed')
        recipes = []
        for recipe in model['recipes']:
            if not recipe['full_recipe_reinference_passed']:
                raise ValueError('An incomplete native scoring recipe cannot enter the union')
            original_path = Path(recipe['original_score_path'])
            replay_path = Path(recipe['replay_score_path'])
            if sha(original_path) != recipe['original_score_sha256'] or sha(replay_path) != recipe['replay_score_sha256']:
                raise ValueError('Full original/replayed scoring artifact changed')
            with np.load(original_path, allow_pickle=False) as original, np.load(replay_path, allow_pickle=False) as replay:
                for field, shape in [('test_scores', (72000,)), ('validation_scores', (3935,)),
                                     ('labels', (72000,)), ('test_row_ids', (72000,)), ('validation_raw_row_ids', (3935,))]:
                    if original[field].shape != shape or replay[field].shape != shape:
                        raise ValueError('Every original validation/test point required')
                    if not np.isfinite(original[field]).all() or not np.isfinite(replay[field]).all():
                        raise ValueError('Complete finite scoring artifacts required')
                    if field not in ('test_scores', 'validation_scores') and not np.array_equal(original[field], replay[field]):
                        raise ValueError('Original full labels/point identities changed')
                for field in ['test_scores', 'validation_scores']:
                    if not np.allclose(original[field], replay[field], **q['score_tolerance']):
                        raise ValueError('Full native scores differ under originally frozen tolerance')
                original_pred = original['test_scores'] > np.quantile(original['validation_scores'], .99)
                replay_pred = replay['test_scores'] > np.quantile(replay['validation_scores'], .99)
                disagreements = int(np.sum(original_pred != replay_pred))
                if disagreements / 72000 > q['metric_tolerance']['maximum_prediction_disagreement_rate']:
                    raise ValueError('Complete original point predictions differ')
                actual = metrics(replay['labels'], replay['test_scores'], replay['validation_scores'])
                differences = {}
                for protocol, values in actual.items():
                    for key, value in values.items():
                        expected = recipe['replayed_metrics'][protocol][key]
                        differences[protocol+'/'+key] = abs(value-expected)
                        if key == 'threshold':
                            if not np.isclose(value, expected, rtol=q['metric_tolerance']['threshold_rtol'],
                                              atol=q['metric_tolerance']['threshold_atol']):
                                raise ValueError('Independent full validation-only threshold differs')
                        elif abs(value-expected) > q['metric_tolerance']['metric_atol']:
                            raise ValueError('Independent full-array metric differs')
            recipes.append(dict(recipe=recipe['recipe'], original_score_path=str(original_path),
                original_score_sha256=sha(original_path), replay_score_path=str(replay_path), replay_score_sha256=sha(replay_path),
                complete_test_points=72000, complete_validation_points=3935,
                independently_recalculated_metrics=actual, independent_metric_absolute_differences=differences,
                prediction_disagreements=disagreements, passed=True))
        verified.append(dict(canonical_id=model['canonical_id'], epochs_completed=original_result['epochs_completed'],
                             optimizer_updates=original_result['optimizer_updates'], parameter_count=original_result['parameter_count'],
                             original_complete_training_result_sha256=model['result_sha256'],
                             all_original_checkpoint_history_and_scoring_artifact_hashes_verified=True, recipes=recipes))
    return dict(full_original_models=13, full_original_scoring_recipes=31, records=verified,
        all_complete_arrays_point_mapping_and_metrics_independently_verified=True,
        original_full_native_worker_exit_success_verified=True, no_new_training_seed_slots=True,
        torch_imported='torch' in sys.modules, full_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False,
        scope_boundary='Full replay of these13 previously trained models/31recipes only; all other original/adopted paper obligations remain.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    output = ROOT / config['output']
    if output.exists():
        raise FileExistsError('Preserve previous independent union verification')
    write(output, verify(config))
    print(json.dumps(dict(verified_original_full_models=13, full_scoring_recipes=31, passed=True)))


if __name__ == '__main__':
    main()
