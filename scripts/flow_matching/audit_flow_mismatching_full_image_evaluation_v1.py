"""Every-image native model replay and different all-pixel metric recomputation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import context, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--case-id', required=True)
    parser.add_argument('--attempt-directory', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    case = next(x for x in queue['cases'] if x['id'] == args.case_id)
    output = (ROOT / case['output_directory']).resolve(); attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output / 'attempts') or not attempt.is_dir():
        raise ValueError('Fresh preserved independent audit attempt required')
    location = read(output / 'score_candidate_location.json')
    candidate_path = Path(location['candidate'])
    if sha(candidate_path) != location['candidate_sha256']:
        raise ValueError('Full native score candidate changed')
    candidate = read(candidate_path)
    if candidate['case'] != case or candidate['queue_sha256'] != sha(ROOT / args.queue):
        raise ValueError('Complete registered native score identity differs')
    verify_receipts(candidate['output_receipts']); verify_receipts(candidate['native_truth_array_receipts'])
    model, arrays, names, binding, checkpoint_sha, torch, np, device = context(queue, case)
    if (binding != candidate['training_binding'] or checkpoint_sha != candidate['original_checkpoint_sha256']
            or sum(p.numel() for p in model.parameters()) != candidate['original_model_parameters']):
        raise ValueError('Full checkpoint/data/model state identity differs')
    from iia_benchmark.models.fm_image_evaluation import score_original_image
    from scripts.flow_matching.flow_mismatching_image_metrics_v1 import independent_aupro, image_scores_from_heatmaps
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    loaded = {Path(x['path']).name: np.load(x['path'], mmap_mode='r', allow_pickle=False)
              for x in candidate['output_receipts']}
    heatmaps = loaded['complete_native_test_heatmaps.npy']
    scores = loaded['complete_native_test_image_scores.npy']
    probe_seeds = loaded['complete_native_test_probe_seeds.npy']
    n = len(arrays['images'])
    if (heatmaps.shape != (n, 128, 128) or heatmaps.dtype != np.float32
            or scores.shape != (n,) or scores.dtype != np.float32
            or probe_seeds.shape != (n, case['score_parameters']['paths']) or probe_seeds.dtype != np.int64
            or not np.isfinite(heatmaps).all() or not np.isfinite(scores).all()):
        raise ValueError('Every original native finite heatmap, score and probe required')
    tolerance = queue['verification']['model_replay_tolerance']
    maximum_heat_error, maximum_score_error = 0., 0.
    for index in range(n):
        image = torch.from_numpy(np.array(arrays['images'][index:index + 1], copy=True)).to(device)
        labels = (torch.tensor([int(arrays['categories'][index])], device=device)
                  if case['variant'] == 'conditional' else None)
        heat, score, ids = score_original_image(model, image, labels=labels,
            parameters=case['score_parameters'], training_seed=case['seed'],
            dataset=case['dataset'], native_image_index=index)
        actual_heat = heat[0].cpu().numpy(); actual_score = float(score[0])
        if (not np.array_equal(probe_seeds[index], np.asarray(ids, dtype=np.int64))
                or not np.allclose(actual_heat, heatmaps[index], **tolerance)
                or not np.isclose(actual_score, scores[index], **tolerance)):
            raise ValueError('Every-image full-model/native-draw heatmap and score replay differs at index%d' % index)
        maximum_heat_error = max(maximum_heat_error, float(np.max(np.abs(actual_heat - heatmaps[index]))))
        maximum_score_error = max(maximum_score_error, abs(actual_score - float(scores[index])))
        if (index + 1) % 100 == 0:
            print(dict(case_id=case['id'], full_native_images_replayed=index + 1, expected=n), flush=True)
    recomputed_scores = image_scores_from_heatmaps(heatmaps, case['score_parameters']['top_fraction'])
    if not np.allclose(recomputed_scores, scores, **queue['verification']['heatmap_image_score_tolerance']):
        raise ValueError('All native image scores must equal independent max-plus-top1percent heatmap reduction')
    records = candidate['local_metrics']['per_category']
    if [r['category'] for r in records] != names:
        raise ValueError('Every original category and native order required')
    independent = []
    metric_tolerance = queue['verification']['metric_tolerance']
    for index, name in enumerate(names):
        subset = arrays['categories'] == index
        values = {}
        for prefix, truth, prediction in [
                ('image_', arrays['labels'][subset], scores[subset]),
                ('pixel_', arrays['masks'][subset].ravel(), heatmaps[subset].ravel())]:
            precision, recall, _ = precision_recall_curve(truth, prediction)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-100)
            values.update({prefix + 'AUROC': float(roc_auc_score(truth, prediction)),
                           prefix + 'AP': float(average_precision_score(truth, prediction)),
                           prefix + 'F1_max': float(f1.max())})
        values['pixel_AUPRO'] = independent_aupro(arrays['masks'][subset], heatmaps[subset],
            fpr_limit=queue['metrics']['aupro_fpr_limit'], connectivity=queue['metrics']['region_connectivity'])
        for name_metric, value in values.items():
            if not np.isclose(value, records[index]['metrics'][name_metric], **metric_tolerance):
                raise ValueError('Independent full original category metric differs: ' + name + '/' + name_metric)
        independent.append(dict(category=name, original_test_images=int(subset.sum()), metrics=values))
    means = {key: float(np.mean([row['metrics'][key] for row in independent])) for key in independent[0]['metrics']}
    for key, value in means.items():
        if not np.isclose(value, candidate['local_metrics']['category_macro_mean'][key], **metric_tolerance):
            raise ValueError('All-category unweighted paper metric mean differs')
    if sha(ROOT / case['checkpoint']) != checkpoint_sha:
        raise ValueError('Immutable full checkpoint changed during independent replay')
    verify_receipts(candidate['output_receipts'])
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True, case=case,
        queue_configuration=args.queue, queue_sha256=sha(ROOT / args.queue),
        native_score_candidate=str(candidate_path), native_score_candidate_sha256=sha(candidate_path),
        complete_original_test_images=n, complete_original_categories=len(names),
        full_original_checkpoint_sha256=checkpoint_sha, training_binding=binding,
        original_full_state_dictionary_loaded_strictly=True, every_native_image_model_replayed=True,
        every_native_path_probe_seed_equal=True, all_native_heatmaps_and_image_scores_equal=True,
        max_absolute_heatmap_replay_error=maximum_heat_error, max_absolute_image_score_replay_error=maximum_score_error,
        independent_metric_algorithm='sklearn binary curves; region positive-vs-background CDF analytic AUPRO integration',
        independently_recomputed_per_category_metrics=independent, category_macro_mean=means,
        category_macro_percent={key: value * 100 for key, value in means.items()},
        native_performance_measured=True, original_author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False,
        local_unreported_settings=queue['local_unreported_settings'],
        original_paper_comparison={key: dict(local_percent=means[key] * 100,
            paper_percent=value, difference_percentage_points=means[key] * 100 - value)
            for key, value in case['paper_reported'].items() if key != 'FPS'},
        original_A100_FPS_comparison_verified=False)
    write_new(attempt / 'independent_full_native_image_evaluation.json', report)
    write_new(output / 'fully_verified_score_result.json', dict(
        report=str(attempt / 'independent_full_native_image_evaluation.json'),
        report_sha256=sha(attempt / 'independent_full_native_image_evaluation.json'),
        candidate=location, passed=True))
    print(dict(case_id=case['id'], complete_original_images_replayed=n, independent_all_metric_checks_passed=True))


if __name__ == '__main__':
    main()
