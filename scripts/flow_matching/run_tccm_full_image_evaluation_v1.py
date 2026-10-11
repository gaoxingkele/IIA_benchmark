"""Eq20 scoring and independent every-image full-model replay on native MVTec."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import context, write_new
from scripts.flow_matching.tccm_full_image_evaluation_contract_v1 import validate_queue


def direct_replay_score(model, images, labels):
    """Independent Eq20 implementation; does not call the producer scorer."""
    import torch
    flat = ((model(images, torch.ones(images.shape[0], device=images.device, dtype=images.dtype), labels)
             + images).square().sum(dim=1).sqrt()).flatten(1)
    # ceil(128*128*0.01)=164; local shared scalar choice is frozen before scoring.
    count = (flat.shape[1] + 99) // 100
    ordered = flat.sort(dim=1).values
    return flat.reshape(images.shape[0], *images.shape[2:]), ordered[:, -1] + ordered[:, -count:].mean(1)


def independent_metrics(arrays, names, heatmaps, scores, queue, candidate, np):
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    from scripts.flow_matching.flow_mismatching_image_metrics_v1 import independent_aupro
    records = candidate['local_metrics']['per_category']
    if [row['category'] for row in records] != names:
        raise ValueError('All native TCCM categories in original order required')
    results = []
    for category, name in enumerate(names):
        subset = arrays['categories'] == category
        values = {}
        for prefix, truth, prediction in (
                ('image_', arrays['labels'][subset], scores[subset]),
                ('pixel_', arrays['masks'][subset].ravel(), heatmaps[subset].ravel())):
            precision, recall, _ = precision_recall_curve(truth, prediction)
            f1 = 2*precision*recall / np.maximum(precision+recall, 1e-100)
            values.update({prefix+'AUROC':float(roc_auc_score(truth,prediction)),
                prefix+'AP':float(average_precision_score(truth,prediction)),
                prefix+'F1_max':float(f1.max())})
        values['pixel_AUPRO'] = independent_aupro(arrays['masks'][subset],heatmaps[subset],
            fpr_limit=queue['metrics']['aupro_fpr_limit'],connectivity=queue['metrics']['region_connectivity'])
        for key,value in values.items():
            if not np.isclose(value,records[category]['metrics'][key],**queue['verification']['metric_tolerance']):
                raise ValueError('Independent full TCCM metric differs: '+name+'/'+key)
        results.append(dict(category=name,original_test_images=int(subset.sum()),metrics=values))
    means = {key:float(np.mean([row['metrics'][key] for row in results])) for key in results[0]['metrics']}
    for key,value in means.items():
        if not np.isclose(value,candidate['local_metrics']['category_macro_mean'][key],
                **queue['verification']['metric_tolerance']):
            raise ValueError('All-category TCCM macro mean differs')
    return results,means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--case-id', required=True)
    parser.add_argument('--attempt-directory',required=True); parser.add_argument('--audit',action='store_true')
    args = parser.parse_args()
    queue = read(ROOT/args.queue); validate_queue(queue)
    case = next(x for x in queue['cases'] if x['id'] == args.case_id)
    output = (ROOT/case['output_directory']).resolve(); attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output/'attempts') or not attempt.is_dir():
        raise ValueError('Fresh preserved native TCCM score/audit attempt required')
    model,arrays,names,binding,checkpoint_sha,torch,np,device = context(queue,case)
    n = len(arrays['images'])
    if n != 1725 or len(names) != 15:
        raise ValueError('Every original MVTec test image and category required')
    if args.audit:
        location = read(output/'score_candidate_location.json')
        candidate_path = Path(location['candidate'])
        if sha(candidate_path) != location['candidate_sha256']:
            raise ValueError('Full TCCM candidate changed')
        candidate = read(candidate_path)
        if (candidate['case'] != case or candidate['queue_sha256'] != sha(ROOT/args.queue)
                or candidate['training_binding'] != binding or candidate['original_checkpoint_sha256'] != checkpoint_sha
                or candidate['original_model_parameters'] != sum(p.numel() for p in model.parameters())):
            raise ValueError('Complete original TCCM model/data/queue identity differs')
        verify_receipts(candidate['output_receipts']); verify_receipts(candidate['native_truth_array_receipts'])
        loaded = {Path(x['path']).name:np.load(x['path'],mmap_mode='r',allow_pickle=False)
                  for x in candidate['output_receipts']}
        if set(loaded) != {'complete_native_test_heatmaps.npy','complete_native_test_image_scores.npy'}:
            raise ValueError('TCCM requires heatmaps/scores; no Gaussian probe stream exists')
        heatmaps = loaded['complete_native_test_heatmaps.npy']; scores = loaded['complete_native_test_image_scores.npy']
        if (heatmaps.shape != (n,128,128) or scores.shape != (n,)
                or heatmaps.dtype != np.float32 or scores.dtype != np.float32
                or not np.isfinite(heatmaps).all() or not np.isfinite(scores).all()):
            raise ValueError('All native TCCM finite float32 outputs required')
    else:
        from iia_benchmark.models.fm_tccm_image import score_tccm_image
        heatmap_path = attempt/'complete_native_test_heatmaps.npy'; score_path = attempt/'complete_native_test_image_scores.npy'
        heatmaps = np.lib.format.open_memmap(heatmap_path,mode='w+',dtype=np.float32,shape=(n,128,128))
        scores = np.lib.format.open_memmap(score_path,mode='w+',dtype=np.float32,shape=(n,))
    maximum_heat_error,maximum_score_error = 0.,0.
    with torch.inference_mode():
        for index in range(n):
            image = torch.from_numpy(np.array(arrays['images'][index:index+1],copy=True)).to(device)
            labels = torch.tensor([int(arrays['categories'][index])],device=device)
            if args.audit:
                heat,score = direct_replay_score(model,image,labels)
            else:
                heat,score = score_tccm_image(model,image,labels=labels,top_fraction=.01)
            if not torch.isfinite(heat).all() or not torch.isfinite(score).all():
                raise FloatingPointError('Nonfinite full-native TCCM score')
            actual_heat = heat[0].cpu().numpy(); actual_score = float(score[0])
            if args.audit:
                tolerance = queue['verification']['model_replay_tolerance']
                if (not np.allclose(actual_heat,heatmaps[index],**tolerance)
                        or not np.isclose(actual_score,scores[index],**tolerance)):
                    raise ValueError('Full native TCCM model replay differs at image%d'%index)
                maximum_heat_error = max(maximum_heat_error,float(np.max(np.abs(actual_heat-heatmaps[index]))))
                maximum_score_error = max(maximum_score_error,abs(actual_score-float(scores[index])))
            else:
                heatmaps[index] = actual_heat; scores[index] = actual_score
            if (index+1)%100 == 0:
                print(dict(case_id=case['id'],full_native_test_images=index+1,expected=n,audit=args.audit),flush=True)
    if not args.audit:
        from scripts.flow_matching.flow_mismatching_image_metrics_v1 import per_category_metrics
        heatmaps.flush(); scores.flush()
        metrics = per_category_metrics(heatmaps,scores,arrays['masks'],arrays['labels'],arrays['categories'],names,
            fpr_limit=queue['metrics']['aupro_fpr_limit'],connectivity=queue['metrics']['region_connectivity'])
        dataset = read(ROOT/read(ROOT/case['training_configuration'])['dataset_configuration'])
        candidate = dict(case=case,queue_configuration=args.queue,queue_sha256=sha(ROOT/args.queue),
            captured_utc=datetime.now(timezone.utc).isoformat(),training_binding=binding,
            original_checkpoint_sha256=checkpoint_sha,complete_original_test_images=n,
            complete_original_categories=len(names),full_test_velocity_calls=n,
            original_model_parameters=sum(p.numel() for p in model.parameters()),
            original_full_state_dictionary_loaded_strictly=True,local_metrics=metrics,
            output_receipts=[dict(path=str(p),sha256=sha(p)) for p in (heatmap_path,score_path)],
            native_truth_array_receipts=[x for x in dataset['array_receipts'] if x['split']=='test'],
            full_independent_metric_and_model_replay_passed=False,
            original_author_equivalence_certified=False,original_TCCM_best_domain_setting_reproduced=False,
            strict_TAB_result=False,all_original_experiments_complete=False,
            boundary='Full50epoch controlled-image TCCM candidate; independent full model replay and metrics still required.')
        path = attempt/'full_native_score_candidate.json'; write_new(path,candidate)
        write_new(output/'score_candidate_location.json',dict(candidate=str(path),candidate_sha256=sha(path)))
    else:
        from scripts.flow_matching.flow_mismatching_image_metrics_v1 import image_scores_from_heatmaps
        reduced = image_scores_from_heatmaps(heatmaps,.01)
        if not np.allclose(reduced,scores,**queue['verification']['heatmap_image_score_tolerance']):
            raise ValueError('Independent full native heatmap-to-scalar reduction differs')
        independent,means = independent_metrics(arrays,names,heatmaps,scores,queue,candidate,np)
        if sha(ROOT/case['checkpoint']) != checkpoint_sha:
            raise ValueError('Full original TCCM checkpoint changed during replay')
        verify_receipts(candidate['output_receipts'])
        report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,case=case,
            queue_configuration=args.queue,queue_sha256=sha(ROOT/args.queue),
            native_score_candidate=str(candidate_path),native_score_candidate_sha256=sha(candidate_path),
            full_original_checkpoint_sha256=checkpoint_sha,training_binding=binding,
            complete_original_test_images=n,complete_original_categories=len(names),
            original_full_state_dictionary_loaded_strictly=True,every_native_image_model_replayed=True,
            all_native_heatmaps_and_image_scores_equal=True,independent_all_category_metrics_passed=True,
            max_absolute_heatmap_replay_error=maximum_heat_error,max_absolute_image_score_replay_error=maximum_score_error,
            independently_recomputed_per_category_metrics=independent,category_macro_mean=means,
            category_macro_percent={key:value*100 for key,value in means.items()},native_performance_measured=True,
            original_author_equivalence_certified=False,original_TCCM_best_domain_setting_reproduced=False,
            strict_TAB_result=False,all_original_experiments_complete=False,
            local_unreported_settings=queue['local_unreported_settings'],
            original_paper_comparison={key:dict(local_percent=means[key]*100,paper_percent=value,
                difference_percentage_points=means[key]*100-value) for key,value in case['paper_reported'].items()})
        path = attempt/'independent_full_native_image_evaluation.json'; write_new(path,report)
        write_new(output/'fully_verified_score_result.json',dict(report=str(path),report_sha256=sha(path),candidate=location,passed=True))
    print(dict(case_id=case['id'],complete_original_images=n,full_independent_audit=args.audit))


if __name__ == '__main__':
    main()
