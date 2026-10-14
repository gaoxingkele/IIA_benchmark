from datetime import datetime,timezone
import json
from pathlib import Path
from scripts.flow_matching import run_flow_mismatching_full_image_evaluation_queue_v1 as native

root,read,sha=native.ROOT,native.read,native.sha
queuepath='configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json'
queue=read(root/queuepath);native.validate_queue(queue)
folder=root/'projects/flow_matching_research/results/2026-10-14/flow_mismatching_epoch700_full_replay_and_seven_checkpoint_curve_v1'
folder.mkdir(exist_ok=False)
curves=[]
for row in range(7):
    identity='table9_epoch_checkpoints__row%02d__mvtec__seed1103'%row
    case=next(c for c in queue['cases'] if c['id']==identity)
    assert case['checkpoint_epoch']==100*(row+1) and native.complete(case)
    locationpath=root/case['output_directory']/'fully_verified_score_result.json'
    location=read(locationpath);reportpath=Path(location['report']);report=read(reportpath)
    candidatepath=Path(report['native_score_candidate']);candidate=read(candidatepath)
    assert sha(reportpath)==location['report_sha256'] and sha(candidatepath)==report['native_score_candidate_sha256']
    assert report['complete_original_test_images']==1725 and report['complete_original_categories']==15
    assert report['max_absolute_heatmap_replay_error']==report['max_absolute_image_score_replay_error']==0
    assert report['every_native_path_probe_seed_equal'] and report['original_full_state_dictionary_loaded_strictly']
    assert not report['original_author_equivalence_certified'] and not report['strict_TAB_result']
    scoreterminal=read(candidatepath.parent/'terminal.json');auditterminal=read(reportpath.parent/'terminal.json')
    for terminal in [scoreterminal,auditterminal]:
        assert terminal['returncode']==0 and not terminal['resource_usage']['resource_guard_aborted']
    assert len(report['independently_recomputed_per_category_metrics'])==15
    assert sum(c['original_test_images'] for c in report['independently_recomputed_per_category_metrics'])==1725
    curves.append(dict(checkpoint_epoch=case['checkpoint_epoch'],seed=1103,
        metrics_percent=report['category_macro_percent'],paper_metrics_percent=case['paper_reported'],
        paper_comparison=report['original_paper_comparison'],full_model_checkpoint_sha256=report['full_original_checkpoint_sha256'],
        full_native_and_independent_proof_path=str(reportpath),full_native_and_independent_proof_sha256=sha(reportpath)))
    if row==6:
        for name,path in [('fully_verified_score_location.json',locationpath),('full_native_score_candidate.json',candidatepath),
            ('independent_full_native_image_evaluation.json',reportpath),
            ('original_full_score_guarded_terminal.json',candidatepath.parent/'terminal.json'),
            ('original_full_replay_guarded_terminal.json',reportpath.parent/'terminal.json')]:
            (folder/name).write_bytes(path.read_bytes())
        native.verify_receipts(candidate['native_truth_array_receipts'])
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_case_seed_evaluations=325,
    full_epoch_checkpoint_cases_complete=7,complete_original_test_images_per_case=1725,
    complete_original_categories_per_case=15,case_curves=curves,additional_training_seed_slots=0,
    independent_training_seed_count_for_this_curve=1,additional_model_inference_by_this_archive=0,
    every_native_score_and_heatmap_replayed_with_zero_max_error=True,
    seed_confidence_interval_from_correlated_checkpoints_estimated=False,
    original_full_2000_epoch_training_complete=False,all325_case_seed_evaluations_complete=False,
    original_author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
    boundary='Seven checkpoints from one original seed1103 training trajectory, not seven independent seeds. Each full1725-image/15-category evaluation preserves K10/T10/minimum, paired CUDA probes and native128 heatmaps. F1-max uses test-label oracle evaluation. All original2000-epoch/five-seed training and325 evaluations/485 runnable plus35 comparator metric targets remain.')
(folder/'all_seven_fully_replayed_checkpoint_curve.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
columns=['image_AUROC','image_AP','image_F1_max','pixel_AUROC','pixel_AP','pixel_F1_max','pixel_AUPRO']
latest=curves[-1]
lines=['MVTec-AD / Flow Mismatching / conditional / seed1103 / epoch700 passed full1725-image,15-category native scoring, checkpoint replay and independently recomputed image/pixel/region metrics. Heatmap and image-score replay maximum errors both0. Both actual guarded processes exited0 without resource abort.\n',
    '| Metric (%) | Local epoch700 | Paper Table9 epoch700 | Difference (pp) |',
    '| --- | ---: | ---: | ---: |']
for key in columns:
    local,paper=latest['metrics_percent'][key],latest['paper_metrics_percent'][key]
    lines.append('| %s | %.6f | %.6f | %+.6f |'%(key,local,paper,local-paper))
lines += ['', 'The seven fully replayed checkpoints100..700 are one correlated training trajectory; no seven-seed mean or confidence interval is claimed. The original2000-epoch/five-seed training and all325 evaluations remain incomplete. F1-max is retrospective test-label oracle evaluation, not strict TAB. Local U-Net/training/preprocessing/inference precision/AUPRO/RNG choices and unavailable author settings remain fidelity gaps; score replay equality certifies the local model, not original-author equivalence. No original A100 FPS claim is certified.']
(folder/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(full_checkpoints=7,latest_epoch=700,independent_training_seeds=1,
    metrics_percent=latest['metrics_percent'],paper_comparison=latest['paper_comparison']),indent=2))
