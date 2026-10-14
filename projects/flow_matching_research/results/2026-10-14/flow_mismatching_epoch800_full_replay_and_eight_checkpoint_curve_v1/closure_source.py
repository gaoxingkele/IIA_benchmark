from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts.flow_matching import run_flow_mismatching_full_image_evaluation_queue_v1 as native

root, read, sha = native.ROOT, native.read, native.sha
queue_path = 'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json'
queue = read(root / queue_path)
native.validate_queue(queue)
previous_path = root / 'projects/flow_matching_research/results/2026-10-14/flow_mismatching_epoch700_full_replay_and_seven_checkpoint_curve_v1/all_seven_fully_replayed_checkpoint_curve.json'
previous = read(previous_path)
folder = root / 'projects/flow_matching_research/results/2026-10-14/flow_mismatching_epoch800_full_replay_and_eight_checkpoint_curve_v1'
assert not folder.exists()
curves = []
latest_sources = {}
for row in range(8):
    case_id = 'table9_epoch_checkpoints__row%02d__mvtec__seed1103' % row
    case = next(item for item in queue['cases'] if item['id'] == case_id)
    assert case['checkpoint_epoch'] == 100 * (row + 1) and native.complete(case)
    location_path = root / case['output_directory'] / 'fully_verified_score_result.json'
    location = read(location_path)
    report_path = Path(location['report'])
    report = read(report_path)
    candidate_path = Path(report['native_score_candidate'])
    candidate = read(candidate_path)
    assert location['passed'] and report['passed'] and report['case'] == candidate['case'] == case
    assert sha(report_path) == location['report_sha256']
    assert sha(candidate_path) == report['native_score_candidate_sha256']
    assert sha(root / case['checkpoint']) == report['full_original_checkpoint_sha256'] == candidate['original_checkpoint_sha256']
    assert report['queue_sha256'] == candidate['queue_sha256'] == sha(root / queue_path)
    assert report['complete_original_test_images'] == 1725 and report['complete_original_categories'] == 15
    assert report['max_absolute_heatmap_replay_error'] == report['max_absolute_image_score_replay_error'] == 0
    assert report['every_native_path_probe_seed_equal'] and report['original_full_state_dictionary_loaded_strictly']
    assert not report['original_author_equivalence_certified'] and not report['strict_TAB_result']
    for path in (candidate_path.parent / 'terminal.json', report_path.parent / 'terminal.json'):
        terminal = read(path)
        assert terminal['returncode'] == 0 and not terminal['resource_usage']['resource_guard_aborted']
    assert len(report['independently_recomputed_per_category_metrics']) == 15
    assert sum(item['original_test_images'] for item in report['independently_recomputed_per_category_metrics']) == 1725
    record = dict(checkpoint_epoch=case['checkpoint_epoch'], seed=1103, metrics_percent=report['category_macro_percent'],
        paper_metrics_percent=case['paper_reported'], paper_comparison=report['original_paper_comparison'],
        full_model_checkpoint_sha256=report['full_original_checkpoint_sha256'],
        full_native_and_independent_proof_path=str(report_path), full_native_and_independent_proof_sha256=sha(report_path))
    if row < 7:
        assert record == previous['case_curves'][row], 'Previous completed evidence must remain exact'
    curves.append(record)
    if row == 7:
        native.verify_receipts(candidate['native_truth_array_receipts'])
        native.verify_receipts(candidate['output_receipts'])
        assert candidate['theoretical_test_forward_evaluations'] == 172500
        assert candidate['full_test_batched_velocity_calls'] == 17250
        latest_sources = dict(fully_verified_score_location=location_path, full_native_score_candidate=candidate_path,
            independent_full_native_image_evaluation=report_path,
            original_full_score_guarded_terminal=candidate_path.parent / 'terminal.json',
            original_full_replay_guarded_terminal=report_path.parent / 'terminal.json')
folder.mkdir()
for name, path in latest_sources.items():
    shutil.copyfile(path, folder / (name + '.json'))
shutil.copyfile(__file__, folder / 'closure_source.py')
report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_case_seed_evaluations=325,
    full_epoch_checkpoint_cases_complete=8, complete_original_test_images_per_case=1725,
    complete_original_categories_per_case=15, case_curves=curves, previous_seven_records_unchanged=True,
    previous_seven_curve_sha256=sha(previous_path), additional_training_seed_slots=0,
    independent_training_seed_count_for_this_curve=1, additional_model_inference_by_this_archive=0,
    every_native_score_and_heatmap_replayed_with_zero_max_error=True,
    seed_confidence_interval_from_correlated_checkpoints_estimated=False,
    original_full_2000_epoch_training_complete=False, all325_case_seed_evaluations_complete=False,
    original_author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
    boundary='Eight checkpoints from one seed1103 trajectory; not eight independent seeds. Each complete1725-image/15-category evaluation retains K10/T10/minimum and native128x128 heatmaps. F1-max uses test-label oracle evaluation. Original2000epochs/five seeds/all325evaluations remain retained.')
(folder / 'all_eight_fully_replayed_checkpoint_curve.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
latest = curves[-1]
lines = ['MVTec-AD / Flow Mismatching / conditional / seed1103 / epoch800.', '',
    'Complete1725images,15categories and native128x128 heatmaps. Full checkpoint/state/probe replay passed with zero maximum heatmap and image-score error. All image/pixel metrics and analytic independent region-AUPRO were recomputed. Both guarded processes exited0 without resource abort.', '',
    '| Metric (%) | Local epoch800 | Paper Table9 epoch800 | Difference (pp) |', '|---|---:|---:|---:|']
for key in ['image_AUROC','image_AP','image_F1_max','pixel_AUROC','pixel_AP','pixel_F1_max','pixel_AUPRO']:
    local, paper = latest['metrics_percent'][key], latest['paper_metrics_percent'][key]
    lines.append('| %s | %.6f | %.6f | %+.6f |' % (key, local, paper, local - paper))
lines += ['', 'All seven previously published100..700 checkpoint records were revalidated and remain exactly unchanged. The eight checkpoints belong to one correlated training trajectory; no eight-seed mean or confidence interval is claimed.', '',
    'Original2000epochs/five seeds and all325evaluations remain incomplete. F1-max selects a threshold from test labels and is not strict TAB calibration. Local U-Net/optimizer/preprocessing/precision/AUPRO/RNG choices and unavailable author settings remain fidelity gaps. Replay equality verifies the local saved model, not author equivalence. Original A100 FPS is unverified.', '',
    'Paper: ' + read(root / case['training_configuration'])['citation']]
(folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
paths = sorted(path.relative_to(root).as_posix() for path in folder.rglob('*') if path.is_file())
manifest = dict(previous_root='ab2c47b949591f50a0727bf970d8a33eeb91ca33',
    previous_parent='a3c72b1fa07603657f7c905508769116bd455dd5', paths=paths)
with (root / 'tmp/image800_eight_checkpoint_publication_manifest_20261014_v1.json').open('x', encoding='utf-8') as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps(dict(complete_checkpoints=8, independent_training_seeds=1, metrics_percent=latest['metrics_percent'], exact_publication_files=len(paths))))
