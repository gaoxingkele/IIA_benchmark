"""Archive one completed full model; recheck every current proof/input digest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'saits_paper_saits_electricity_author_missing0.3_seed27'
FOLDER = ROOT / 'projects/flow_matching_research/results/2026-10-14/saits_electricity30_seed27_closed_full_replay_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

nativepath = ROOT / f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{IDENTITY}.native.json'
reviewpath = nativepath.with_name(f'{IDENTITY}.review.json')
native, review = read(nativepath), read(reviewpath)
for config in (native, review):
    for receipt in config.get('source_receipts', []):
        assert sha(ROOT / receipt['path']) == receipt['sha256'], receipt['path']
reportpath = ROOT / native['artifact_directory'] / 'checkpoint_replay_audit.json'
proofpath = ROOT / review['output_directory'] / 'independent_full_replay_batch_verification.json'
report, proof = read(reportpath), read(proofpath)
assert report['full_models_replayed'] == proof['full_native_models_verified'] == 1
assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
assert report['configuration_sha256'] == sha(nativepath)
assert proof['configuration_sha256'] == sha(reviewpath)
assert proof['native_replay_report_sha256'] == sha(reportpath)
record, row = report['records'][0], proof['records'][0]
assert record['id'] == row['id'] == IDENTITY
assert row['full_test_shape'] == [290, 100, 370]
assert row['held_out_coordinates'] == 2781282 and row['months'] == 10
for key in ('full_truth_masks_units_groups_equal', 'full_finite_predictions',
            'full_group_statistics_independently_recomputed', 'actual_full_checkpoint_reinference_completed'):
    assert row[key]
result = read(record['result_path'])
assert (result['config']['method'], result['config']['dataset'], result['config']['seed'],
        result['config']['missing_ratio']) == ('saits', 'electricity', 27, .3)
data = ROOT / result['config']['dataset_root']
output = ROOT / result['config']['output_root']
paths = dict(config=ROOT / record['config'], result=Path(record['result_path']),
             checkpoint=Path(record['checkpoint_path']), arrays=Path(record['replay_arrays']),
             dataset=data / 'datasets.h5', resume=output / 'resume.pt',
             original_group_statistics=output / 'group_statistics.npz', data_audit=data / 'data_audit.json')
assert {key: sha(path) for key, path in paths.items()} == row['hashes']
completion = row['full_training_completion']
assert completion['full_training_finished'] and not completion['benchmark_smoke']
assert completion['original_maximum_epochs'] == 10000
assert completion['next_epoch'] == 436 and completion['author_patience_remaining'] == 0
assert result['validation_best']['epoch'] == 404 and result['parameters'] == 11511000
sources = dict(native_config=nativepath, independent_config=reviewpath,
               original_model_config=paths['config'], original_result=paths['result'],
               native_report=reportpath, independent_proof=proofpath,
               native_guarded_receipt=ROOT / native['state_root'] / 'receipt.json',
               independent_guarded_receipt=ROOT / review['state_root'] / 'receipt.json')
nr, rr = read(sources['native_guarded_receipt']), read(sources['independent_guarded_receipt'])
assert nr['exit_code'] == rr['exit_code'] == 0
assert not nr['resource_guard_aborted'] and not rr['resource_guard_aborted']
assert nr['full_native_model_replay_passed'] and rr['full_native_batch_independent_review_passed']
assert proof['metric_tolerance'] == dict(rtol=2e-6, atol=2e-6)

handles = []
for statepath in (ROOT / 'experiments/runs/fm_saits_completion_replay_queue_v1/status.json',
                  Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/status.json')):
    state = read(statepath)
    p = psutil.Process(state['pid'])
    assert p.create_time() == state['create_time']
    handles.append(dict(state_path=str(statepath), state=state,
                        actual_handle=dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline())))
assert IDENTITY in handles[0]['state']['verified_original_ids']
FOLDER.mkdir(exist_ok=False)
for name, path in sources.items():
    (FOLDER / (name + '.json')).write_bytes(path.read_bytes())
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), id=IDENTITY,
    method='saits', dataset='electricity', nominal_missing_ratio=.3, seed=27,
    full_original_training_and_model_reinference_and_independent_array_review_passed=True,
    current_full_artifact_hashes_rechecked=row['hashes'],
    proof_source_hashes={name: sha(path) for name, path in sources.items()},
    original_maximum_epochs=10000, author_early_stop_patience=30,
    next_epoch=436, best_epoch_source_index=404, author_patience_exhausted=True,
    model_parameters=11511000, full_test_shape=row['full_test_shape'],
    full_test_grid_cells=10730000, held_out_coordinates=2781282,
    effective_target_fraction_of_all_grid_cells=2781282 / 10730000,
    independently_recomputed_metrics=row['independently_recomputed_metrics'],
    original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],
    original_month_groups=10, metric_tolerance=proof['metric_tolerance'],
    additional_training_updates=0, additional_model_inference_in_this_closure=0,
    original_prediction_bitwise_equivalence_claimed=False, author_equivalence_certified=False,
    strict_TAB_result=False, required_full_model_seeds=[26, 27, 28, 29, 30],
    completed_verified_seeds_of_this_rate=[26, 27], five_seed_aggregate_complete=False,
    required_SAITS_BRITS_models=210, verified_SAITS_BRITS_models=len(handles[0]['state']['verified_original_ids']),
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False, citation=result['config']['citation'],
    data_citation=result['config']['data_citation'], protocol_boundaries=result['protocol_boundaries'])
(FOLDER / 'actual_original_saits_electricity30_seed27_complete_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(FOLDER / 'actual_live_full210_source_and_collector_handles.json').write_text(json.dumps(handles, indent=2) + '\n')
(FOLDER / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(FOLDER / '.gitattributes').write_text('*.py -text\n')
(FOLDER / 'README.md').write_text('''SAITS / Electricity / nominal 30% missing / seed 27

The frozen author protocol (10000 maximum epochs, validation-only selection, patience 30) completed at next_epoch 436; best epoch uses the original zero-based source index 404. The 11511000-parameter checkpoint passed native CPU re-inference and independent full-array review of all 290x100x370 cells, 2781282 held-out coordinates and 10 month groups. Current model, full data, resume, result, replay arrays, group statistics and proof hashes were rechecked; the existing 2e-6 tolerances were retained.

Independent MAE 0.8888732051942096; RMSE 1.3081622936196426; MRE 0.47574185981339756. Source month-block sensitivity intervals are MAE [0.8662775005737158, 0.9110409124867695], RMSE [1.2795294415073046, 1.3357416135657032], MRE [0.46911259678362827, 0.4817127380026762]. They describe ten temporal blocks for one model seed; they are not five-seed uncertainty intervals.

Replacement masking yields 25.9206151% held-out targets over the full test grid. Historical author per-rate tuning, preprocessing seeds and paper-era snapshot remain unverified. Original prediction arrays were unavailable, so this verifies metric agreement under the frozen tolerance rather than bitwise prediction agreement. This is not a strict TAB result or certified agreement with the paper. Seeds 26 and 27 of this rate are closed; seeds 28-30 and the full 210-slot SAITS/BRITS scope remain incomplete. The broader 45-paper / 681-method-record / 273-data-task scope is retained.

Paper: https://arxiv.org/abs/2202.08516
Dataset: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
''')
print(json.dumps(dict(folder=str(FOLDER), metrics=closure['independently_recomputed_metrics'],
                      verified_SAITS_BRITS_models=closure['verified_SAITS_BRITS_models'])))
