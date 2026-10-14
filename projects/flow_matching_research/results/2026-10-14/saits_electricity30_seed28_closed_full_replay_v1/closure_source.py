"""Archive one completed full model; recheck every current proof/input digest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'saits_paper_saits_electricity_author_missing0.3_seed28'
FOLDER = ROOT / 'projects/flow_matching_research/results/2026-10-14/saits_electricity30_seed28_closed_full_replay_v1'

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
        result['config']['missing_ratio']) == ('saits', 'electricity', 28, .3)
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
assert completion['next_epoch'] == 378 and completion['author_patience_remaining'] == 0
assert result['validation_best']['epoch'] == 346 and result['parameters'] == 11511000
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
    method='saits', dataset='electricity', nominal_missing_ratio=.3, seed=28,
    full_original_training_and_model_reinference_and_independent_array_review_passed=True,
    current_full_artifact_hashes_rechecked=row['hashes'],
    proof_source_hashes={name: sha(path) for name, path in sources.items()},
    original_maximum_epochs=10000, author_early_stop_patience=30,
    next_epoch=378, best_epoch_source_index=346, author_patience_exhausted=True,
    model_parameters=11511000, full_test_shape=row['full_test_shape'],
    full_test_grid_cells=10730000, held_out_coordinates=2781282,
    effective_target_fraction_of_all_grid_cells=2781282 / 10730000,
    independently_recomputed_metrics=row['independently_recomputed_metrics'],
    original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],
    original_month_groups=10, metric_tolerance=proof['metric_tolerance'],
    additional_training_updates=0, additional_model_inference_in_this_closure=0,
    original_prediction_bitwise_equivalence_claimed=False, author_equivalence_certified=False,
    strict_TAB_result=False, required_full_model_seeds=[26, 27, 28, 29, 30],
    completed_verified_seeds_of_this_rate=[26, 27, 28], five_seed_aggregate_complete=False,
    required_SAITS_BRITS_models=210, verified_SAITS_BRITS_models=len(handles[0]['state']['verified_original_ids']),
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False, citation=result['config']['citation'],
    data_citation=result['config']['data_citation'], protocol_boundaries=result['protocol_boundaries'])
(FOLDER / 'actual_original_saits_electricity30_seed28_complete_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(FOLDER / 'actual_live_full210_source_and_collector_handles.json').write_text(json.dumps(handles, indent=2) + '\n')
(FOLDER / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(FOLDER / '.gitattributes').write_text('*.py -text\n')
(FOLDER / 'README.md').write_text(
    'SAITS / Electricity / nominal 30% missing / seed 28\n\n'
    'The frozen 10000 maximum-epoch, validation-only selection, patience30 protocol ended at next_epoch378; best epoch uses source index346. The11511000-parameter full model passed native CPU replay and independent review of all290x100x370 cells,2781282 targets and10 months at the original2e-6 tolerance. Current checkpoint/data/resume/result/arrays/group/proof hashes were rechecked.\n\n'
    + 'Independent metrics: ' + json.dumps(closure['independently_recomputed_metrics'])
    + '\nSource month-block sensitivity intervals: ' + json.dumps(closure['original_month_block_bootstrap_95_ci'])
    + '\nThese describe temporal blocks for one model seed, not five-seed uncertainty. Actual held-out coverage is25.9206151% under replacement masking. Historical per-rate tuning and preprocessing seeds remain unknown; original predictions were not retained, so no bitwise prediction claim or author-equivalence/TAB certificate. Seeds26,27,28 are closed; seeds29,30 and all210 SAITS/BRITS slots remain required. Preserve full45/681/273scope.\n\n'
    + 'Paper: https://arxiv.org/abs/2202.08516\nDataset: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014\n')
print(json.dumps(dict(folder=str(FOLDER), metrics=closure['independently_recomputed_metrics'],
                      verified_SAITS_BRITS_models=closure['verified_SAITS_BRITS_models'])))
