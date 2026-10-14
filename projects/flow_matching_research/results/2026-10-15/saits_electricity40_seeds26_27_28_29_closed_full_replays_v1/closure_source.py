"""Archive four full original models without claiming a complete five-seed cohort."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'projects/flow_matching_research/results/2026-10-15/saits_electricity40_seeds26_27_28_29_closed_full_replays_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

state_path = ROOT / 'experiments/runs/fm_saits_completion_replay_queue_v1/status.json'
state = read(state_path)
process = psutil.Process(state['pid'])
assert process.create_time() == state['create_time']
assert len(state['verified_original_ids']) == 43 and state['required_original_models'] == 210
rows, all_sources = [], {}
for seed in (26, 27, 28, 29):
    identity = f'saits_paper_saits_electricity_author_missing0.4_seed{seed}'
    assert identity in state['verified_original_ids']
    native_path = ROOT / f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{identity}.native.json'
    review_path = native_path.with_name(identity + '.review.json')
    native, review = read(native_path), read(review_path)
    for config in (native, review):
        for pin in config['source_receipts']:
            assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
    report_path = ROOT / native['artifact_directory'] / 'checkpoint_replay_audit.json'
    proof_path = ROOT / review['output_directory'] / 'independent_full_replay_batch_verification.json'
    report, proof = read(report_path), read(proof_path)
    assert report['full_models_replayed'] == proof['full_native_models_verified'] == 1
    assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
    assert report['configuration_sha256'] == sha(native_path)
    assert proof['configuration_sha256'] == sha(review_path)
    assert proof['native_replay_report_sha256'] == sha(report_path)
    record, checked = report['records'][0], proof['records'][0]
    assert record['id'] == checked['id'] == identity
    assert checked['full_test_shape'] == [290, 100, 370]
    assert checked['held_out_coordinates'] == 3537183 and checked['months'] == 10
    for key in ('full_truth_masks_units_groups_equal', 'full_finite_predictions',
                'full_group_statistics_independently_recomputed', 'actual_full_checkpoint_reinference_completed'):
        assert checked[key]
    result = read(record['result_path'])
    assert result['status'] == 'completed' and result['parameters'] == 11511000
    assert (result['config']['method'], result['config']['dataset'], result['config']['seed'],
            result['config']['missing_ratio']) == ('saits', 'electricity', seed, .4)
    dataset = ROOT / result['config']['dataset_root']
    output = ROOT / result['config']['output_root']
    artifacts = dict(config=ROOT / record['config'], result=ROOT / record['result_path'],
        checkpoint=ROOT / record['checkpoint_path'], arrays=ROOT / record['replay_arrays'],
        dataset=dataset / 'datasets.h5', resume=output / 'resume.pt',
        original_group_statistics=output / 'group_statistics.npz', data_audit=dataset / 'data_audit.json')
    assert {k:sha(p) for k,p in artifacts.items()} == checked['hashes']
    completion = checked['full_training_completion']
    assert completion['full_training_finished'] and not completion['benchmark_smoke']
    assert completion['original_maximum_epochs'] == 10000
    assert completion['author_patience_remaining'] == 0 or completion['next_epoch'] == 10000
    original_guard_path = Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts') / (identity + '.json')
    original_guard = read(original_guard_path)
    assert original_guard['exit_code'] == 0 and original_guard['full_result_validated'] and not original_guard['resource_guard_aborted']
    sources = dict(original_result=artifacts['result'], original_training_guard=original_guard_path,
        native_configuration=native_path, independent_configuration=review_path,
        full_native_report=report_path, full_independent_proof=proof_path,
        native_guard=ROOT / native['state_root'] / 'receipt.json', independent_guard=ROOT / review['state_root'] / 'receipt.json')
    nr, rr = read(sources['native_guard']), read(sources['independent_guard'])
    assert nr['exit_code'] == rr['exit_code'] == 0 and not nr['resource_guard_aborted'] and not rr['resource_guard_aborted']
    assert nr['full_native_model_replay_passed'] and rr['full_native_batch_independent_review_passed']
    assert report['metric_tolerance'] == proof['metric_tolerance'] == dict(rtol=2e-6, atol=2e-6)
    rows.append(dict(id=identity, seed=seed, full_training_completion=completion,
        validation_best=result['validation_best'], independently_recomputed_metrics=checked['independently_recomputed_metrics'],
        original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'], full_artifact_hashes=checked['hashes'],
        current_proof_hashes={k:sha(p) for k,p in sources.items()}, citation=result['config']['citation'],
        data_citation=result['config']['data_citation'], protocol_boundaries=result['protocol_boundaries']))
    all_sources[identity] = sources
assert len({r['full_artifact_hashes']['dataset'] for r in rows}) == 1
BASE.mkdir(parents=True, exist_ok=False)
for identity, sources in all_sources.items():
    folder = BASE / identity
    folder.mkdir()
    for name, source in sources.items():
        shutil.copyfile(source, folder / (name + '.json'))
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), method='saits', dataset='electricity',
    nominal_missing_ratio=.4, full_test_shape=[290,100,370], full_test_cells_per_model=10730000,
    held_out_coordinates_per_model=3537183, actual_target_fraction=3537183/10730000,
    model_parameters=11511000, required_maximum_epochs=10000, author_early_stop_patience=30,
    required_original_training_seeds=[26,27,28,29,30], fully_verified_original_seeds=[26,27,28,29],
    complete_five_seed_aggregate=False, reviewed_SAITS_BRITS_models=43, required_SAITS_BRITS_models=210,
    all_four_original_trainings_full_native_models_and_independent_reviews_passed=True, per_model=rows,
    original_prediction_bitwise_equivalence_claimed=False, metric_tolerance=dict(rtol=2e-6,atol=2e-6),
    within_model_month_block_uncertainty_is_not_training_seed_uncertainty=True,
    additional_training_updates=0, additional_model_reinference_in_this_archive=0,
    author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False)
(BASE / 'four_complete_original_models_and_full_grid_reviews.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf-8')
(BASE / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(BASE / '.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
lines = ['SAITS / Electricity / nominal40% missing / full seeds26,27,28 and29.', '',
    'All four11511000-parameter original models completed max10000/patience30 training and passed full native checkpoint inference plus independent verification of all290x100x370 cells,3537183targets and10months at unchanged2e-6 tolerance. Current data/artifact/guard/proof hashes were rechecked.', '',
    '|Seed|MAE|RMSE|MRE|', '|---|---:|---:|---:|']
for row in rows:
    m = row['independently_recomputed_metrics']
    lines.append(f"|{row['seed']}|{m['mae']:.9g}|{m['rmse']:.9g}|{m['mre']:.9g}|")
lines += ['', '43/210 original SAITS/BRITS models are fully reviewed. Nominal40% replacement masking holds out' + str(3537183/10730000) + 'of all test cells. Each original month-block interval is preserved; no five-training-seed mean or interval is claimed for this4/5cohort. Original prediction arrays were not retained, so no bitwise original-prediction equivalence claim. Historical per-rate tuning/preprocessing remain unproved. Seed30 and every original210slot/45paper/681method/273data obligation remain required.']
(BASE / 'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(full_models_closed=4, total_fully_reviewed=43, metrics=[r['independently_recomputed_metrics'] for r in rows])))
