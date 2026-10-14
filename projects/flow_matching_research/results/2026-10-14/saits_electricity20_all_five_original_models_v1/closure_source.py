from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import psutil

from scripts.flow_matching.publish_saits_two_seed_cohort_v52 import verify_registration, check_record
from scripts.flow_matching.publish_saits_electricity20_seed26_v50 import ROOT, read, digest
from scripts.flow_matching.summarize_saits_full_five_seed_cohort_v1 import summarize

BASE = 'projects/flow_matching_research/results/2026-10-14/saits_electricity20_all_five_original_models_v1'
CONFIG = 'configs/reproducibility/fm_saits_electricity20_all_five_original_models.2026-10-14_v1.json'
previous_path = 'configs/reproducibility/fm_saits_electricity20_two_seed_publication.2026-10-11_v52.json'
previous = read(ROOT / previous_path)
verify_registration(previous)
rows = []
evidence = []
pins = [previous_path, 'scripts/flow_matching/publish_saits_two_seed_cohort_v52.py',
    'scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py',
    'scripts/flow_matching/audit_cfm_full_arrays_v1.py', 'tests/test_fm_saits_full_five_seed_cohort_v1.py',
    'tests/test_fm_saits_two_seed_cohort_v52.py', 'tests/test_fm_saits_completed_full_replay_batch_v1.py']
archives = []
for seed in range(26, 31):
    identity = f'saits_paper_saits_electricity_author_missing0.2_seed{seed}'
    if seed < 28:
        native_path = previous['model_evidence'][seed - 26]['native']
        review_path = previous['model_evidence'][seed - 26]['independent']
        guards = previous['native_guard_receipts'][2 * (seed - 26):2 * (seed - 26) + 2]
        config_paths = []
    else:
        nc = f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{identity}.native.json'
        rc = f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{identity}.review.json'
        native_config, review_config = read(ROOT / nc), read(ROOT / rc)
        for configuration in (native_config, review_config):
            for pin in configuration['source_receipts']:
                assert digest(ROOT / pin['path']) == pin['sha256'], pin['path']
        native_path = native_config['artifact_directory'] + '/checkpoint_replay_audit.json'
        review_path = review_config['output_directory'] + '/independent_full_replay_batch_verification.json'
        guards = [native_config['state_root'] + '/receipt.json', review_config['state_root'] + '/receipt.json']
        config_paths = [nc, rc]
    report, proof = read(ROOT / native_path), read(ROOT / review_path)
    assert report['full_checkpoint_contents_and_re_inference_passed'] and not report['cuda_context_initialized']
    assert report['metric_tolerance'] == proof['metric_tolerance'] == dict(rtol=2e-6, atol=2e-6)
    assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
    assert proof['native_replay_report_sha256'] == digest(ROOT / native_path)
    assert report['full_models_replayed'] == len(report['records']) == proof['full_native_models_verified']
    matches = [record for record in report['records'] if record['id'] == identity]
    checks = [record for record in proof['records'] if record['id'] == identity]
    assert len(matches) == len(checks) == 1
    record, checked = matches[0], checks[0]
    original_config = read(ROOT / record['config'])
    result = read(ROOT / record['result_path'])
    check_record(record, checked, original_config, result, seed)
    dataset = ROOT / original_config['dataset_root']
    output = ROOT / original_config['output_root']
    paths = dict(config=ROOT / record['config'], result=ROOT / record['result_path'],
        checkpoint=ROOT / record['checkpoint_path'], arrays=ROOT / record['replay_arrays'],
        dataset=dataset / 'datasets.h5', resume=output / 'resume.pt',
        original_group_statistics=output / 'group_statistics.npz', data_audit=dataset / 'data_audit.json')
    hashes = {name: digest(path) for name, path in paths.items()}
    assert hashes == checked['hashes']
    for guard_path in guards:
        guard = read(ROOT / guard_path)
        assert guard['exit_code'] == 0 and not guard['resource_guard_aborted']
        assert guard.get('full_native_model_replay_passed') or guard.get('full_native_batch_independent_review_passed')
    rows.append(dict(seed=seed, method='saits', dataset='electricity', nominal_missing_ratio=.2,
        dataset_sha256=hashes['dataset'], original_metrics=result['metrics'],
        independent_metrics=checked['independently_recomputed_metrics'],
        full_training_completion=checked['full_training_completion'],
        validation_best=result['validation_best'], original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],
        full_test_shape=checked['full_test_shape'], held_out_coordinates=checked['held_out_coordinates'],
        months=checked['months'], model_parameters=result['parameters'], full_artifact_hashes=hashes,
        citation=original_config['citation'], data_citation=original_config['data_citation'],
        protocol_boundaries=result['protocol_boundaries']))
    evidence.append(dict(id=identity, native_report=native_path, native_report_sha256=digest(ROOT / native_path),
        independent_review=review_path, independent_review_sha256=digest(ROOT / review_path), guard_receipts=guards))
    pins.extend([native_path, review_path, *guards, *config_paths, *[str(path) for path in paths.values()]])
    archives.append((identity, dict(original_result=paths['result'], native_report=ROOT / native_path,
        independent_proof=ROOT / review_path, native_guard=ROOT / guards[0], independent_guard=ROOT / guards[1])))

summary = summarize(rows)
folder = ROOT / BASE
folder.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, folder / 'closure_source.py')
shutil.copyfile(ROOT / 'tmp/saits_electricity20_five_seed_closure_tests_20261014_v1.xml', folder / 'validation.xml')
for identity, files in archives:
    target = folder / 'models' / identity
    target.mkdir(parents=True)
    for name, source in files.items():
        shutil.copyfile(source, target / (name + '.json'))


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


pins.extend([BASE + '/closure_source.py', BASE + '/validation.xml'])
save(CONFIG, dict(method='saits', dataset='electricity', nominal_missing_ratio=.2,
    original_training_seeds=[26, 27, 28, 29, 30], model_evidence=evidence,
    source_receipts=[dict(path=path, sha256=digest(ROOT / path)) for path in dict.fromkeys(pins)],
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)))
for pin in read(ROOT / CONFIG)['source_receipts']:
    assert digest(ROOT / pin['path']) == pin['sha256']
save(BASE + '/full_five_seed_cohort_summary.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    method='saits', dataset='electricity', nominal_missing_ratio=.2, **summary,
    required_SAITS_BRITS_models=210, all_original_experiments_complete=False, per_original_seed=rows,
    full_test_grid_cells_per_model=10730000, held_out_coordinates_per_model=1944331,
    actual_replacement_mask_target_fraction=1944331 / 10730000,
    additional_training_updates_in_this_closure=0, additional_model_reinference_in_this_closure=0,
    original_prediction_bitwise_equivalence_claimed=False,
    frozen_configuration=CONFIG, frozen_configuration_sha256=digest(ROOT / CONFIG)))
handles = []
for statepath in [ROOT / 'experiments/runs/fm_saits_completion_replay_queue_v1/status.json',
        Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/status.json')]:
    state = read(statepath)
    process = psutil.Process(state['pid'])
    assert process.create_time() == state['create_time']
    handles.append(dict(state_path=str(statepath), actual_pid=process.pid, actual_create_time=process.create_time(),
        actual_command=process.cmdline(), state=state))
assert all(item['id'] in handles[0]['state']['verified_original_ids'] for item in evidence)
save(BASE + '/actual_live_full210_source_and_collector_handles.json', handles)
save(BASE + '/closure_validation.json', dict(passed=True, all_five_original_full_training_and_native_reinference_proofs_verified=True,
    all_five_full_original_test_grid_and_month_statistics_independently_checked=True,
    every_current_result_checkpoint_resume_dataset_arrays_and_original_group_hash_rechecked=True,
    source_metrics_vs_replay_metrics_tolerance=dict(rtol=2e-6, atol=2e-6),
    original_five_seed_mean_STD_and_df4_t95_interval_independently_verified=True,
    frozen_config_sha256=digest(ROOT / CONFIG), additional_original_training_seed_slots=0,
    author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
lines = ['SAITS / Electricity / nominal20% missing: all five original seeds26..30.', '',
    'All five 11511000-parameter models completed the original max10000/patience30 protocol and passed full native checkpoint reinference plus independent review of all290x100x370 cells,1944331 held-out coordinates and10 months. Current full input/artifact/proof hashes were rechecked. No new fitting or inference was performed for this aggregation.', '',
    '|Metric|Five-model mean|SampleSTD|Student-t95% interval (df4)|', '|---|---|---|---|']
for metric, values in summary['summaries'].items():
    lines.append(f"|{metric}|{values['mean']:.9g}|{values['std']:.9g}|[{values['ci95_low']:.9g}, {values['ci95_high']:.9g}]|")
lines += ['', 'Training next_epoch by seed26..30: ' + ', '.join(str(row['full_training_completion']['next_epoch']) for row in rows) + '.', '',
    'These intervals summarize five training models on one frozen split. The ten-month block intervals remain separate within-model temporal sensitivity, recorded per seed in full_five_seed_cohort_summary.json. Actual replacement-mask coverage is18.1205126% of all grid cells. Original prediction arrays were not retained, so metric agreement is certified at the original2e-6 tolerance rather than bitwise source-prediction equivalence. Historical per-rate tuning/preprocessing conditions are unknown; author equivalence and strictTAB performance are not certified.', '',
    'The full210 SAITS/BRITS queue and45-paper/681-method-ablation/273-data-task objective remain active. Electricity nominal30% seed30 is independently tracked.', '',
    'Paper: https://arxiv.org/abs/2202.08516', 'Data: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014']
with (folder / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(lines) + '\n')
changed = ['scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py',
    'tests/test_fm_saits_full_five_seed_cohort_v1.py', CONFIG]
changed.extend(path.relative_to(ROOT).as_posix() for path in folder.rglob('*') if path.is_file())
save('tmp/saits_electricity20_five_seed_publication_manifest_20261014_v1.json', dict(
    previous_root='17cec7f8fb28141c532467b145a228e7ce072fa0', previous_parent='4d3082e682c11ef933eb6b6079e607bbbbdbe7a7', paths=sorted(changed)))
print(json.dumps(dict(full_original_training_seeds=5, summaries=summary['summaries'], exact_publication_files=len(changed))))
