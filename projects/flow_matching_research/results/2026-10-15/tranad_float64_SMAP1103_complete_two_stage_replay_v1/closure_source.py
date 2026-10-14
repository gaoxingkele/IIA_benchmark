"""Close both actual full TranAD audits and prepare an exact publication manifest."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import audit_tranad_float64_constant_fit_complete_v1 as audit

root = audit.ROOT
base = 'projects/flow_matching_research/results/2026-10-15/tranad_float64_SMAP1103_complete_two_stage_replay_v1'
registration = 'projects/flow_matching_research/results/2026-10-15/tranad_float64_SMAP1103_constant_fit_full_replay_v1'
sources, proofs, receipts = {}, {}, {}
configs = []
for mode in ('point', 'native'):
    path = 'configs/reproducibility/fm_tranad_float64_SMAP1103_constant_fit_full_replay.' + mode + '.2026-10-15_v1.json'
    config = audit.read(root / path)
    queue, job, result, model = audit.binding(config)
    proof_path = Path(config['output_directory']) / 'independent_full_saved_score_verification.json'
    receipt_path = Path(config['state_root']) / 'receipt.json'
    proof, receipt = audit.read(root / proof_path), audit.read(root / receipt_path)
    assert proof['passed'] and proof['canonical_id'] == audit.CANONICAL and proof['mode'] == mode
    assert proof['configuration_sha256'] == audit.sha(root / path)
    assert proof['original_result_sha256'] == audit.sha(root / job['output_directory'] / 'result.json')
    assert proof['original_checkpoint_sha256'] == result['checkpoint_sha256']
    assert proof['original_scores_sha256'] == result['scores_sha256']
    assert proof['original_scope'] == audit.SCOPE and proof['original_queue_jobs'] == 5
    assert proof['original_precision_control_jobs'] == 25 and proof['original_data_control_jobs'] == 291
    assert proof['all_original_point_core_event_bootstrap_metrics_exact']
    assert receipt['exit_code'] == 0 and receipt['full_saved_score_review_passed'] and not receipt['resource_guard_aborted']
    assert (proof['full_entities'], proof['full_train_points'], proof['full_validation_points'], proof['full_test_points']) == (51, 101565, 26684, 408160)
    assert proof['parameters'] == 58073 and proof['configured_training_epochs'] == 5
    assert len(proof['coverage']) == 153 and len({(r['entity'], r['split']) for r in proof['coverage']}) == 153
    sources.update({mode + '_configuration': root / path, mode + '_complete_proof': root / proof_path,
                    mode + '_guarded_receipt': root / receipt_path})
    proofs[mode], receipts[mode] = proof, receipt
    configs.append(path)
for key in ('full_entities', 'full_test_points', 'full_validation_points', 'full_train_points', 'coverage',
            'independently_recomputed_primary_metrics', 'conditional_entity_cluster_f1_ci95',
            'original_event_metrics', 'original_test_calibrated_TAB_core_diagnostic'):
    assert proofs['point'][key] == proofs['native'][key], key
native = proofs['native']
assert native['full_native_checkpoint_reinference_passed'] and native['all_replayed_timestamp_scores_bitwise_equal']
assert native['replay_scores_sha256'] == native['original_scores_sha256'] == result['scores_sha256']
assert audit.sha(Path(config['output_directory']) / 'replayed_scores.npz') == result['scores_sha256']
sources['original_result'] = root / job['output_directory'] / 'result.json'
sources['original_training_guard'] = root / config['terminal_receipt']
sources['original_model_configuration'] = root / job['model_config']
xml = root / 'tmp/TranAD_SMAP1103_full_replay_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 33
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)
windows = sum(r['windows'] for r in native['coverage'] if r['split'] == 'train')
steps = (windows + 127) // 128
folder = root / base
folder.mkdir(parents=True, exist_ok=False)
for name, path in sources.items():
    shutil.copyfile(path, folder / (name + '.json'))
shutil.copyfile(xml, folder / 'validation.xml')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_id=audit.CANONICAL,
    full_registered_two_stage_audit_complete=True, original_training_guard_exit_code=0,
    independently_recomputed_primary_metrics=native['independently_recomputed_primary_metrics'],
    conditional_entity_cluster_f1_ci95=native['conditional_entity_cluster_f1_ci95'],
    original_event_metrics=native['original_event_metrics'],
    original_test_calibrated_TAB_core_diagnostic=native['original_test_calibrated_TAB_core_diagnostic'],
    full_entities=51, full_train_points=101565, full_validation_points=26684, full_test_points=408160,
    parameters=58073, precision='float64_fit_and_score', cpu_threads=2, batch_size=128,
    complete_score_arrays=153, complete_label_arrays=51,
    all_full_train_validation_test_scores_bitwise_equal=True,
    original_and_replay_score_archives_same_sha256=True, scores_sha256=result['scores_sha256'],
    checkpoint_sha256=result['checkpoint_sha256'], configured_epochs=5,
    full_registered_training_windows=windows, configured_optimizer_steps_per_epoch_from_original_code=steps,
    configured_optimizer_steps_all_epochs_from_original_code=5 * steps,
    actual_per_epoch_optimizer_update_trace_available=False, additional_optimizer_updates=0,
    additional_training_seed_slots=0, full_TAB_range_metrics_completed=False,
    all_original_point_core_event_bootstrap_metrics_exact=True,
    required_original_cohort_seeds=[1103,1104,1105,1106,1107], verified_cohort_seeds_in_this_closure=[1103],
    complete_five_seed_aggregate=False, original_full_precision_controls=25,
    selected_corrected_SMAP_precision_queue_jobs=5, full_original_baseline_queue_jobs=35,
    full_corrected_data_controls=291, original_scope=audit.SCOPE,
    historical_author_equivalence_certified=False, full_TAB_training_harness_certified=False,
    all_original_experiments_complete=False, current_full_proof_hashes={k:audit.sha(p) for k,p in sources.items()},
    validation_tests_passed=33, citation=model['citation'], data_citations=model['data_citations'],
    fidelity_boundaries=model['known_deviations'])
(folder / 'complete_two_stage_native_and_independent_saved_score_closure.json').write_text(json.dumps(closure, indent=2) + '\n', encoding='utf-8')
(folder / 'README.md').write_text(
    'TranAD / SMAP / corrected constant-fit preprocessing / full float64 fit and score / seed1103.\n\n'
    'Full original training exited0 without resource abort. Both independent saved-score review and full native checkpoint inference exited0 without abort. All153score arrays and51label arrays across51entities (101565train,26684validation,408160test points) match bitwise, including the compressed archive SHA256. Original float64,58073parameters,CPU2,batch128 and5configured epochs retained.\n\n'
    'Validation1% threshold, noPA primary metrics: ' + json.dumps(closure['independently_recomputed_primary_metrics']) + '\n'
    'Conditional entity-cluster F1 sensitivity interval: ' + json.dumps(closure['conditional_entity_cluster_f1_ci95']) + '\n'
    'This is one fitted model with200entity-bootstrap resamples, not five-training-seed uncertainty. Original test-calibrated/test-label-selected diagnostics remain separate.\n\n'
    f'Training geometry implies{windows}original nonoverlapping/end-aligned windows,{steps}configured steps per epoch,{5*steps}configured steps across5epochs. These are derived from frozen executed code and original data, not an independently measured optimizer-update trace. No per-epoch trace was saved.\n\n'
    'The model restores the pinned author float64 precision, but local window construction/preprocessing/thresholds still differ from the full historical author padding/POT/environment. No author-pipeline or fullTAB-training-harness certification. All25original precision controls,35baseline slots,5corrected SMAP precision seeds,291corrected range-evaluation jobs and45papers/681methods/273data obligations remain required. 33relevant checks passed.\n\n'
    'Paper: https://doi.org/10.14778/3514061.3514067\nData/evaluation citation: https://github.com/decisionintelligence/TAB\n', encoding='utf-8')
paths = ['scripts/flow_matching/audit_tranad_float64_constant_fit_complete_v1.py',
         'tests/test_fm_tranad_float64_constant_fit_complete_v1.py', *configs]
for directory in (root / registration, folder):
    paths.extend(p.relative_to(root).as_posix() for p in directory.rglob('*') if p.is_file() and not p.name.endswith('.log'))
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest = dict(previous_root='5233bbabd28b684c14906c9eff110f8b7e3be46b',
                previous_parent='f932131cc18a618011e1c7bd4256e87f8f0e78f0', paths=sorted(set(paths)))
(root / 'tmp/TranAD_SMAP1103_full_replay_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(complete_two_stage_audit=True, configured_updates=5*steps, training_windows=windows,
    primary=closure['independently_recomputed_primary_metrics'], exact_files=len(manifest['paths']), validation_tests=33)))
