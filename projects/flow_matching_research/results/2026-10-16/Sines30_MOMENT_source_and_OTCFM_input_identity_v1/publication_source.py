"""Archive verified progress without promoting failed inputs or pending replays."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_tsad_queue import complete_result
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v3 as reconstruction

BASE = 'projects/flow_matching_research/results/2026-10-16/Sines30_MOMENT_source_and_OTCFM_input_identity_v1'
REVIEW = 'projects/flow_matching_research/results/2026-10-16/spectral_sines_paper_seed42_full_native_review_v1'
CLOSURE = 'projects/flow_matching_research/results/2026-10-16/spectral_after29_PaperSines42_full30_native_model_increment_v1'
folder = ROOT / BASE
folder.mkdir(parents=True, exist_ok=False)


def save(name, value):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tests(path, expected):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(x.get('tests', 0)) for x in suites) == expected
    assert all(int(x.get('failures', 0)) == int(x.get('errors', 0)) == 0 for x in suites)


tests(ROOT / REVIEW / 'validation.xml', 43)
tests(ROOT / CLOSURE / 'validation.xml', 10)
test_path = Path('F:/aicoding/IIA_Data/experiments/otcfm_multiome13176_reconstruction_v3_tests.xml')
tests(test_path, 57)
shutil.copyfile(test_path, folder / 'input_validation57.xml')
shutil.copyfile(__file__, folder / 'publication_source.py')

queue_path = 'configs/experiments/fm_tsad_constant_fit_moment_pretrained_queue.v2.json'
queue = read(ROOT / queue_path)
rows = []
pins = []
for context in ['release512', 'TAB100']:
    job = next(j for j in queue['jobs'] if j['id'] == f'fm_moment_pretrained_base__{context}__smap__seed42__constant_fit_v2')
    assert complete_result(ROOT, job)
    result_path = ROOT / job['output_directory'] / 'result.json'
    result = read(result_path)
    guard_path = ROOT / 'experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/moment_pretrained/receipts' / (job['id'] + '.json')
    guard = read(guard_path)
    assert guard['job_id'] == job['id'] and guard['exit_code'] == 0
    assert not guard['resource_guard_aborted'] and guard['full_registered_budget_retained']
    assert guard['queue_sha256'] == sha(ROOT / queue_path)
    for pin in job['frozen_files']:
        assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
    manifest = read(ROOT / job['dataset_manifest_path'])
    data = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    assert len(data['entities']) == 51 and data['grouped_splits_verified']
    assert sha(ROOT / data['input_npz']) == data['input_sha256']
    metrics = result['metrics']
    row = dict(id=job['id'], context=context, external_window=job['window'], entities=51,
        dataset_input_sha256=data['input_sha256'], primary=metrics['strict']['1']['micro'],
        primary_f1_entity_cluster_CI95=metrics['strict']['1']['f1_ci95'],
        primary_calibration=metrics['strict']['1']['calibration'],
        diagnostic_best_test_label_selected_plain_F1=metrics['tab_core_diagnostic']['best_grid_f_score'],
        diagnostic_best_test_label_selected_PA_F1=metrics['tab_core_diagnostic']['best_grid_adjust_f_score'],
        original_result_sha256=sha(result_path), original_scores_sha256=result['scores_sha256'],
        original_guard_sha256=sha(guard_path), actual_source_complete=True,
        fresh_independent_numerical_replay_certified_by_this_packet=False,
        historical_paper_equivalence=False, full54_entity_SMAP=False, full_TAB_harness=False)
    rows.append(row)
    shutil.copyfile(result_path, folder / ('moment_' + context + '_source_result.json'))
    shutil.copyfile(guard_path, folder / ('moment_' + context + '_source_guard.json'))
    pins.extend([str(result_path), str(guard_path), str(ROOT / data['input_npz'])])
assert rows[0]['dataset_input_sha256'] == rows[1]['dataset_input_sha256']
save('moment_same_data_context_comparison.json', dict(rows=rows,
    comparison='Source results on the identical legacy51-entity SMAP input; strict validation-only threshold/noPA versus explicitly test-label-selected diagnostics.',
    paired_training_seed_uncertainty_certified=False, all_original_experiments_complete=False))

config = read(ROOT / reconstruction.CONFIG)
metadata = {}
for role, entry in config['inputs'].items():
    with h5py.File(ROOT / entry['path'], 'r') as f:
        donors = reconstruction.column(f['obs'], 'donor').astype(str)
        days = reconstruction.column(f['obs'], 'day').astype(int)
        metadata[role] = dict(cells=len(days), donor_day_counts={f'{d}/day{t}': n
            for (d, t), n in sorted(Counter(zip(donors.tolist(), days.tolist())).items())})
save('actual_full_metadata_rosters.json', metadata)
for version in [1, 2, 3]:
    directory = ROOT / f'experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v{version}'
    guard = read(directory / 'guard_receipt.json')
    assert not guard['passed'] and guard['exit_code'] == 1 and not guard['resource_guard_aborted']
    handle = guard['exact_child_handle']
    try:
        p = psutil.Process(handle['pid'])
        assert abs(p.create_time() - handle['create_time']) > .01
    except psutil.NoSuchProcess:
        pass
    for source in ['guard_receipt.json', 'worker.stderr.log', 'status.json', 'launch.json']:
        shutil.copyfile(directory / source, folder / f'input_v{version}_{source}')
    output = ROOT / f'data/public_datasets/flow_matching/processed_otcfm_multiome13176_public_pca5_v{version}'
    assert not (output / 'paper13176_author_loader_input.npz').exists()
    assert not (output / 'paper_input_reconstruction_report.json').exists()
    shutil.copyfile(output / 'progress.json', folder / f'input_v{version}_last_progress.json')
    pins.extend([str(directory / 'guard_receipt.json'), str(directory / 'worker.stderr.log')])
failure = read(folder / 'input_v3_last_progress.json')
assert failure['author_row'] == 1755 and failure['expression_candidates'] == 0
assert failure['closest_public_expression_absolute_error'] > .06
save('input_qualification.json', dict(original_full_reference_cells=33701,
    public_reference_candidates=33704, full_paper_target_cells=35384, features=23418,
    frozen_encoding=config['encoding_transform'], frozen_tolerance=config['encoding_tolerance'],
    numerical_identity_gate_passed=False, PCA_input_completed=False, original_author_training_started=False,
    first_reported_expression_counterexample=failure,
    entire_reference_overlap_count_known=False, cause_of_reference_difference_known=False,
    scope_or_tolerance_reduced=False, failed_partial_outputs_preserved=True,
    next_required_work='Audit the full reference overlap and encoding/cell provenance; do not infer whole-matrix identity from matched anchors or silently fit PCA after a failed gate.',
    all_original_experiments_complete=False))

handles = []
for pid in [36912, 38680, 41992, 2972, 36640, 25136, 22316, 29336, 11436, 42104]:
    p = psutil.Process(pid)
    handles.append(dict(pid=pid, create_time=p.create_time(), command=p.cmdline(),
        cpu_seconds=sum(p.cpu_times()[:2]), children=[dict(pid=x.pid, create_time=x.create_time(), command=x.cmdline())
            for x in p.children(recursive=True)]))
save('actual_remaining_experiment_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), actual_handles=handles))
closed = read(ROOT / CLOSURE / 'unique30_full_native_model_catalog.json')
assert closed['full_native_generators'] == 30 and closed['original_registered_generation_jobs'] == 70
assert closed['original_total_optimizer_updates'] == 467000
assert closed['original_total_training_minibatches'] == 934000
assert closed['original_total_full_evaluator_runs'] == 600
save('iteration_validation.json', dict(passed=True, regression_tests_passed=110,
    new_complete_generator_artifact_reviews=1, total_complete_generator_artifact_reviews=30,
    original_generation_slots_required=70, paper_Sines_independent_training_seeds=[42],
    paper_Sines_missing_training_seeds=[43, 44, 45, 46], failed_input_attempts_preserved=3,
    input_reconstruction_complete=False, full_reference_identity_proven=False,
    MOMENT_new_source_complete=True, MOMENT_new_fresh_replay_claimed=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_tasks=273, paper_scalar_targets=702),
    all_original_experiments_complete=False, archived_source_hashes=[dict(path=p, sha256=sha(p)) for p in pins]))
text = '''Diffusion-TS paper-distribution Sines seed42 completed its full original training, sampling and four five-repeat evaluators and passed the native artifact audit. The independent artifact-reviewed catalog is30/70; original paper-Sines seeds43..46 remain required. Five evaluator repeats are one trained model. See the neighboring spectral_after29_PaperSines42_full30_native_model_increment_v1 packet for all30 records and seven separate cohorts.

MOMENT Base100 SMAP constant-fit source has a successful original guard. On the identical51-entity input, strict validation-only/noPA F1 is0.0484329 (512-window:0.107917). Test-label-selected PA diagnostic F1 is0.756096 (512-window:0.738891). These protocols are recorded separately. Entity-bootstrap intervals are conditional on one fixed model and threshold, not training-seed or paired-method uncertainty. Fresh native/independent replay remains with existing controllers, not duplicated here. This legacy51-entity track is not canonical54-entity SMAP or completeTAB.

OT-CFM Multiome input reconstruction remains unqualified. The reference donor has33701 author cells versus33704 public candidates at days2,3,4,7. The paper-target donor still has35384 cells and23418 genes. Corrected isolated versions retained all cells/features, original encoding/tolerance, full matrix gate, PCA/whitening and resource safeguards. Actual expression matching then failed at author row1755: gene13316 value4.060491085; nearest fixed-transform public candidate differs by0.064515139, exceeding frozen tolerance. The full overlap and discrepancy cause remain unknown. No PCA/whitened training input or original single-cell training was produced. All three terminal failures and their last progress are preserved. No zero-cell, QC, complete-overlap or historical-paper equivalence claim is made.

110 relevant regression checks passed. Existing numerical workers were not interrupted. The complete45-paper/681-method-baseline-ablation/273-data-task/702-paper-target objective remains active.
'''
(folder / 'README.md').write_text(text, encoding='utf-8', newline='\n')
for path in [folder, ROOT / REVIEW, ROOT / CLOSURE]:
    (path / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8', newline='\n')

targets = [BASE, REVIEW, CLOSURE,
    'scripts/flow_matching/audit_spectral_completed_cohorts_v2.py', 'tests/test_fm_spectral_completed_cohorts_v2.py',
    'scripts/flow_matching/reconstruct_otcfm_multiome13176_public_v2.py',
    'scripts/flow_matching/reconstruct_otcfm_multiome13176_public_v3.py',
    'tests/test_fm_otcfm_multiome13176_reconstruction_v2.py',
    'tests/test_fm_otcfm_multiome13176_reconstruction_v3.py',
    'configs/reproducibility/fm_otcfm_multiome13176_public_reconstruction.v2.json',
    'configs/reproducibility/fm_otcfm_multiome13176_public_reconstruction.v3.json',
    'configs/reproducibility/fm_spectral_sines_paper_seed42_full_native_review.2026-10-16_v1.json',
    'configs/reproducibility/fm_spectral_sines_paper42_review_20261016_v1',
    'configs/reproducibility/fm_spectral_after29_PaperSines42_full30_native_increment.2026-10-16_v1.json']
changed = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    changed.update(p for p in subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines()
                   if not p.endswith('.log'))
assert changed and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT).strip()
manifest = dict(previous_root=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    previous_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent / 'flow-matching-research', text=True).strip(),
    paths=sorted(changed))
(ROOT / 'tmp/Sines30_MOMENT_OTCFM_identity_publication_manifest_20261016_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(changed), tests=110, closed_generation_models=30,
    input_reconstruction_complete=False, all_original_experiments_complete=False)))
