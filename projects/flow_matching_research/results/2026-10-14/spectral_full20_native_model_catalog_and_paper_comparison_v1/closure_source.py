from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha
from scripts.flow_matching.audit_spectral_completed_cohorts_v1 import registration
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort

BASE = 'projects/flow_matching_research/results/2026-10-14/spectral_full20_native_model_catalog_and_paper_comparison_v1'
CONFIG = 'configs/reproducibility/fm_spectral_full20_native_catalog_and_comparison.2026-10-14_v1.json'
(ROOT / BASE).mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, ROOT / BASE / 'closure_source.py')


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


queue_path = 'configs/experiments/fm_spectral_original_resource_admission_queue.v3.json'
queue = read(ROOT / queue_path)
jobs = {job['id']: job for job in queue['jobs']}
pins = [queue_path, 'scripts/flow_matching/summarize_spectral_full_cohorts_v1.py',
    'tests/test_fm_spectral_full_cohort_summary_v1.py', BASE + '/closure_source.py']
records = []
grouped = defaultdict(list)
source_receipts = []
for plan_path in ['configs/reproducibility/fm_spectral_energy_etth_fmri_full14_native_reviews.2026-10-14_v1.json',
        'configs/reproducibility/fm_spectral_mujoco_seed42_full_native_review.2026-10-14_v1.json']:
    plan = read(ROOT / plan_path)
    registration(plan)
    receipt_path = plan['state_root'] + '/receipt.json'
    receipt = read(ROOT / receipt_path)
    assert receipt['passed'] and receipt['full_native_artifact_audits_completed'] == len(plan['audits'])
    assert len(receipt['records']) == len(plan['audits'])
    pins.extend([plan_path, receipt_path])
    source_receipts.append(receipt_path)
    for record in receipt['records']:
        assert record['native_exit_code'] == 0 and record['full_native_artifact_audit_passed'] and not record['resource_guard_aborted']
        assert sha(ROOT / record['configuration']) == record['configuration_sha256']
        assert sha(Path(record['proof_path'])) == record['proof_sha256']
        records.append(record)

for seed in range(42, 47):
    version = 'v1' if seed == 42 else 'v3' if seed == 46 else 'v2'
    path = f'configs/reproducibility/fm_spectral_diffusion_ts_stocks_seed{seed}_full_audit.2026-10-11_{version}.json'
    config = read(ROOT / path)
    for pin in config['source_receipts']:
        assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
    proof_path = config['output_directory'] + '/independent_full_native_artifact_audit.json'
    pins.extend([path, proof_path, config['output_directory'] + '/validation.json'])
    records.append(dict(id=config['job_id'], configuration=path, configuration_sha256=sha(ROOT / path),
        proof_path=proof_path, proof_sha256=sha(ROOT / proof_path), reused_existing_full_native_review=True))

assert len(records) == len({record['id'] for record in records}) == 20
catalog = []
for record in records:
    job = jobs[record['id']]
    config = read(ROOT / record['configuration'])
    proof_path = Path(record['proof_path'])
    if proof_path.is_absolute():
        proof_path = proof_path.relative_to(ROOT)
    proof_path = proof_path.as_posix()
    proof = read(ROOT / proof_path)
    validation_path = str(Path(proof_path).parent / 'validation.json').replace('\\', '/')
    validation = read(ROOT / validation_path)
    assert validation['passed'] and validation['outputs']['independent_full_native_artifact_audit.json'] == sha(ROOT / proof_path)
    assert proof['canonical_id'] == job['id'] and proof['seed'] == job['seed']
    assert proof['original_optimizer_updates'] == job['training_updates']
    assert proof['original_training_minibatches'] == job['training_updates'] * job['gradient_accumulate_every']
    assert proof['actual_full_original_checkpoint_tensors_restored_bitwise']
    assert proof['complete_original_reference_array_bitwise_verified'] and proof['all_original_generated_arrays_finite']
    assert proof['all_four_five_repeat_evaluator_statistics_independently_recomputed']
    assert not proof['learned_evaluators_independently_retrained'] and not proof['generative_model_reinferred_by_this_audit']
    result_path = job['output_directory'] + '/result.json'
    terminal_path = config['terminal_receipt']
    result = read(ROOT / result_path)
    terminal = read(ROOT / terminal_path)
    assert sha(ROOT / result_path) == proof['original_result_sha256']
    assert sha(ROOT / terminal_path) == proof['actual_terminal_receipt_sha256']
    assert terminal['exitcode'] == 0 and terminal['full_result_validated'] and not terminal['resource_guard_aborted']
    for artifact in proof['artifacts']:
        assert sha(ROOT / artifact['path']) == artifact['sha256']
    labelled = dict(proof, model=job['model'], dataset=job['dataset'], data_track=job['data_track'])
    grouped[(job['model'], job['dataset'], job['data_track'])].append(labelled)
    archive = BASE + '/original_results/' + job['id']
    (ROOT / archive).mkdir(parents=True)
    shutil.copyfile(ROOT / result_path, ROOT / archive / 'result.json')
    shutil.copyfile(ROOT / terminal_path, ROOT / archive / 'terminal_receipt.json')
    pins.extend([record['configuration'], proof_path, validation_path, result_path, terminal_path])
    catalog.append(dict(canonical_id=job['id'], model=job['model'], dataset=job['dataset'], seed=job['seed'],
        optimizer_updates=proof['original_optimizer_updates'], training_minibatches=proof['original_training_minibatches'],
        complete_reference_shape=proof['complete_reference_shape'], complete_generated_shape=proof['complete_generated_shape'],
        model_parameters=proof['actual_model_parameters'], metrics=proof['metrics'],
        original_result_sha256=proof['original_result_sha256'], artifacts=proof['artifacts'],
        full_native_proof=proof_path, full_native_proof_sha256=sha(ROOT / proof_path),
        independently_retrained_learned_evaluators=False, generative_output_reinferred=False))

claims_path = 'configs/reproducibility/spectral_mean_flow_table1_claims.v1.json'
claims = read(ROOT / claims_path)['claims']
pins.extend([claims_path, *[claim['source_pdf'] for claim in claims]])
for claim in claims:
    assert sha(ROOT / claim['source_pdf']) == claim['source_pdf_sha256']
summaries = []
for key, proofs in sorted(grouped.items()):
    summary = summarize_cohort(proofs)
    for metric, row in summary['metrics'].items():
        claim = next(item for item in claims if item['method'] == key[0] and item['dataset'] == key[1] and item['metric'] == metric)
        local = row['full_cohort_mean']
        row.update(paper_mean=claim['paper_mean'], paper_reported_spread=claim['paper_reported_spread'],
            paper_spread_scope=claim['spread_type'], paper_source_page=claim['source_page'], paper_source_table=claim['source_table'],
            full_five_seed_signed_difference_from_paper=None if local is None else local - claim['paper_mean'],
            relative_difference_from_paper=None if local is None or claim['paper_mean'] == 0 else (local - claim['paper_mean']) / claim['paper_mean'])
    summaries.append(summary)

original_models = len(catalog)
save(BASE + '/unique20_full_native_model_catalog.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_native_generators=original_models, original_registered_generation_jobs=len(jobs),
    newly_independently_artifact_reviewed_models=15, reused_previous_full_native_reviews=5,
    original_total_optimizer_updates=sum(item['optimizer_updates'] for item in catalog),
    original_total_training_minibatches=sum(item['training_minibatches'] for item in catalog),
    original_total_full_evaluator_runs=original_models * 4 * 5, records=catalog,
    full_generative_reinference_certified=False, learned_evaluator_retraining_certified=False,
    author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
save(BASE + '/five_cohort_metrics_and_paper_comparison.json', dict(cohorts=summaries,
    boundary='Complete five-training-seed statistics only for Stocks/Energy/ETTh. fMRI four-model and MuJoCo one-model per-seed measurements are retained; full-cohort means/STD/CI and aggregate paper differences stay null. Table1 comparison is descriptive, no equivalence certification. Five evaluator repeats per model are not five training seeds. Original all-source scaling and reconstruction evaluation do not constitute strictTAB anomaly detection.',
    all_original_experiments_complete=False))
save(CONFIG, dict(original_queue=queue_path, completed_canonical_ids=[row['canonical_id'] for row in catalog],
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    required_original_training_seeds=[42, 43, 44, 45, 46],
    source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in dict.fromkeys(pins)]))
for pin in read(ROOT / CONFIG)['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256']
save(BASE + '/closure_validation.json', dict(passed=True, frozen_config=CONFIG, frozen_config_sha256=sha(ROOT / CONFIG),
    all20_canonical_ids_unique=True, all20_full_source_result_and_guard_receipts_checked=True,
    all20_full_native_proof_hashes_and_checkpoint_generated_reference_hashes_checked=True,
    partial_fmri_and_mujoco_cohort_aggregates_are_null=True, original_full_scope_preserved=True))
live = []
for process in psutil.process_iter(['pid', 'name']):
    try:
        command = process.cmdline()
        if any('flow_matching' in item for item in command):
            cpu = process.cpu_times()
            live.append(dict(pid=process.pid, create_time=process.create_time(), command=command,
                cpu_seconds=cpu.user + cpu.system))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
save(BASE + '/actual_remaining_full_scope_live_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), actual_handles=live))
lines = ['Full native generation result catalogue: 20/70 original jobs independently artifact-reviewed.', '',
    '15 new native reviews +5 previously closed Stocks models, unique by canonical ID. All checkpoint/EMA/optimizer tensors, complete arrays and evaluator statistics passed the unchanged v1 checks. This is artifact/statistic reproduction; generative reinference and independent learned-evaluator retraining are not certified.', '',
    '|Dataset|Reviewed original seeds|Full cohort|Context-FID|Correlational|Discriminative|Predictive|',
    '|---|---|---|---|---|---|---|']
for summary in summaries:
    values = [summary['metrics'][key]['full_cohort_mean'] for key in ['context_fid', 'correlational', 'discriminative', 'predictive']]
    lines.append('|'+summary['dataset']+'|'+','.join(map(str, summary['independently_reviewed_training_seeds']))+'|'+('5/5' if summary['full_five_seed_cohort_complete'] else 'pending')+'|'+'|'.join('pending' if value is None else f'{value:.9g}' for value in values)+'|')
lines += ['', 'See unique20_full_native_model_catalog.json for every method/dataset/seed metric, full data shape and SHA256 provenance. five_cohort_metrics_and_paper_comparison.json retains all model-specific evaluator repetitions, five-model sample STD/t95% intervals and cited paper differences. fMRI seed43 and MuJoCo seeds43..46 remain required; no surviving-seed cohort aggregate is reported.', '',
    'The intervals use five training-model evaluator means (df4), not25 independent training seeds. They are descriptive small-n intervals; paper reported spread does not identify the same uncertainty scope. Original reconstruction/all-source normalization boundaries remain; these generation results do not establish author equivalence or strictTAB anomaly-detection performance.', '',
    'The full45-paper/681-method-ablation/273-data-task scope remains active. Full budgets, frozen scientific sources, live numerical workers, original raw data and the four user-modified reports are preserved.']
with (ROOT / BASE / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(lines)+'\n')
changed = ['scripts/flow_matching/audit_spectral_completed_generation_v2.py',
    'scripts/flow_matching/audit_spectral_completed_cohorts_v1.py', 'scripts/flow_matching/summarize_spectral_full_cohorts_v1.py',
    'tests/test_fm_spectral_completed_cohorts_v1.py', 'tests/test_fm_spectral_full_cohort_summary_v1.py', CONFIG]
for path in ['configs/reproducibility/fm_spectral_energy_etth_fmri_full14_native_reviews.2026-10-14_v1.json',
        'configs/reproducibility/fm_spectral_mujoco_seed42_full_native_review.2026-10-14_v1.json']:
    changed.append(path)
    changed.extend(entry['configuration'] for entry in read(ROOT / path)['audits'])
for name in [BASE,
        'projects/flow_matching_research/results/2026-10-14/spectral_energy_etth_fmri_full14_native_reviews_v1',
        'projects/flow_matching_research/results/2026-10-14/spectral_mujoco_seed42_full_native_review_v1']:
    changed.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / name).rglob('*') if path.is_file())
save('tmp/spectral_full20_native_models_publication_manifest_20261014_v1.json', dict(
    previous_root='4aa07178d3538850dcd67c0fa510de9ddac2684c', previous_parent='50671590af381444d8aa16de7b3736c8b421d33d',
    paths=sorted(set(changed))))
print(json.dumps(dict(verified_full_native_models=20, newly_artifact_reviewed=15,
    complete_five_seed_cohorts=[row['dataset'] for row in summaries if row['full_five_seed_cohort_complete']],
    optimizer_updates=sum(row['optimizer_updates'] for row in catalog), exact_publication_files=len(set(changed)))))
