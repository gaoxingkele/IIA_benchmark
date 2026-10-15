"""Archive actually live full215statistics automation without promoting fixtures."""
from datetime import datetime, timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_anomaly_transformer_full_cohort_statistics_v1 as runner
from scripts.flow_matching.register_anomaly_transformer_full_cohort_statistics_v1 import CONFIG

ROOT, read, sha, write = runner.ROOT, runner.read, runner.sha, runner.write
runtime = read(ROOT/CONFIG); producer, cases, contract, selected = runner.verify(runtime)
launch = read(ROOT/runtime['state_root']/'launch.json'); process = psutil.Process(launch['pid'])
assert process.create_time() == launch['create_time'] and process.cmdline() == launch['command']
state = read(ROOT/runtime['state_root']/'status.json')
assert state['pid'] == process.pid and state['create_time'] == process.create_time()
assert state['required_models'] == 500 and state['required_tasks'] == 215
assert state['completed_task_count'] == 0 and not state['failed_preserved']
assert state['state'] == 'waiting_all_five_actual_full_model_proofs' and state['active_pid'] is None
xml = ROOT/'tmp/AnomalyTransformer_full_cohort_statistics_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 21
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures','errors','skipped'])
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_full_five_seed_statistics_v1'
folder.mkdir(exist_ok=False)
sources = {
    'frozen_full215_statistics_runtime.json': ROOT/CONFIG,
    'frozen_full500_producer_runtime.json': ROOT/runner.PRODUCER,
    'statistics_queue_and_calculation_source.py': ROOT/'scripts/flow_matching/run_anomaly_transformer_full_cohort_statistics_v1.py',
    'registration_source.py': ROOT/'scripts/flow_matching/register_anomaly_transformer_full_cohort_statistics_v1.py',
    'validation_tests.py': ROOT/'tests/test_fm_anomaly_transformer_full_cohort_statistics_v1.py',
    'validation.xml': xml,
    'existing_df4_quantile_reference_source.py': ROOT/runtime['student_t_critical_reference'],
    'primary_paper_contract.json': ROOT/read(ROOT/runner.producer.FAMILIES['main'][0])['contract']}
for name,path in sources.items():
    shutil.copyfile(path,folder/name)
write(folder/'actual_live_waiting_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    launch=launch, actual_live=True, private_bytes=process.memory_info().private, state_at_capture=state,
    runtime_configuration=CONFIG, runtime_configuration_sha256=sha(ROOT/CONFIG),
    real_completed_cohort_statistics=0, new_benchmark_completions_claimed=0,
    original_training_workers_stopped_or_restarted=0))
write(folder/'full_task_and_model_coverage.json',dict(unique_required_training_models=500,
    statistics_tasks=215,task_counts={k:sum(t['kind']==k for t in selected.values()) for k in ['cohort','paired','macro']},
    tasks=list(selected.values()), seeds=runner.SEEDS,datasets=runner.DATASETS,
    primary_endpoint=runtime['primary_endpoint'],score_criterion='predeclared_multiplication_or_plain_reconstruction',
    all_score_criteria_and_all_PA_noPA_controls_retained=True,
    model_proof_requirements='original training guard;full native replay guard;independent all-array review guard',
    samples_per_dataset_CI=5,samples_per_macro_CI=5,df=4,confidence_level=.95,
    paired_test_support=runtime['paired_test_support'],paired_calibration=runtime['paired_calibration'],
    shared_prefix_preserves_original_model_window_context=True,
    fixed_delta_controls_are_diagnostic=True,statistics_fixtures_are_benchmark_results=False,
    no_partial_cohort_or_missing_seed_deletion=True,no_multiple_comparison_or_generalization_guarantee=True,
    original_scope=runtime['original_scope'],paper_equivalence_certified=False,strict_TAB_result=False,
    other_paper_experiments_and_full_project_complete=False))
(folder/'README.md').write_text(
    'Full five-seed and five-dataset reconstruction statistics automation\n\n'
    'The actually live statistics collector waits for all original five seeds and every passed full-model '
    'verification guard. It retains all500unique paper reconstruction models and automatically creates '
    '100dataset/variant cohorts,95variant-vs-main paired comparisons and20five-dataset macro summaries '
    '(215read-only statistics tasks). Zero actual cohort statistics or new benchmark completions are '
    'claimed at this capture. All other original methods/papers and all45/681/273minimumscope remain.\n\n'
    'Every dataset cohort keeps every original score criterion,validation-ratio and fixed-delta control, '
    'PA/noPA confusion counts,continuousAUROC/AP and all sevenROC operating points separately. Complete '
    'five-seed summaries give mean,sampleSD,standard error,unclippeddf4Student95%interval,median,range '
    'and five leave-one-seed-out means. Missing seeds or mixed undefined metrics cannot be silently '
    'dropped. These are descriptive training-seed intervals,not dataset-generalization or multiplicity '
    'guarantees. Leave-one-out calculations do not create extra training seeds.\n\n'
    'Pairing matches original seed and dataset. Window sensitivity can change covered test tails,so '
    'the paired stage independently recomputes both models on their common frozen covered truth prefix '
    'while preserving each model full original validation calibration and original model-window context. '
    'No new model inference or threshold tuning occurs. The predeclared primary endpoint is validation-only '
    'noPAF1,with PA reported separately; fixed-delta controls remain diagnostic. Positive differences favor '
    'the variant. Every paired primary criterion is fixed before test results:all.multiplication except '
    'plain Transformer all.reconstruction. No criterion is selected using test scores.\n\n'
    'Five-dataset macros first compute an equal-weight dataset average within each matching seed,then '
    'summarize the resulting five seed observations. Twenty-five dataset/seed records are not treated '
    'as25independent training seeds. Every original score stream is retained. Original per-dataset '
    'results use each source frozen full covered support; common-prefix comparisons are labeled separately.\n\n'
    'Twenty-one checks passed,including an independently checkeddf4quantile/standard error,missing/duplicate '
    'seed rejection,unclipped intervals,all controls and undefined-value handling,paired unequal-support '
    'fixtures,unchanged validation calibration,proper macro sample size,guard/proof prerequisites,actual '
    'cohort-worker metadata plumbing and changed-native-artifact rejection. Fixtures are calculation and '
    'provenance checks only,not full-data benchmark results. A fresh-process probe confirms metadata '
    'imports neitherNumPy/Pandas norTorch. Numerical libraries are isolated to ready short-lived children.\n\n'
    'All original training/data/model code and every numerical worker remain unchanged. Failed or '
    'incomplete attempts are preserved without automatic retry. The same sharedCPUslot,32GiBphysical/ '
    '40GiBcommit admission and16GiBemergency guard apply. Derived statistics stay onF:. This iteration '
    'adds no training or inference and certifies neither historical paper equivalence nor strictTAB.\n',
    encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=21, actual_live_controller=process.pid, actual_completed_cohort_statistics=0,
    artifact_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    new_benchmark_completions_claimed=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),CONFIG,
    'scripts/flow_matching/run_anomaly_transformer_full_cohort_statistics_v1.py',
    'scripts/flow_matching/register_anomaly_transformer_full_cohort_statistics_v1.py',
    'tests/test_fm_anomaly_transformer_full_cohort_statistics_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/AnomalyTransformer_full_cohort_statistics_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),
    paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=21,live_controller=process.pid,
    statistics_tasks=215,actual_complete_new_cohorts=0,new_training_or_inference_jobs=0)))
