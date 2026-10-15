"""Archive tested, actually live all500 checkpoint-review automation precisely."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as runner
from scripts.flow_matching.register_anomaly_transformer_full_checkpoint_review_v1 import CONFIG

ROOT, read, sha, write = runner.ROOT, runner.read, runner.sha, runner.write
runtime = read(ROOT/CONFIG); cases, contract = runner.verify(runtime)
launch = read(ROOT/runtime['state_root']/'launch.json')
process = psutil.Process(launch['pid'])
assert process.create_time() == launch['create_time'] and process.cmdline() == launch['command']
state = read(ROOT/runtime['state_root']/'status.json')
assert state['pid'] == process.pid and state['create_time'] == process.create_time()
assert state['required_models'] == 500 and not state['all_original_experiments_complete']
assert not state['verified_model_ids'] and not state['failed_preserved']
assert state['state'] == 'waiting_actual_full_training_completion' and state['active_pid'] is None
xml = ROOT/'tmp/AnomalyTransformer_full500_checkpoint_review_validation_20261015_v2.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 27
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_full500_checkpoint_review_v1'
folder.mkdir(exist_ok=False)
sources = {
    'frozen_full500_runtime.json': ROOT/CONFIG,
    'automatic_full_checkpoint_and_review_source.py': ROOT/'scripts/flow_matching/run_anomaly_transformer_full_checkpoint_review_v1.py',
    'independent_formula_and_metric_source.py': ROOT/'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
    'registration_source.py': ROOT/'scripts/flow_matching/register_anomaly_transformer_full_checkpoint_review_v1.py',
    'validation_tests.py': ROOT/'tests/test_fm_anomaly_transformer_full_checkpoint_review_v1.py',
    'validation.xml': xml,
    'primary_paper_contract.json': ROOT/read(ROOT/runner.FAMILIES['main'][0])['contract'],
    'AUTHOR_CODE_LICENSE.txt': ROOT/read(ROOT/runner.FAMILIES['main'][0])['source_directory']/'LICENSE'}
for family, (path, _, _) in runner.FAMILIES.items():
    sources['frozen_'+family+'_training_registration.json'] = ROOT/path
for name, path in sources.items():
    shutil.copyfile(path, folder/name)
write(folder/'actual_live_launch_and_waiting_receipt.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), launch=launch, actual_live=True,
    controller_private_bytes=process.memory_info().private, state_at_capture=state,
    runtime_configuration=CONFIG, runtime_configuration_sha256=sha(ROOT/CONFIG),
    completed_new_full_replays=0, benchmark_completions_claimed=0,
    numerical_workers_stopped_or_restarted=0, original_numerical_budgets_changed=False))
write(folder/'verification_coverage_and_limits.json', dict(
    family_counts={k: sum(c['family'] == k for c in cases.values()) for k in runner.FAMILIES},
    unique_required_training_models=500, required_seeds=[1103,1104,1105,1106,1107],
    tests_passed=27, fixture_performance_is_benchmark_performance=False,
    native_replay=dict(complete_checkpoint=True, splits=['train','validation','test'],
        coverage='every timestamp covered by the frozen source window/tail policy',
        array_equality='exact dtype,shape,content bytes,including signed zero',
        inline_independent_float64_formula_tolerance=runtime['formula_tolerance'],
        attention_checks='every returned head/layer/query/key for the scoring formula',
        independent_full_network_implementation=False),
    separate_independent_array_review=dict(full_arrays=True, validation_only_percentile=True,
        strict_greater_comparison=True, exact_confusion_counts=True, segment_PA=True,
        tied_ranking_AUROC_AP=True, all_seven_predefined_ROC_operating_points=True,
        numeric_metric_absolute_tolerance=2e-12, fresh_model_inference_in_this_second_stage=False),
    independent_raw_preprocessing_value_reconstruction=False,
    independent_preprocessing_proof='source/cache hashes,Table13 row counts,scaler fit rows and split provenance only',
    original_training_guard_required=True, each_replay_stage_guard_required=True,
    no_automatic_retries=True, no_training_updates_during_review=True,
    original_author_source_and_restored_SWaT_runtime_tracks_in_this500=False,
    five_seed_cohort_uncertainty_implemented_by_this_iteration=False,
    paper_equivalence_certified=False, strict_TAB_result=False,
    all_original_experiments_complete=False, original_scope=runtime['original_scope']))
(folder/'README.md').write_text(
    'Automatic full checkpoint review for500 paper reconstruction models\n\n'
    'The live metadata collector covers25main,400single-factor,50direct-Max and25plain Transformer '
    'models across five original real datasets and all five frozen seeds. Source-runtime author '
    'models and restoredSWaT-loader models remain separate. No new benchmark model is claimed '
    'completed at this capture. Training queues,source files,raw data,budgets and guards remain unchanged.\n\n'
    'Only a completed original result with a passed training resource guard is admitted. The first '
    'guarded child checks full epoch/window/update/backward counts,early stopping,validation checkpoint '
    'selection,all data and artifact hashes. It strictly restores the whole checkpoint in a fresh '
    'interpreter with the exact frozen attention patch and architecture. Every covered train,validation '
    'and test timestamp is re-inferred; every score and component array must have identical dtype,shape '
    'and content bytes. Every batch also receives an independent NumPyfloat64 calculation of reconstruction, '
    'head-mean symmetricKL,layer means,softmax criteria or explicitly diagnostic plain-attention entropy '
    'at the frozen rtol/atol2e-6. No window,layer,head or score-stream subsampling occurs.\n\n'
    'A separate fresh child rereads and hashes the full saved/native arrays and truth labels,independently '
    'recalculates all validation-only thresholds,strict-greater predictions,noPA and segmentPA confusion '
    'counts,seven predefinedROC operating points,tie-awareAUROC andAP. Counts and thresholds must agree '
    'exactly; rate/ranking values use the frozen2e-12absolute tolerance. Plain fixed-delta controls remain '
    'diagnostic. The second child performs array/metric review; the first child performs model inference.\n\n'
    'Twenty-seven meaningful checks passed,including actual full-width checkpoint restoration and complete '
    'native/independent arrays on four small fixtures covering each source family,analytical tied scores, '
    'head averaging order,PA,validation calibration,exact signed-zero/dtype/shape rejection,complete '
    'early-stop/update accounting,cold metadata imports,and preserved failed/missing source completion. '
    'Fixtures validate plumbing only. The live collector is waiting for actual full-data training results, '
    'with zero model replays complete. Failed or interrupted attempts are preserved without automatic retry.\n\n'
    'Both fresh stages retain32GiBphysical/40GiBcommit admission and16GiBemergency protection and use '
    'the shared CPU resource slot only while doing review work. Idle monitoring imports noNumPy/Pandas/Torch. '
    'Checkpoints,derived arrays and replay outputs stay onF:. No numerical worker was stopped or restarted.\n\n'
    'Limits remain explicit:the full network is the pinned source plus disclosed reconstruction,not an '
    'independently written second network. Raw preprocessing values are not independently reconstructed; '
    'their source/cache hashes,split metadata and scaler fitting provenance are checked. No new five-seed '
    'uncertainty aggregate is produced before complete cohorts. Historical author equivalence,strictTAB '
    'certification,other paper experiments and the full45papers/681records/273data-task minimum remain open.\n',
    encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.txt -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=27, live_controller=process.pid, actual_completed_new_model_replays=0,
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), CONFIG,
    'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
    'scripts/flow_matching/run_anomaly_transformer_full_checkpoint_review_v1.py',
    'scripts/flow_matching/register_anomaly_transformer_full_checkpoint_review_v1.py',
    'tests/test_fm_anomaly_transformer_full_checkpoint_review_v1.py']
paths = set()
for arguments in [('diff','--name-only'), ('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *arguments, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest = dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'), paths=sorted(paths))
write(ROOT/'tmp/AnomalyTransformer_full500_checkpoint_review_publication_manifest_20261015_v1.json',manifest)
print(json.dumps(dict(exact_files=len(paths), tests_passed=27, live_controller=process.pid,
    required_models=500, actual_new_benchmark_completions=0)))
