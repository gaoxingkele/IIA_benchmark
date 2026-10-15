"""Archive the actual complete Base/MSL replay and a matched Small comparison."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as v

ROOT, read, write, sha = v.ROOT, v.read, v.write, v.sha
runtime_path = 'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
runtime = read(ROOT/runtime_path)
cases = v.verify(runtime)
state = read(ROOT/runtime['state_root']/'status.json')
name = 'fm_moment_pretrained_base__release512__msl__seed42'
small = 'fm_moment_pretrained_small__release512__msl__seed42'
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_base512_MSL42_full_native_closed_v1'
assert not folder.exists()
files, records, summaries = {}, [], []
for identity, parameters in [(small, 35341512), (name, 109641608)]:
    case = cases[identity]
    source, data, binding = v.check_source(runtime, case)
    directory = Path(runtime['artifact_root'])/identity
    native_path, review_path = directory/'native/native_replay.json', directory/'independent/review.json'
    native, review = read(native_path), read(review_path)
    assert identity in state['verified_ids']
    assert native['passed'] and review['passed']
    assert native['source_binding'] == review['source_binding'] == binding
    assert native['job'] == review['job'] == case['job']
    assert native['runtime_sha256'] == sha(ROOT/runtime_path)
    assert review['native_replay_sha256'] == sha(native_path)
    assert native['all_original_groups_reinferred'] and native['all_array_dtype_shape_content_identical']
    assert review['all_original_arrays_independently_checked']
    assert review['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert native['checkpoint_load_audit']['parameter_count'] == parameters
    assert native['checkpoint_load_audit']['all_keys_strictly_loaded']
    assert native['training_updates_performed'] == 0
    assert native['native_coverage'] == source['native_coverage']
    assert {role: sum(r['native_points'] for r in native['native_coverage'] if r['role'] == role) for role in ['train','validation','test']} == dict(train=43692, validation=11586, test=71418)
    assert len(native['native_coverage']) == 78 and len(native['arrays']) == 104
    assert native['inline_full_window_MSE_check']['forward_calls'] == 285
    assert sha(native['scores_path']) == native['scores_sha256'] == binding['scores_sha256']
    selected = dict(original_result=ROOT/case['job']['output_directory']/'result.json', native_replay=native_path,
                    independent_review=review_path, model_configuration=ROOT/case['job']['model_config'])
    for phase, proof in [('native', native_path), ('independent', review_path)]:
        base = ROOT/runtime['state_root']/'attempts'/identity/phase
        guard = read(base/'guard_receipt.json')
        assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
        assert guard['proof_sha256'] == sha(proof)
        assert guard['exact_child_handle']['command'] == [str(ROOT/case['queue']['python']), '-B', '-X', 'utf8', '-u', '-m', v.MODULE,
            '--config', runtime_path, '--stage', phase, '--job-id', identity]
        selected[phase+'_guard'] = base/'guard_receipt.json'
        selected[phase+'_launch'] = base/'launch.json'
    size = 'Base' if identity == name else 'Small'
    files[size] = selected
    controls = review['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio, original in source['metrics']['strict'].items():
        c = controls[ratio]
        assert c['micro']['f1'] == original['micro']['f1']
        assert c['threshold'] == original['calibration']['threshold']
        ci, events = original['f1_ci95'], original['event_metrics']
        records.append(dict(model='MOMENT-1-'+size.lower(), parameters=parameters, dataset='MSL', context=512, pretrained_seed=42,
            validation_quantile_exceedance_percent=float(ratio), threshold=c['threshold'], no_PA_F1=c['micro']['f1'],
            same_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'], AUROC=c['micro']['auroc'],
            AP=c['micro']['average_precision'], precision=c['micro']['precision'], recall=c['micro']['recall'],
            F1_CI95_low=ci['low'], F1_CI95_high=ci['high'], bootstrap_unit=ci['unit'], bootstrap_resamples=ci['resamples'],
            bootstrap_seed=ci['seed'], bootstrap_block_points=ci['block_points'], event_recall=events['event_recall'],
            events=events['events'], detected_events=events['detected_events'],
            false_alarms_per1000_normal=events['false_alarms_per_1000_normal_points']))
    import numpy as np
    with np.load(native['scores_path'], allow_pickle=False) as saved:
        actual_arrays = []
        for key in saved.files:
            if any(key.endswith('_'+role) for role in ['train','validation','test']):
                values = saved[key]
                assert values.ndim == 1 and np.isfinite(values).all()
                actual_arrays.append(values)
        full = np.concatenate(actual_arrays)
        assert len(full) == 43692+11586+71418
        summaries_profile = dict(points=len(full), minimum=float(full.min()), maximum=float(full.max()),
            absolute_maximum=float(np.abs(full).max()), scores_over1e6=int((np.abs(full)>1e6).sum()),
            scores_over1e12=int((np.abs(full)>1e12).sum()), all_finite=True,
            independent_float64_window_MSE_check=native['inline_full_window_MSE_check'])
    summaries.append(dict(job_id=identity, parameters=parameters, source_binding=binding,
                          role_points=dict(train=43692, validation=11586, test=71418), forward_calls=285,
                          checkpoint_load_audit=native['checkpoint_load_audit'], full_saved_score_magnitude_profile=summaries_profile, new_training_seed=False))
assert summaries[0]['source_binding']['input_sha256'] == summaries[1]['source_binding']['input_sha256']
validation = ROOT/'tmp/MOMENT_base512_PSM42_native_closure_validation_20261015_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
checks = sum(int(s.get('tests', 0)) for s in suites)
assert checks >= 15 and all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
folder.mkdir(parents=True, exist_ok=False)
for size, selected in files.items():
    target = folder/size.lower()
    target.mkdir()
    for label, path in selected.items():
        shutil.copyfile(path, target/(label+'.json'))
        assert sha(path) == sha(target/(label+'.json'))
shutil.copyfile(ROOT/runtime_path, folder/'frozen_full78_replay_configuration.json')
shutil.copyfile(validation, folder/'validation.xml')
write(folder/'collector_status_snapshot.json', state)
write(folder/'metrics_and_matched_comparison.json', dict(records=records, full_native_models=summaries,
    additional_closed_cases=1, reused_small_case=small, new_training_seeds=0,
    all_validation_quantiles_preserved=True, bootstrap_is_model_seed_uncertainty=False,
    full_TAB_harness_certified=False, historical_paper_equivalence_certified=False,
    independent_lowlevel_VUS_affiliation_integrals=False, independent_full_preprocessing_reconstruction=False,
    original_missing_guard_history_retroactively_certified=False, all_original_experiments_complete=False))
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator='\n')
writer.writeheader(); writer.writerows(records)
(folder/'all_validation_quantile_metrics.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
primary = [r for r in records if r['validation_quantile_exceedance_percent'] == 1.]
lines = ['# MOMENT Base / MSL / context512: full native closure and matched Small comparison', '',
    'New complete109,641,608-parameter Base replay;Small is the previously closed35,341,512-parameter control and is not counted twice. Both use the identical input hash,all26registered entities and78TRAIN/validation/TEST roles:43692/11586/71418 native points. Each performs285full-window forwards. All104source score/label/timestamp arrays have identical dtype,shape and content bytes. Both actual guarded stages exited successfully without resource abort;the second independent process checks every array,core/event metric,original200resample interval and all10pooled diagnostic ratios.', '',
    '|Model|Frozen validation1% no-PA F1|Same-threshold PA F1|AUROC|AP|', '|---|---:|---:|---:|---:|']
for r in primary:
    lines.append(f"|{r['model']}|{r['no_PA_F1']:.8f}|{r['same_threshold_PA_F1']:.8f}|{r['AUROC']:.8f}|{r['AP']:.8f}|")
lines += ['',
    'All three prespecified validation quantiles0.1/1/5are retained;the primary1%is not chosen by test F1. No test-derived score inversion when AUROC is below0.5. Base and Small differ in model size,on the same fixed preprocessing;this single frozen-checkpoint comparison does not establish significance or broad model-scale effects. The grouped bootstrap intervals remain conditional on model and threshold,not training-seed uncertainty.', '',
    'The original full-window float64-MSE cross-check retains frozen relative/absolute tolerance2e-6. Its maximum absolute difference is0.8076958516612649for Base and0.9373569972813129for Small,reflecting the numerical scale of this unchanged source track;do not interpret the accepted relative check as near-zero absolute error. Full saved-score magnitude profiles are archived separately. No clipping,rescaling,preprocessing fix or tolerance change is made here. The distinct constant-fit normalization controls remain separately registered and are not certified by this packet.', '',
    'Full26-entity grouped source splits are retained;the registered MSL subset and preprocessing have not been certified identical to the historical paper;source historical guard gaps,all preprocessing-value reconstruction,VUS/Affiliation low-level integrals,the complete TAB executor and historical-paper equivalence remain unproven. Windows include future context,so event delay is not online causal delay. These official release zero-shot tasks are separate from Tiny/ICM full-PILE training,classification and forecasting.', '',
    f'Reuse of{checks}unchanged frozen validation checks is backed by rehashing source/config/data/weights/proofs and both exact successful guard commands. One additional complete checkpoint/dataset case is closed,no new training seed;queue snapshot{state["verified_count"]}/78. Other cases and at least45papers/681method records/273data tasks plus added experiments remain required. The full research objective is not complete.', '',
    'Sources:[MOMENT paper](https://arxiv.org/abs/2402.03885),[official Base weights](https://huggingface.co/AutonLab/MOMENT-1-base),[NASA source](https://github.com/khundman/telemanom). Large weights and arrays stay at their original registered disk paths.']
(folder/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), new_complete_cases=1,
    reused_previous_complete_cases=1, tests_passed=checks, files_sha256={p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_base512_MSL42_full_native_closed.2026-10-15_v1.json'
write(ara, dict(method='MOMENT-1-base', dataset='MSL', context=512, official_pretrained_seed=42,
    evidence=folder.relative_to(ROOT).as_posix(), full_checkpoint_replay_and_independent_review=True, additional_closed_cases=1,
    original_scope_required_replay_cases=78, new_training_seeds=0, all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix()]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_base512_MSL42_publication_manifest_20261015_v1.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(new_closed_case=name, collector_closed=state['verified_count'], exact_files=len(paths), primary=primary)))
