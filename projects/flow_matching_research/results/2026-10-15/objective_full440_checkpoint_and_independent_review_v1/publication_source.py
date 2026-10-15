"""Archive exact live full440 objective review,including negative precision evidence."""
from datetime import datetime,timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_objective_full440_checkpoint_replay_v1 as runner
from scripts.flow_matching.register_objective_full440_checkpoint_replay_v1 import CONFIG

ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
runtime=read(ROOT/CONFIG);cases=runner.verify(runtime)
launch=read(ROOT/runtime['state_root']/'launch.json');process=psutil.Process(launch['pid'])
assert process.create_time()==launch['create_time'] and process.cmdline()==launch['command']
state=read(ROOT/runtime['state_root']/'status.json')
assert state['pid']==process.pid and state['create_time']==process.create_time()
assert state['required_cases']==440 and state['verified_count']==0 and not state['failed_verifications']
assert state['source_failure_count']==105 and state['active_pid'] is None
assert state['state'] in ['waiting_for_memory','waiting_for_heavy_lock']
xml=ROOT/'tmp/Objective_full440_checkpoint_replay_validation_20261015_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==21
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
rows=[]
for identity,case in cases.items():
    path=ROOT/case['job']['output_directory']/'result.json'
    row=dict(id=identity,family=case['family'],dataset=case['job']['dataset'],seed=case['job']['seed'],
        model_config=case['job']['model_config'],source_result_path=str(path),source_result_exists=path.exists(),
        original_terminal_failure_preserved=identity in runtime['preserved_original_failures'],
        preexisting_full_native_certificate=identity==runtime['existing_native_certificate']['id'])
    if path.exists():
        result=read(path);assert result['status']=='completed' and result['job']==case['job']
        assert len(result['training_losses'])==20
        row.update(source_result_sha256=sha(path),source_parameter_count=result['parameter_count'],source_environment=result['environment'])
    rows.append(row)
assert sum(r['source_result_exists'] for r in rows)==237
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/objective_full440_checkpoint_and_independent_review_v1'
folder.mkdir(exist_ok=False)
for name,path in {
    'frozen_full440_runtime.json':ROOT/CONFIG,
    'native_and_independent_queue_source.py':ROOT/'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
    'independent_probability_path_and_MC_source.py':ROOT/'scripts/flow_matching/objective_replay_math_v1.py',
    'registration_source.py':ROOT/'scripts/flow_matching/register_objective_full440_checkpoint_replay_v1.py',
    'validation_tests.py':ROOT/'tests/test_fm_objective_full440_checkpoint_replay_v1.py',
    'validation.xml':xml,
    'original_resource_limits.json':ROOT/runner.RESOURCE,
    'original_terminal_280_case_history.json':ROOT/runner.LEGACY_TERMINAL,
    'reused_existing_full_SB_native_configuration.json':ROOT/runner.OLD_SB}.items():
    shutil.copyfile(path,folder/name)
write(folder/'actual_live_waiting_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    launch=launch,actual_live=True,private_bytes=process.memory_info().private,state_at_capture=state,
    configuration=CONFIG,configuration_sha256=sha(ROOT/CONFIG),new_verified_full_models=0,
    existing_native_certificate_to_reuse=runtime['existing_native_certificate'],
    numerical_workers_stopped_or_restarted=0,new_benchmark_completions_claimed=0))
write(folder/'all440_case_source_result_and_negative_outcome_inventory.json',dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),required_cases=440,
    original_cases=280,distinct_numerical_repair_cases=105,preprocessing_controls=40,preprocessing_repair_controls=15,
    source_completed_result_files=237,source_completed_by_family={k:sum(r['family']==k and r['source_result_exists'] for r in rows) for k in runner.QUEUES},
    original_terminal_failures_preserved=105,preexisting_native_model_certificate=1,new_native_reviews_complete=0,
    source_file_presence_is_benchmark_certification=False,rows=rows,all_original_experiments_complete=False))
failure=ROOT/'tmp/Objective_full440_checkpoint_replay_validation_20261015_v1.xml'
write(folder/'preserved_float64_precision_audit_limitation.json',dict(
    detected_in='full20epoch Target-CFM source/native toy integration fixture; no formal benchmark replay had launched',
    original_validation_failures=1,original_validation_passes=20,preserved_raw_failure_xml=str(failure),preserved_raw_failure_xml_sha256=sha(failure),
    original_failure='Independent full-coordinate probability path mismatch: velocity',
    interpretation='The preserved target velocity formula uses FP32 subtraction and division; some coordinates differ from its ideal float64 evaluation beyond the fixed2e-6tolerance.',
    unchanged_tolerance=dict(rtol=2e-6,atol=2e-6),original_source_algorithm_modified=False,
    ideal_float64_comparison_removed=False,ideal_float64_failure_counts_or_errors_suppressed=False,
    source_FP32_program_check='independent NumPy FP32 evaluation with source operation order at unchanged2e-6tolerance',
    native_array_check='exact dtype/shape/content bytes,not an approximate match',
    every_ideal_float64_coordinate_still_audited=True,
    required_runtime_evidence=['float64_path_failure_coordinate_counts','maximum_path_absolute_errors','ideal_float64_path_allclose_passed'],
    certification_rule='Any failed ideal coordinate leaves ideal_float64_path_equivalence_certified=false in both native and independent reports.',
    final_target_fixture_explicitly_asserts_failed_ideal_equivalence=True,final_tests_passed=21,
    benchmark_performance_claim=False))
(folder/'README.md').write_text(
    'Full440local FM objective checkpoint and independent review queue\n\n'
    'The actually live queue covers280original objective-adapter jobs,105separate numerical repairs, '
    '40constant-fit controls and15constant-fit numerical repairs. The source inventory contains237 '
    'completed result files:175original,60repaired and2preprocessing controls. This is source-file '
    'presence,not237certified native models. All105terminal failed original attempts remain negative '
    'outcomes with their original history; they are not retried,overwritten or pooled with repaired runs. '
    'The existing fullSB_SWaTseed1103 native certificate is reused and receives a separate new independent '
    'metric review,without duplicate model inference. Zero new full-model review completions are '
    'claimed at this capture. Other methods and the full45/681/273minimumscope remain open.\n\n'
    'A ready guarded native child restores the entire trained three-linear-layer SiLUMLP,including '
    'both velocity and score heads when configured,and checks every checkpoint tensor bitwise. It '
    'does no fitting or optimization. The exact original environment,window100,hidden128,batch64, '
    'seed+1score RNG reset,fourMCsamples,five scoring times,all entity/split calls,full end-aligned '
    'tail windows and4CPUthreads remain. Every saved score and label array must match dtype,shape '
    'and content bytes; all source timestamps/rows/fault-label metadata are bound to input hashes.\n\n'
    'Every fresh native path coordinate is independently evaluated with NumPy in the originalFP32 '
    'operation order. Every one of the20MC/time contributions is independently reduced in float64 '
    'and checked at fixedrtol/atol2e-6. Every coordinate also receives an ideal float64path comparison '
    'at that unchanged tolerance,with all failure coordinate counts and maximum errors retained. '
    'The full Target-CFM fixture exposed FP32subtraction-cancellation differences from the ideal '
    'float64formula. These are preserved numerical audit outcomes:they are not hidden or reclassified '
    'as high-precision equivalence. Any failed ideal coordinate leaves the explicit ideal-equivalence '
    'certification false. Original source code,weights,scores and tolerances are unchanged. The '
    'native source-array comparison remains exact bytes regardless of this extra diagnostic.\n\n'
    'A separate child rereads all native/source arrays and labels,replays the complete original '
    'strict/core/event metrics and200resample grouped bootstrap intervals exactly,and independently '
    'checks all validation-only thresholds,entity/micro/macro confusion/ranking metrics,PA controls '
    'and ten test-pooled diagnostic ratios. Bootstrap replay uses the original implementation. '
    'Raw preprocessing values and VUS/Affiliation integrals are not independently recomputed. The '
    'reused SBcertificate is not falsely credited with the newly added inline path audit. Missing '
    'historical source execution guards remain unproven; fresh guards cannot retroactively certify them.\n\n'
    'Twenty-one checks passed:all440case identities,all105original failures,rejected narrowed budgets/ '
    'guards/tolerances,independent equations for seven objectives,existing native certificate reuse, '
    'source terminal-receipt handling,cold metadata,and real20epoch/full128hidden toy source/native/ '
    'independent integration for independent,target and both SF2Mhead settings. Every toy integration '
    'checks all120path/forward calls,16full windows and1600window points. The target fixture explicitly '
    'requires retained negative float64precision evidence. Fixtures validate plumbing and numerical '
    'audit semantics only; they are not benchmark results.\n\n'
    'The exact live metadata controller is waiting at original16GiBphysical/32GiBcommit admission, '
    '16GiBemergency and10GiBRSS/12GiBprivate caps in the same boundedCPUlane. Large replay outputs '
    'stay onF:. No existing numerical process was stopped,restarted or hot-patched. Failed/interrupted '
    'attempts are preserved without automatic retry. These are explicitly local velocity-residualTSAD '
    'adapters,not likelihoods or original image/single-cell experiments,nor historical author or '
    'fullTABharness reproductions. Test-selected pooled/PA controls remain diagnostic.\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=21,
    actual_live_controller=process.pid,new_verified_native_models=0,existing_native_certificate_reused=1,
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False,new_benchmark_completions_claimed=0))
targets=[folder.relative_to(ROOT).as_posix(),CONFIG,
    'scripts/flow_matching/objective_replay_math_v1.py',
    'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
    'scripts/flow_matching/register_objective_full440_checkpoint_replay_v1.py',
    'tests/test_fm_objective_full440_checkpoint_replay_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/Objective_full440_checkpoint_replay_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=21,live_controller=process.pid,required_cases=440,
    source_result_files=237,original_failure_outcomes_preserved=105,new_verified_native_models=0)))
