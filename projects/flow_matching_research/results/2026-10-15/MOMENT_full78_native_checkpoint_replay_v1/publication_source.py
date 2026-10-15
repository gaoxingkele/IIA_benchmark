"""Publish live full78 MOMENT inference replay and explicit evidence limits."""
from datetime import datetime, timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as runner
from scripts.flow_matching.register_moment_full_checkpoint_replay_v1 import CONFIG

ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
runtime=read(ROOT/CONFIG);cases=runner.verify(runtime)
launch=read(ROOT/runtime['state_root']/'launch.json');process=psutil.Process(launch['pid'])
assert process.create_time()==launch['create_time'] and process.cmdline()==launch['command']
state=read(ROOT/runtime['state_root']/'status.json')
assert state['pid']==process.pid and state['create_time']==process.create_time()
assert state['required_jobs']==78 and state['verified_count']==0 and not state['failed_preserved']
assert state['active_pid'] is None and state['state'] in ['waiting_for_memory','waiting_for_heavy_lock']
xml=ROOT/'tmp/MOMENT_full78_checkpoint_replay_validation_20261015_v3.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
rows=[]
for identity,case in cases.items():
    path=ROOT/case['job']['output_directory']/'result.json'
    row=dict(id=identity,family=case['family'],dataset=case['job']['dataset'],window=case['job']['window'],
        model_config=case['job']['model_config'],pretrained_checkpoint_sha256=case['config']['parameters']['checkpoint_sha256'],
        seed=case['job']['seed'],source_result_path=str(path),source_result_exists=path.exists(),native_full_replay_completed=False)
    if path.exists():
        result=read(path);assert result['status']=='completed' and result['job']==case['job']
        assert result['training_seconds']==0 and result['training_losses']==[]
        row.update(source_result_sha256=sha(path),source_guard_history=runner.guard_history(runtime,case),
            source_environment=result['environment'],source_parameter_count=result['parameter_count'])
    rows.append(row)
complete=[r for r in rows if r['source_result_exists']]
assert len(complete)==35 and sum(r['family']=='original' for r in complete)==32
assert sum(r['source_guard_history']['original_execution_guard_history_proven'] for r in complete)==7
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_full78_native_checkpoint_replay_v1'
folder.mkdir(exist_ok=False)
for name,path in {
    'frozen_full78_replay_runtime.json':ROOT/CONFIG,
    'full_native_replay_and_independent_review_source.py':ROOT/'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
    'registration_source.py':ROOT/'scripts/flow_matching/register_moment_full_checkpoint_replay_v1.py',
    'validation_tests.py':ROOT/'tests/test_fm_moment_full_checkpoint_replay_v1.py',
    'validation.xml':xml,
    'original_resource_limits.json':ROOT/runner.RESOURCE,
    'original_pretrained_execution_contract.json':ROOT/'configs/experiments/fm_moment_pretrained_execution.v1.json'}.items():
    shutil.copyfile(path,folder/name)
write(folder/'actual_live_waiting_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    launch=launch,actual_live=True,private_bytes=process.memory_info().private,state_at_capture=state,
    configuration=CONFIG,configuration_sha256=sha(ROOT/CONFIG),
    completed_new_full_checkpoint_replays=0,new_benchmark_completions_claimed=0,
    numerical_workers_stopped_or_restarted=0,source_model_or_environment_changed=False))
write(folder/'all78_original_and_preprocessing_control_inventory.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_registered_jobs=72,additional_distinct_constant_fit_controls=6,total_unique_case_ids=78,
    existing_original_complete_result_files=32,existing_constant_fit_complete_result_files=3,
    total_existing_source_complete_result_files=35,source_guard_history_proven=7,source_guard_history_unproven=28,
    full_native_replay_complete_at_capture=0,saved_score_audits_are_native_replays=False,
    zero_shot_repeats_are_new_training_seeds=False,rows=rows,all_original_experiments_complete=False))
failed=ROOT/'tmp/MOMENT_full78_checkpoint_replay_validation_20261015_v2.xml'
write(folder/'prelaunch_Python310_hash_compatibility_fix.json',dict(
    discovered_during='two added full-body standard/industrial toy integration fixtures in the actual producer interpreter',
    preserved_local_failure_xml=str(failed),preserved_local_failure_xml_sha256=sha(failed),
    prelaunch_tests_failed=2,prelaunch_tests_passed=13,error="AttributeError: module 'hashlib' has no attribute 'file_digest'",
    cause='The new verifier initially reused a Python3.11-only hashing helper; the MOMENT producer uses Python3.10.',
    correction='Use an equivalent chunked SHA256 implementation scoped only to the new verifier.',
    frozen_source_algorithm_modified=False,producer_interpreter_upgraded=False,tolerance_relaxed=False,
    formal_replay_workers_launched_before_fix=0,final_tests_passed=15))
(folder/'README.md').write_text(
    'Full native checkpoint replay for72MOMENT jobs and6preprocessing controls\n\n'
    'This live queue closes the gap between saved-score audits and actual checkpoint reinference. '
    'It preserves all72original small/base/large xcontext512/100 x12dataset jobs,plus all6distinct '
    'SMAPconstant-fit controls. A current inventory finds32original and3control completed source '
    'result files(35total),not35native replay certificates. Older33/72snapshots describe narrower '
    'captures; preprocessing controls are kept separate and never substituted for original cases. '
    'Zero new full native replay completions are claimed at this capture.\n\n'
    'The first guarded child uses the exact original Python3.10/NumPy/Torch producer environment, '
    'loads the complete registered official release weights strictly through the unchanged model '
    'adapter and verifies the saved checkpoint reference/load receipt. It scores every original '
    'train,validation and test group/timestamp with batch1,original full context,padding and all-observed '
    'masks. An instance-local observer independently checks float64reconstruction MSE for every forward '
    'window at frozenrtol/atol2e-6. Scores,labels,timestamps,source rows and fault labels must match '
    'every saved array in exact dtype,shape and content bytes. Short groups and end-aligned overlapping '
    'tail windows remain fully included. Input groups are read separately to avoid redundant array '
    'retention; no source points,window operations or model outputs are reduced.\n\n'
    'A separate fresh producer-environment child rereads all native/source arrays and metadata, '
    'replays every strict/core metric,event metric and original200resample entity/time-block interval '
    'exactly using the original library implementation. Independent NumPy formulas additionally '
    'check validation-only thresholds,all entity/micro/macro confusion/ranking values,PA controls '
    'and every one of the ten test-pooled diagnostic ratios. Pooled/test-label-selected results remain '
    'diagnostics. VUS/Affiliation numerical integrals and raw preprocessing values are not independently '
    'reconstructed here. Historical paper checkpoint equivalence and the completeTABharness remain open.\n\n'
    'Seven completed sources have matching passed execution-guard receipts in the registered history '
    'locations;28have unproven guard history there. Missing history is explicitly preserved and cannot '
    'be retroactively certified by a fresh replay guard. Existing failed or mismatched receipts cause '
    'preserved failure,not automatic retry. Zero-shot deterministic replay is not another training seed.\n\n'
    'Fifteen checks passed:complete78case matrices,unchanged resource limits,rejection of partial/reduced '
    'bindings,standard/industrial group ordering,missing/failed historical guard handling,actual producer '
    'environment source padding and all-observed windows,tied and single-class rank controls,and '
    'standard/industrial end-to-end source/native/independent worker bodies using toy frozen checkpoint '
    'fixtures. These fixtures validate plumbing only. The integration checks caught a Python3.11-only '
    'hash helper in the new verifier; scoped streamingSHA256fixed Python3.10compatibility before any '
    'formal replay launch. Original model code,environment and numerical tolerances were unchanged.\n\n'
    'The actual collector is live and waiting at the original bounded CPU safeguards:16GiBphysical/ '
    '32GiBcommit admission,16GiBemergency,10GiBRSS and12GiBprivate caps,shared original CPU lane. '
    'Metadata stays numerically cold; ready native/review children use the actual producer interpreter '
    'with offlineHubsettings. All new large outputs stay onF:. No numerical worker was stopped or '
    'restarted,no raw data or downloaded materials were overwritten,and the original45/681/273minimum '
    'scope and all other unfinished methods remain.\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=15,
    actual_live_controller=process.pid,actual_completed_new_full_native_replays=0,
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False,new_benchmark_completions_claimed=0))
targets=[folder.relative_to(ROOT).as_posix(),CONFIG,
    'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
    'scripts/flow_matching/register_moment_full_checkpoint_replay_v1.py',
    'tests/test_fm_moment_full_checkpoint_replay_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_full78_checkpoint_replay_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),
    paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=15,live_controller=process.pid,
    original_jobs=72,preprocessing_controls=6,existing_source_results=35,full_native_replay_completions=0)))
