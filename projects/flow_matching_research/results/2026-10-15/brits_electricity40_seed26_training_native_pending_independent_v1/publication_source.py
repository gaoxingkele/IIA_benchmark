"""Archive actual full training/native inference without certifying pending review."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import psutil

ROOT=Path(__file__).resolve().parents[1]
IDENTITY='saits_paper_brits_electricity_author_missing0.4_seed26'
BASE=ROOT/'projects/flow_matching_research/results/2026-10-15/brits_electricity40_seed26_training_native_pending_independent_v1'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(65536),b''):digest.update(chunk)
 return digest.hexdigest()

collector_path=ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json'
collector=read(collector_path)
owner=psutil.Process(collector['pid'])
assert owner.create_time()==collector['create_time']
assert collector['required_original_models']==210 and len(collector['verified_original_ids'])==44
assert IDENTITY not in collector['verified_original_ids']
native_path=ROOT/'configs/reproducibility/fm_saits_completion_replay_queue_v1'/(IDENTITY+'.native.json')
review_path=native_path.with_name(IDENTITY+'.review.json')
native,review=read(native_path),read(review_path)
for config in [native,review]:
 for pin in config['source_receipts']:assert sha(ROOT/pin['path'])==pin['sha256'],pin['path']
native_report_path=ROOT/native['artifact_directory']/'checkpoint_replay_audit.json'
native_guard_path=ROOT/native['state_root']/'receipt.json'
report,guard=read(native_report_path),read(native_guard_path)
assert report['configuration_sha256']==sha(native_path)
assert report['full_models_replayed']==1 and report['full_checkpoint_contents_and_re_inference_passed']
assert report['metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
assert report['additional_training_seed_slots']==0
assert guard['exit_code']==0 and guard['full_native_model_replay_passed'] and not guard['resource_guard_aborted']
record=report['records'][0]
assert record['id']==IDENTITY and record['seed']==26 and record['model_parameters']==6993328
assert record['full_test_shape']==[290,100,370] and record['held_out_coordinates']==3537183
assert record['full_original_checkpoint_and_data_identity_verified'] and record['full_test_grid_and_grouped_statistics_replayed']
training=record['full_training_completion']
assert training['full_training_finished'] and not training['benchmark_smoke']
assert training['original_maximum_epochs']==10000 and training['author_patience_remaining']==0
result_path=ROOT/record['result_path'];result=read(result_path)
assert result['status']=='completed' and result['parameters']==6993328
assert tuple(result['config'][k] for k in ['method','dataset','seed','missing_ratio'])==('brits','electricity',26,.4)
assert result['target_count']==3537183 and result['test_windows']==290 and result['bootstrap_groups']==10
dataset=ROOT/result['config']['dataset_root'];output=ROOT/result['config']['output_root']
artifacts=dict(config=ROOT/record['config'],result=result_path,checkpoint=ROOT/record['checkpoint_path'],
 arrays=ROOT/record['replay_arrays'],dataset=dataset/'datasets.h5',resume=output/'resume.pt',
 original_group_statistics=output/'group_statistics.npz',data_audit=dataset/'data_audit.json')
hashes={k:sha(p) for k,p in artifacts.items()}
assert hashes['config']==record['config_sha256'] and hashes['result']==record['result_sha256']
assert hashes['checkpoint']==record['checkpoint_sha256'] and hashes['arrays']==record['replay_arrays_sha256']
assert hashes['dataset']==record['original_full_dataset_sha256']==result['dataset_sha256']
for metric in ['mae','rmse','mre']:
 assert result['metrics'][metric]==record['original_metrics'][metric]
 assert math.isclose(record['replayed_metrics'][metric],result['metrics'][metric],rel_tol=2e-6,abs_tol=2e-6)
source_guard_path=Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts')/(IDENTITY+'.json')
source_guard=read(source_guard_path)
assert source_guard['exit_code']==0 and source_guard['full_result_validated'] and not source_guard['resource_guard_aborted']
review_proof=ROOT/review['output_directory']/'independent_full_replay_batch_verification.json'
review_guard=ROOT/review['state_root']/'receipt.json'
assert not review_proof.exists() and not review_guard.exists(),'Full review may now be ready: inspect it and use full closure instead'
review_state_path=ROOT/review['state_root']/'status.json';review_state=read(review_state_path)
pending=psutil.Process(34564)
assert pending.create_time()==1792026286.63036
assert review_state['state']=='waiting_for_memory' and review_state.get('active_pid') is None
assert review['settings']['minimum_commit_headroom_bytes']==28*1024**3
tests=ROOT/'tmp/BRITS_electricity40_seed26_full45_validation_20261015_v1.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==40
assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),id=IDENTITY,method='brits',dataset='electricity',
 nominal_missing_ratio=.4,actual_held_out_target_fraction=3537183/10730000,
 full_test_shape=[290,100,370],full_test_cells=10730000,held_out_coordinates=3537183,month_groups=10,
 model_parameters=6993328,original_maximum_epochs=10000,original_early_stop_patience=30,
 actual_training_completion=training,actual_validation_best=result['validation_best'],
 original_metrics=result['metrics'],full_native_replayed_metrics=record['replayed_metrics'],
 original_training_guard_passed=True,full_checkpoint_reinference_passed=True,
 separate_independent_full_grid_review_passed=False,complete_original_model_review_claimed=False,
 collector_verified_original_models=44,required_original_models=210,
 required_original_training_seeds=[26,27,28,29,30],complete_five_training_seed_aggregate=False,
 between_training_seed_mean_and_ci=None,within_model_month_block_bootstrap=result['month_block_bootstrap_95_ci'],
 source_prediction_bitwise_equivalence_claimed=False,metric_tolerance=report['metric_tolerance'],
 full_artifact_sha256=hashes,independent_guardian=dict(pid=pending.pid,create_time=pending.create_time(),
  exact_live=True,actual_state=review_state,settings=review['settings']),
 original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
 strict_TAB_result=False,author_equivalence_certified=False,full_goal_complete=False,
 additional_training_updates_in_this_archive=0,additional_model_reinference_in_this_archive=0,
 protocol_boundaries=result['protocol_boundaries'],citation=result['config']['citation'],data_citation=result['config']['data_citation'])
sources=dict(original_result=result_path,original_training_guard=source_guard_path,
 native_configuration=native_path,independent_configuration=review_path,
 full_native_report=native_report_path,native_guard=native_guard_path,
 actual_pending_independent_state=review_state_path,actual_collector_status=collector_path)
summary['source_receipts']=[dict(name=k,path=str(p),sha256=sha(p)) for k,p in sources.items()]
BASE.mkdir(parents=True,exist_ok=False)
for name,path in sources.items():shutil.copyfile(path,BASE/(name+'.json'))
(BASE/'complete_training_native_passed_independent_pending.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
shutil.copyfile(tests,BASE/'validation.xml')
shutil.copyfile(Path(__file__),BASE/'publication_source.py')
(BASE/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(BASE/'README.md').write_text('''# BRITS Electricity40 seed26: full training and checkpoint replay

The original 6,993,328-parameter model completed the registered maximum10,000 /
patience30 policy. The validation-selected checkpoint passed full native inference
over all290x100x370 test coordinates,3,537,183 targets and10months. Original source,
dataset, checkpoint, saved full-grid arrays and guard hashes were checked.

|Metric|Original full-test result|
|---|---:|
|MAE|1.009792447090149|
|RMSE|1.503408432006836|
|MRE|0.5407029390335083|

The separate independent full-grid review is still waiting for its unchanged28GiB
commit-headroom admission. It has no successful terminal receipt yet. Consequently
this snapshot does not add a fully reviewed model: the collector remains44/210.
All40 relevant implementation/guard/integrity checks passed. This is neither a
five-model aggregate nor an author-equivalence/TAB certification. Nominal40%
replacement masking yields32.9653588% actual held-out coordinates. Original
within-model month-block intervals are retained separately; source prediction
arrays were not retained, so bitwise original prediction equivalence is unproved.

The next original seed27 training is already running. Original seeds26..30,
all210 SAITS/BRITS slots and the full45-paper/681-method/273-data-task scope remain.
No numerical training or checkpoint replay was added by this archival step.
''',encoding='utf-8')
targets=[BASE.relative_to(ROOT).as_posix(),native_path.relative_to(ROOT).as_posix(),review_path.relative_to(ROOT).as_posix()]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
 raw=subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True)
 paths.update(p for p in raw.splitlines() if not p.endswith('.log'))
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
(ROOT/'tmp/BRITS40_seed26_native_pending_review_manifest_20261015_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(full_original_training_completed=True,full_native_checkpoint_reinference_passed=True,
 separate_independent_review_pending=True,collector_fully_reviewed=44,required_original_models=210,
 original_metrics=result['metrics'],validation_checks_passed=40,exact_publication_files=len(paths))))
