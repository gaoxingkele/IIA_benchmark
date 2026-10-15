from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_idle_review_headroom_cached_bundle_v5 as adapter
base=adapter.original
CONFIG='configs/runtime/fm_idle_review_headroom_cached_bundle.2026-10-15_v5.json'
META=ROOT/'projects/flow_matching_research/results/2026-10-15/idle_review_headroom_cached_bundle_v5'
MODEL=ROOT/'projects/flow_matching_research/results/2026-10-15/brits_electricity40_seed26_closed_full45_model_increment_v1'
IDENTITY='saits_paper_brits_electricity_author_missing0.4_seed26'
config=base.read(ROOT/CONFIG)
registration=base.read(META/'registration_and_actual_guard_api_preflight.json')
launch=base.read(META/'actual_launch.json')
assert registration['configuration_sha256']==launch['configuration_sha256']==base.digest(ROOT/CONFIG)
tests=[(ROOT/'tmp/idle_review_headroom_cached_bundle_v5_tests_20261015.xml',30),
       (ROOT/'tmp/BRITS_electricity40_seed26_full45_validation_20261015_v1.xml',40)]
for path,count in tests:
 suites=list(ET.parse(path).getroot().iter('testsuite'))
 assert sum(int(s.get('tests','0')) for s in suites)==count
 assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
old_total=new_total=0;current=[]
for row in launch['actual_successors']:
 name=row['role'];role=adapter.verify(config,name)
 process=psutil.Process(row['pid']);assert base.exact_live(process,row)
 state=base.read(ROOT/role['previous_state_path'])
 assert state['pid']==row['pid'] and adapter.idle_allowed(state,
  dict(pid=process.pid,create_time=process.create_time()),base.descendants_of(process))
 previous=role['previous_owner']
 try:assert not base.exact_live(psutil.Process(previous['pid']),previous)
 except psutil.NoSuchProcess:pass
 source=ROOT/config['handoff_state_root']/name
 for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','successor_entry.json']:
  value=base.read(source/file)
  target=META/'actual_handoffs'/name/file;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(source/file,target)
 assert not base.read(source/'handoff.json')['original_native_child_stopped']
 old=registration['previous_live_idle_owners'][name]['actual_private_bytes'];new=process.memory_info().private
 old_total+=old;new_total+=new
 current.append(dict(role=name,pid=process.pid,create_time=process.create_time(),command=process.cmdline(),
  exact_live=True,old_private_bytes=old,current_private_bytes=new,actual_state=state,
  original_registry_jobs=role['required_original_slots'],scientific_settings_unchanged=True))
numeric=[]
for expected in launch['original_numerical_workers_before'].values():
 process=psutil.Process(expected['pid']);assert base.exact_live(process,expected)
 numeric.append(dict(pid=process.pid,create_time=process.create_time(),exact_live=True,
  cpu_seconds=sum(process.cpu_times()[:2]),command_unchanged=True))
model=base.read(MODEL/'one_complete_original_model_and_full_grid_review.json')
assert model['method']=='brits' and model['model_parameters']==6993328
assert model['fully_verified_original_seeds']==[26] and model['required_original_training_seeds']==[26,27,28,29,30]
assert model['one_original_training_full_native_model_and_independent_review_passed']
assert model['reviewed_SAITS_BRITS_models']==45 and model['required_SAITS_BRITS_models']==210
assert not model['complete_five_seed_aggregate'] and not model['all_original_experiments_complete']
assert model['full_test_shape']==[290,100,370] and model['held_out_coordinates_per_model']==3537183
record=model['per_model'][0];assert record['id']==IDENTITY
guard=base.read(MODEL/IDENTITY/'independent_guard.json')
assert guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_native_batch_independent_review_passed']
collector=base.read(ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json')
owner=psutil.Process(collector['pid']);assert owner.create_time()==collector['create_time']
assert len(collector['verified_original_ids'])==45 and IDENTITY in collector['verified_original_ids']
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
closure=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=base.SCOPE,
 actual_current_full_registry_successors=current,previous_metadata_private_bytes=old_total,
 current_metadata_private_bytes=new_total,measured_metadata_private_bytes_released=old_total-new_total,
 original_four_numerical_workers=numeric,original_numerical_children_stopped=False,
 scientific_settings_or_resource_thresholds_changed=False,actual_resource_snapshot=resource_snapshot(),
 post_handoff_observed_completed_independent_model=IDENTITY,SAITS_BRITS_fully_reviewed_models=45,
 SAITS_BRITS_required_original_models=210,independent_full_grid_proof=record['current_proof_hashes'],
 model_closure_path=(MODEL/'one_complete_original_model_and_full_grid_review.json').relative_to(ROOT).as_posix(),
 model_closure_sha256=base.digest(MODEL/'one_complete_original_model_and_full_grid_review.json'),
 validation=dict(metadata_handoff_checks=30,native_and_integrity_checks=40,total_passed=70),
 new_numerical_seeds_added=0,author_equivalence_certified=False,full_goal_complete=False,
 boundary='Actual lower metadata private memory and a completed independent review observed after the handoff. This is not a causal experiment about global memory availability; unchanged original resource guards decide admission. No trained model was stopped, no budget or guard was reduced, and no five-seed aggregate or whole-research completion is claimed.')
base.write(META/'actual_full_registry_handoff_and_completed_review_closure.json',closure)
shutil.copyfile(ROOT/'tmp/launch_idle_review_headroom_bundle_20261015_v5.py',META/'launcher_source.py')
shutil.copyfile(Path(__file__),META/'closure_and_publication_source.py')
shutil.copyfile(tests[1][0],MODEL/'validation.xml')
shutil.copyfile(Path(__file__),MODEL/'publication_source.py')
with (META/'README.md').open('a',encoding='utf-8') as f:
 f.write('\n## Actual transition and completed independent review\n\nAll four exact idle host transitions passed. The original four numerical training\nprocesses kept their OS identity and command. Every source/registry/configuration\npin was rechecked, and the actual new owners wrote the same original state paths.\n')
 f.write(f'\nMeasured idle-host private memory changed from {old_total/1048576:.3f} MiB to\n{new_total/1048576:.3f} MiB at these observations. Global headroom remains variable.\n')
 f.write('\nThe BRITS Electricity40 original seed26 separate full-grid review was observed\ncomplete after these handoffs: original training, full native checkpoint inference,\nand independent guarded review all passed. The SAITS/BRITS collector now records\n45/210 complete original models. This cohort remains1/5 and the entire research\nobjective remains incomplete. Previous pending snapshots are retained unchanged.\n')
auto=ROOT/'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1'/IDENTITY
targets=[META.relative_to(ROOT).as_posix(),MODEL.relative_to(ROOT).as_posix(),auto.relative_to(ROOT).as_posix(),
 CONFIG,'scripts/flow_matching/run_idle_review_headroom_cached_bundle_v5.py',
 'tests/test_fm_idle_review_headroom_cached_bundle_v5.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
 raw=subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True)
 paths.update(p for p in raw.splitlines() if not p.endswith('.log'))
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),
 previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
base.write(ROOT/'tmp/Idle_review_headroom_and_BRITS45_publication_manifest_20261015_v5.json',manifest)
print(json.dumps(dict(passed=True,metadata_private_before_MiB=old_total/1048576,
 metadata_private_after_MiB=new_total/1048576,released_MiB=(old_total-new_total)/1048576,
 fully_reviewed_SAITS_BRITS_models=45,required_original_models=210,
 independent_metrics=record['independently_recomputed_metrics'],checks_passed=70,publication_files=len(paths))))
