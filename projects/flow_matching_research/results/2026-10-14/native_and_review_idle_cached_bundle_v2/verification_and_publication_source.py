from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_idle_native_and_review_cached_bundle_v2 as cached

root=cached.previous.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/native_and_review_idle_cached_bundle_v2'
read=cached.previous.cold.original.read
write=cached.previous.cold.original.write
cfgpath='configs/runtime/fm_native_and_review_idle_cached_bundle.v2.json'
config=read(root/cfgpath)
handoff=read(folder/'actual_all4_cold_host_handoff.json')
current=[]
for row in handoff['actual_live_owners']:
    p=psutil.Process(row['pid']);assert p.create_time()==row['create_time'] and p.cmdline()==row['command']
    role,_=cached.namespace()['verify'](config,row['role'])
    state=read(root/role['previous_state_path']);assert state['pid']==p.pid and state['create_time']==p.create_time()
    current.append(dict(role=row['role'],live_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private),state=state))
    source=root/config['handoff_state_root']/row['role']/'actual_first_cached_telemetry.json'
    if source.exists():(folder/row['role']/'actual_first_cached_telemetry.json').write_bytes(source.read_bytes())
runtime=read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
snapshot=read(root/runtime['state_root']/'verified_full_model_records.json')
assert len(snapshot['records'])>=130
identity='cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed1106'
record=next(r for r in snapshot['records'] if r['original_canonical_id']==identity)
cached.previous.cold.validate_closed_primary_record(runtime,record)
target=folder/'new_closed_GP600_sine1106';target.mkdir(exist_ok=False)
nativepath=root/runtime['child_config_root']/(identity+'.native.json');native=read(nativepath)
reviewpath=root/runtime['child_config_root']/(identity+'.review.json');review=read(reviewpath)
for source,name in [(nativepath,'native_configuration.json'),
                    (root/native['output_directory']/'checkpoint_replay_audit.json','native_full_model_report.json'),
                    (root/native['state_root']/'receipt.json','native_guard_receipt.json'),
                    (reviewpath,'independent_configuration.json'),
                    (root/review['output_proof'],'independent_full_array_proof.json'),
                    (root/review['state_root']/'receipt.json','independent_guard_receipt.json')]:
    (target/name).write_bytes(Path(source).read_bytes())
write(folder/'actual_new130th_CFM_complete_certificate.json',dict(required_original_models=285,
    actual_successful_full_native_and_independent_certificates=len(snapshot['records']),newly_verified_full_budget_record=record,
    complete_five_seed_GP600_cohort_proved=False,all_original_experiments_complete=False,author_equivalence_certified=False))
suites=list(ET.parse(folder/'validation.xml').getroot().iter('testsuite'))
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
assert sum(int(s.get('tests',0)) for s in suites)==12
workers=[]
for p in psutil.process_iter(['pid','name']):
    if 'python' not in (p.info['name'] or '').lower():continue
    try:
        cmd=p.cmdline()
        if ('scripts.flow_matching.run_cfm_ts_original_job' in cmd
                or any('run_saits.py' in a or 'run_anomaly.py' in a or 'run_tsad_experiment' in a or 'run_pi_transformer_job' in a for a in cmd)):
            workers.append(dict(pid=p.pid,create_time=p.create_time(),command=cmd,cpu_seconds=sum(p.cpu_times()[:2])))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
write(folder/'iteration_actual_CFM130_and_full_scope_live_evidence.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=config['original_scope'],retained_original_Energy_commands=4,retained_complete_TCCM_capacity_cases=1,
    retained_GRASP_full_independent_reviews=2,actual_live_idle_host_successors=current,
    released_private_bytes=handoff['released_private_bytes'],new_full_original_CFM_certificate=record,
    successful_complete_CFM_models=len(snapshot['records']),required_original_CFM_models=285,
    original_GP600_Sine_seed1106_epochs=600,original_GP600_Sine_seed1106_optimizer_updates=375000,
    original_GP600_Sine_seed1106_full_grid=[80,500,1],original_GP600_Sine_seed1106_max_replay_difference=0.,
    complete_five_seed_GP600_cohort_or_author_equivalence_certified=False,validation_tests_passed=12,
    actual_remaining_scientific_training_handles=workers,scientific_worker_stopped=False,
    unchanged_resource_guards_and_numerical_budgets=True,strict_TAB_result=False,all_original_experiments_complete=False))
for source,name in [(root/'tmp/register_native_review_cached_bundle_v2_20261014.py','registration_source.py'),
                    (root/'tmp/probe_launch_native_review_bundle_v2_20261014.py','actual_cold_handoff_source.py'),
                    (Path(__file__),'verification_and_publication_source.py')]:
    (folder/name).write_bytes(source.read_bytes())
paths=['scripts/flow_matching/run_idle_native_and_review_cached_bundle_v2.py','tests/test_idle_native_and_review_cached_bundle_v2.py',cfgpath]
paths += [p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json','.xml','.py'}]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(root/'tmp/native_review_bundle_CFM130_publication_manifest_20261014.json',dict(previous_root=head(root),previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),tests=12,released_private_bytes=handoff['released_private_bytes'],CFM_full_certificates=len(snapshot['records']),new_GP600_metrics=record['replayed_metrics'],actual_scientific_worker_PIDs=[p['pid'] for p in workers])))
