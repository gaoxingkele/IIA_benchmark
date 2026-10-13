from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_idle_full_scope_cached_bundle_v1 as cached

root=cached.ROOT
base=root/'projects/flow_matching_research/results/2026-10-14'
folder=base/'spectral_CRNNGAN_image_idle_cached_bundle_v1'
diagnostic=base/'cfm_GP6001105_actual_full80_failure_v1'
read=cached.cold.original.read
write=cached.cold.original.write
cfgpath='configs/runtime/fm_spectral_CRNNGAN_image_idle_cached_bundle.v1.json'
config=read(root/cfgpath)
handoff=read(folder/'actual_all6_cold_host_handoff.json')
current=[]
for row in handoff['new_live_owners']:
    p=psutil.Process(row['pid']);assert p.create_time()==row['create_time'] and p.cmdline()==row['command']
    role,_=cached.verify(config,row['name'])
    state=read(root/role['previous_state_path'])
    assert state['pid']==p.pid and state['create_time']==p.create_time()
    if row['name']=='image325':assert state['completed_case_seed_evaluations']>=6
    current.append(dict(role=row['name'],pid=p.pid,create_time=p.create_time(),command=p.cmdline(),
                        private_bytes=p.memory_info().private,original_slots=role['required_original_slots'],state=state,
                        descendants=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline()) for c in p.children(recursive=True)]))
    source=root/config['handoff_state_root']/row['name']/'actual_first_cached_telemetry.json'
    if source.exists():(folder/row['name']/'actual_first_cached_telemetry.json').write_bytes(source.read_bytes())
post=read(diagnostic/'independent_every_original_grid_and_failure.json')
assert post['attempted_original_trajectories']==80 and post['finite_full_grid_trajectories']==79
assert post['failed_trajectory_ids']==[324] and post['independent_checked_grid_coordinates']==40000
assert post['full_original_MSE'] is None and not post['original_scoring_certified_complete']
native=read(diagnostic/'native_guard_receipt.json')
assert native['exit_code']==0 and not native['resource_guard_aborted'] and native['full80_failure_observation_captured']
tests=0
for xml in [folder/'validation.xml',diagnostic/'validation.xml']:
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
    tests+=sum(int(s.get('tests',0)) for s in suites)
assert tests==23
workers=[]
for p in psutil.process_iter(['pid','name']):
    if 'python' not in (p.info['name'] or '').lower():continue
    try:
        command=p.cmdline()
        if 'scripts.flow_matching.run_cfm_ts_original_job' in command or any('run_saits.py' in a or 'run_anomaly.py' in a for a in command):
            workers.append(dict(pid=p.pid,create_time=p.create_time(),command=command,cpu_seconds=sum(p.cpu_times()[:2])))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
write(folder/'iteration_actual_full_scope_handoff_and_GP_diagnostic.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=config['original_scope'],actual_new_live_host_handles=current,unique_original_scientific_slots_preserved=412,
    full_original_preflight_preserved=True,released_private_bytes=handoff['released_private_bytes'],
    newly_completed_native_original_GP600_full80_failure_observation=True,
    native_GP600_training_epochs=600,native_GP600_original_optimizer_updates=375000,
    independent_original_grid_coordinates_checked=40000,finite_original_trajectories=79,failed_original_trajectory=324,
    failed_original_initial_state=post['failure_evidence'][0]['original_initial_state'],
    observed_training_state_range=post['failure_evidence'][0]['actual_training_state_range'],
    initial_state_inside_observed_training_range=False,original_full80_MSE=None,
    original_scoring_complete=False,causal_training_or_paper_algorithm_root_cause_certified=False,
    actual_remaining_scientific_training_handles=workers,scientific_worker_stopped=False,
    original_budgets_and_solver_tolerances_changed=False,validation_tests_passed=23,
    strict_TAB_result=False,author_equivalence_certified=False,all_original_experiments_complete=False))
for source,name in [(root/'tmp/register_idle_full_scope_cached_bundle_20261014.py','registration_source.py'),
                    (root/'tmp/probe_launch_idle_cached_bundle_20261014.py','actual_cold_handoff_source.py'),
                    (Path(__file__),'verification_and_publication_source.py')]:
    (folder/name).write_bytes(source.read_bytes())
paths=['scripts/flow_matching/run_idle_full_scope_cached_bundle_v1.py','tests/test_idle_full_scope_cached_bundle_v1.py',cfgpath,
       'scripts/flow_matching/audit_cfm_gp600_seed1105_full80_failure_v1.py','tests/test_cfm_gp600_seed1105_full80_failure_v1.py',
       'configs/runtime/fm_cfm_GP6001105_full80_failure_observation.v1.json',
       'configs/reproducibility/fm_cfm_GP6001105_full80_observed_arrays.2026-10-14_v1.json']
paths += [p.relative_to(root).as_posix() for directory in [folder,diagnostic] for p in directory.rglob('*') if p.is_file()
          and (p.suffix in {'.json','.xml','.py'} or p.name=='.gitattributes' or p.name.startswith(('native_worker.','original_failed_scoring.')) and p.suffix=='.log')]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(root/'tmp/idle_bundle_GP600_full80_publication_manifest_20261014.json',dict(previous_root=head(root),
      previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),tests=23,actual_host_handoffs=6,original_scientific_slots=412,native_GP600_full80_attempts=80,finite_trajectories=79,failed_id=324)))
