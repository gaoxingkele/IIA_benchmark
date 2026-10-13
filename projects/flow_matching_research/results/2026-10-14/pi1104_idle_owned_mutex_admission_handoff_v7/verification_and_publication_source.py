from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_pi1104_admission_before_mutex_handoff_v7 as current

root=current.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/pi1104_idle_owned_mutex_admission_handoff_v7'
read=current.cold.original.read;write=current.cold.original.write
cfgpath='configs/runtime/fm_pi1104_admission_before_mutex_handoff.v7.json'
config=read(root/cfgpath)
registration,function,queue,binding,case=current.verify(config)
assert not (root/case['job']['output_directory']).exists()
identity=read(folder/'actual_new_admission_before_mutex_owner.json')['live_owner']
p=psutil.Process(identity['pid']);assert p.create_time()==identity['create_time'] and p.cmdline()==identity['command']
assert not any(Path(f.path).name=='fm_bounded_strict_baseline_cpu.lock' for f in p.open_files())
native=psutil.Process(22432);assert native.create_time()==1791927002.4746633
assert 'pi_author__causal_axis_repaired__psm__full__seed42__full_incident_recovery_v5' in native.cmdline()
native_cpu=sum(native.cpu_times()[:2])
prior=read(folder/'verified_idle_mutex_release_and_started_full_Pi_seed42.json')
assert native_cpu>prior['actual_original_full_seed42_native_child']['cpu_seconds']
suites=list(ET.parse(folder/'validation.xml').getroot().iter('testsuite'))
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
assert sum(int(s.get('tests',0)) for s in suites)==43
saits=read(root/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json')
collector=psutil.Process(saits['pid']);assert collector.create_time()==saits['create_time']
completed='saits_paper_brits_electricity_author_missing0.2_seed28'
assert completed in saits['verified_original_ids'] and len(saits['verified_original_ids'])>=27
verified=read(folder/'new_complete_BRITS_Electricity20_seed28/current_every_grid_and_complete_training_validation.json')
write(folder/'iteration_actual_live_training_and_full_model_evidence.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=config['original_scope'],all_original_incident_records_retained=len(queue['jobs']),all_original_Pi_jobs_retained=1140,
    selected_source1104_40GiB_admission_retained=True,original_shared_bounded_CPU_mutex_retained=True,
    previous_empty_slot_owner_replaced_without_native_worker_stop=True,
    current_source1104_live_owner=identity,current_source1104_phase=read(root/config['previous_state_path'])['state'],
    actual_Pi_seed42_full_native_worker=dict(pid=native.pid,create_time=native.create_time(),command=native.cmdline(),cpu_seconds=native_cpu),
    native_cpu_progress_since_first_actual_posthandoff_check=native_cpu-prior['actual_original_full_seed42_native_child']['cpu_seconds'],
    full_original_SAITS_scope=210,actual_verified_SAITS_full_models=len(saits['verified_original_ids']),
    actual_SAITS_completion_collector=dict(pid=collector.pid,create_time=collector.create_time(),command=collector.cmdline()),
    new_completed_full_BRITS_model=verified,validation_tests_passed=43,
    original_numeric_training_or_replay_code_changed=False,scientific_worker_stopped=False,
    original_early_stopping_and_checkpoint_selection_unchanged=True,full_seed_cohort_complete=False,
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False))
for source,name in [(root/'tmp/register_launch_pi1104_admission_handoff_20261014.py','registration_and_actual_handoff_source.py'),
                    (root/'tmp/verify_brits28_complete_and_pi_mutex_recovery_20261014.py','full_saved_grid_and_mutex_verification_source.py'),
                    (Path(__file__),'verification_and_publication_source.py')]:
    (folder/name).write_bytes(source.read_bytes())
paths=['scripts/flow_matching/run_pi1104_admission_before_mutex_handoff_v7.py','tests/test_pi1104_admission_before_mutex_handoff_v7.py',cfgpath]
paths += [p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.xml'}]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(root/'tmp/pi_idle_mutex_BRITS28_publication_manifest_20261014.json',dict(previous_root=head(root),previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),tests=43,SAITS_verified_full_models=len(saits['verified_original_ids']),original_scope=210,
                     Pi_seed42_native_CPU_progress=native_cpu-prior['actual_original_full_seed42_native_child']['cpu_seconds'],new_BRITS_metrics=verified['metrics_independently_recomputed'])))
