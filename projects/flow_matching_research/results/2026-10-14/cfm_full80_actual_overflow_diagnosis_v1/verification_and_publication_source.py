from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root=cold.ROOT
handofffolder=root/'projects/flow_matching_research/results/2026-10-14/tsad_all121_cached_idle_handoff_v3'
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_full80_actual_overflow_diagnosis_v1'
read=cold.original.read
write=cold.original.write
handoff=read(handofffolder/'actual_all6_controller_handoff.json')
for row in handoff['actual_live_owners']:
    p=psutil.Process(row['pid'])
    assert p.create_time()==row['create_time'] and p.cmdline()==row['command']
cfg=read(root/'configs/runtime/fm_tsad_all121_cached_idle_handoff.v3.json')
for pin in cfg['source_receipts']:
    assert cold.streaming_sha(root/pin['path'])==pin['sha256']
report=read(folder/'independent_every_grid_and_failure_audit_v2.json')
assert report['attempted_original_trajectories']==80 and report['finite_full_grid_trajectories']==79
assert report['failed_trajectory_ids']==[103] and report['independent_checked_grid_coordinates']==40000
runtime=read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
snapshot=read(root/runtime['state_root']/'verified_full_model_records.json')
assert len(snapshot['records'])>=129
identity='cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed1104'
record=next(row for row in snapshot['records'] if row['original_canonical_id']==identity)
cold.validate_closed_primary_record(runtime,record)
target=folder/'new_closed_GP600_sine1104'
target.mkdir(exist_ok=False)
nativepath=root/runtime['child_config_root']/(identity+'.native.json')
native=read(nativepath)
reviewpath=root/runtime['child_config_root']/(identity+'.review.json')
review=read(reviewpath)
for source,name in [(nativepath,'native_configuration.json'),
                    (root/native['output_directory']/'checkpoint_replay_audit.json','native_full_model_report.json'),
                    (root/native['state_root']/'receipt.json','native_guard_receipt.json'),
                    (reviewpath,'review_configuration.json'),
                    (root/review['output_proof'],'independent_full_array_review.json'),
                    (root/review['state_root']/'receipt.json','independent_guard_receipt.json')]:
    (target/name).write_bytes(Path(source).read_bytes())
write(folder/'actual_new129th_CFM_complete_certificate.json',dict(required_original_models=285,
    actual_successful_full_native_and_independent_model_certificates=len(snapshot['records']),
    newly_verified_full_budget_record=record,full80_failure_is_separate_and_not_a_successful_certificate=True,
    all_original_experiments_complete=False,author_equivalence_certified=False))
counts=[]
for path in [handofffolder/'validation.xml',folder/'validation_v2.xml']:
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
    counts.append(sum(int(s.get('tests',0)) for s in suites))
assert counts==[27,7]
state=read(root/runtime['state_root']/'status.json')
owner=psutil.Process(state['pid'])
assert owner.create_time()==state['create_time']
scientific=[]
for p in psutil.process_iter(['pid','name']):
    if 'python' not in (p.info['name'] or '').lower():continue
    try:
        command=p.cmdline()
        if 'scripts.flow_matching.run_cfm_ts_original_job' in command or any('run_saits.py' in s or 'run_anomaly.py' in s for s in command):
            scientific.append(dict(pid=p.pid,create_time=p.create_time(),command=command,cpu_seconds=sum(p.cpu_times()[:2])))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
write(folder/'iteration_actual_scientific_and_runtime_evidence.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    complete_original_diagnostic_trajectories_attempted=80,finite_full_grids=79,failed_full_grids=1,
    independent_full_grid_coordinates_checked=40000,resource_guard_aborted=False,
    original_full80_benchmark_complete=False,full_original_MSE=None,
    actual_new_complete_CFM_certificate=record,actual_successful_CFM_full_model_certificates=len(snapshot['records']),
    CFM_primary_live_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),
    corrected_SMAP_CPU_queues_preserved=121,separate_GPU_controls_preserved=170,
    released_idle_monitor_private_bytes=handoff['released_private_bytes'],validation_tests_passed=34,
    remaining_actual_scientific_worker_handles=scientific,training_worker_stopped=False,
    original_solver_or_numerical_precision_changed=False,author_equivalence_certified=False,strict_TAB_result=False,
    all_original_experiments_complete=False))
for source,name in [(root/'tmp/register_full80_observed_array_audit_20261014.py','original_evidence_registration_source.py'),
                    (Path(__file__),'verification_and_publication_source.py')]:
    (folder/name).write_bytes(source.read_bytes())
methods=['scripts/flow_matching/run_tsad_constant_fit_cached_handoff_v3.py','tests/test_tsad_constant_fit_cached_handoff_v3.py',
         'configs/runtime/fm_tsad_all121_cached_idle_handoff.v3.json',
         'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py','scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
         'tests/test_cfm_full80_observed_arrays_v1.py','tests/test_cfm_full80_observed_arrays_v2.py',
         'configs/reproducibility/fm_cfm_full80_observed_arrays.2026-10-14_v1.json',
         'configs/reproducibility/fm_cfm_full80_observed_arrays.2026-10-14_v2.json']
paths=methods+[p.relative_to(root).as_posix() for directory in [folder,handofffolder] for p in directory.rglob('*')
               if p.is_file() and (p.suffix in {'.json','.xml','.py'} or p.name=='.gitattributes'
                                  or p.name.startswith('native_worker.') and p.suffix=='.log')]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(root/'tmp/tsad_all121_full80_actual_diagnosis_manifest_20261014.json',dict(previous_root=head(root),
      previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),argument_characters=sum(len(p)+3 for p in paths),tests=34,
    CFM_full_certificates=len(snapshot['records']),new_GP600_metrics=record['replayed_metrics'],failed_trajectories=report['failed_trajectory_ids'])))
