from datetime import datetime,timezone
import json
from pathlib import Path
import h5py
import numpy as np
import psutil
from scripts.flow_matching import verify_saits_completed_full_replay_batch_v1 as review
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root=cold.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/pi1104_idle_owned_mutex_admission_handoff_v7'
identity=cold.original.read(folder/'actual_new_admission_before_mutex_owner.json')['live_owner']
owner=psutil.Process(identity['pid'])
assert owner.create_time()==identity['create_time'] and owner.cmdline()==identity['command']
assert not any(Path(f.path).name=='fm_bounded_strict_baseline_cpu.lock' for f in owner.open_files())
locks=cold.original.read(root/'tmp/actual_CPU_GPU_open_lock_handles_20261014.json')
(folder/'actual_CPU_GPU_lock_handles_after_handoff.json').write_bytes((root/'tmp/actual_CPU_GPU_open_lock_handles_20261014.json').read_bytes())
active=next(p for p in locks['actual_open_lock_handles'] if any(Path(path).name=='fm_bounded_strict_baseline_cpu.lock' for path in p['open_lock_handles']))
assert active['pid']==9344 and active['create_time']==1791732960.307726
p=psutil.Process(active['pid']);assert p.create_time()==active['create_time'] and p.cmdline()==active['command']
native=next(c for c in p.children(recursive=True) if c.pid==22432)
assert native.create_time()==1791927002.4746633
assert 'scripts.flow_matching.run_resource_recovery_job' in native.cmdline()
assert 'pi_author__causal_axis_repaired__psm__full__seed42__full_incident_recovery_v5' in native.cmdline()
name='saits_paper_brits_electricity_author_missing0.2_seed28'
cfgpath='configs/reproducibility/fm_saits_completion_replay_queue_v1/'+name+'.review.json'
config=cold.original.read(root/cfgpath)
producer=review.verify_registration(config)
proofpath=root/config['output_directory']/'independent_full_replay_batch_verification.json'
proof=cold.original.read(proofpath)
guard=cold.original.read(root/config['state_root']/'receipt.json')
reportpath=root/producer['artifact_directory']/'checkpoint_replay_audit.json'
report=cold.original.read(reportpath)
nativeguardpath=root/producer['state_root']/'receipt.json'
nativeguard=cold.original.read(nativeguardpath)
review.check_terminal(report,nativeguard,producer)
assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
assert proof['configuration_sha256']==cold.streaming_sha(root/cfgpath)
assert proof['native_replay_report_sha256']==cold.streaming_sha(reportpath)
assert guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_native_batch_independent_review_passed']
row=proof['records'][0];item=producer['completed_jobs'][0];nativecfg=cold.original.read(root/item['config'])
resultpath=root/nativecfg['output_root']/'result.json';result=cold.original.read(resultpath)
native_record=report['records'][0]
review.verify_record_identity(native_record,nativecfg,result,item)
output=root/nativecfg['output_root'];data=root/nativecfg['dataset_root']
paths=dict(config=root/item['config'],result=resultpath,checkpoint=output/'best.pt',arrays=Path(native_record['replay_arrays']),
           dataset=data/'datasets.h5',resume=output/'resume.pt',original_group_statistics=output/'group_statistics.npz',data_audit=data/'data_audit.json')
for key,path in paths.items():assert cold.streaming_sha(path)==row['hashes'][key]
with h5py.File(data/'datasets.h5','r') as stream:truth,mask=stream['test']['X'][:],stream['test']['indicating_mask'][:]
with np.load(data/'audit_arrays.npz',allow_pickle=False) as ids:units,groups=ids['test_ids'],ids['test_groups']
with np.load(paths['arrays'],allow_pickle=False) as arrays:
    metrics,grouped=review.check_arrays(arrays,truth,mask,units,groups,result['target_count'],producer['metric_tolerance'])
for metric,value in metrics.items():assert np.isclose(value,result['metrics'][metric],**producer['metric_tolerance'])
with np.load(output/'group_statistics.npz',allow_pickle=False) as original:
    assert np.array_equal(original['group_ids'],np.unique(groups)) and np.allclose(grouped,original['statistics'],**producer['metric_tolerance'])
target=folder/'new_complete_BRITS_Electricity20_seed28';target.mkdir(exist_ok=False)
for source,filename in [(root/cfgpath,'independent_configuration.json'),(root/config['replay_configuration'],'native_configuration.json'),
                         (proofpath,'independent_full_array_proof.json'),(reportpath,'native_full_model_report.json'),
                         (root/config['state_root']/'receipt.json','independent_guard_receipt.json'),
                         (nativeguardpath,'native_guard_receipt.json'),(resultpath,'original_complete_result.json')]:
    (target/filename).write_bytes(Path(source).read_bytes())
evidence=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=name,metrics_independently_recomputed=metrics,
               full_test_shape=list(truth.shape),full_test_grid_cells=int(truth.size),held_out_coordinates=result['target_count'],
               original_full_training_completion=row['full_training_completion'],all_checkpoint_data_and_grid_hashes_current=True,
               month_groups_independently_recomputed=True,month_groups=len(grouped),model_parameters=result['parameters'],
               original_metric_tolerance=producer['metric_tolerance'],independent_native_full_replay_and_guard_verified=True,
               additional_model_inference=0,additional_training_updates=0,original_configuration_boundary=nativecfg['boundary'],
               dataset_and_model_seed_cohort_complete=False,author_equivalence_certified=False,strict_TAB_result=False,
               all_original_experiments_complete=False)
cold.original.write(target/'current_every_grid_and_complete_training_validation.json',evidence)
cold.original.write(folder/'verified_idle_mutex_release_and_started_full_Pi_seed42.json',dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),idle_40GiB_waiter_current_live_owner=identity,
    idle_waiter_no_longer_has_shared_CPU_mutex=True,Pi_source1104_original_40GiB_admission_unchanged=True,
    original_full_seed42_recovery_parent=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),
    actual_original_full_seed42_native_child=dict(pid=native.pid,create_time=native.create_time(),command=native.cmdline(),cpu_seconds=sum(native.cpu_times()[:2])),
    original_resource_lock_preserved='experiments/runs/fm_bounded_strict_baseline_cpu.lock',scientific_worker_stopped=False,
    newly_completed_BRITS_Electricity20_seed28=evidence,all_original_experiments_complete=False))
print(json.dumps(dict(new_full_native_BRITS_metrics=metrics,full_test_shape=list(truth.shape),held_out_coordinates=result['target_count'],
                     full_training_completion=row['full_training_completion'],started_original_Pi_seed42_native_pid=native.pid)))
