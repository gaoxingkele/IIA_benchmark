import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as current

root=current.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_GP6001105_actual_full80_failure_v1'
runtimepath='configs/runtime/fm_cfm_GP6001105_full80_failure_observation.v1.json'
runtime=current.read(root/runtimepath)
output=Path(runtime['output_root'])
paths=[runtimepath,'scripts/flow_matching/audit_cfm_gp600_seed1105_full80_failure_v1.py',
       'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py','scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
       str(output/'full80_native_failure_observation.json'),str(output/'all_original_grid_attempts.npz'),
       runtime['state_root']+'/receipt.json']+[str(output/('trajectory_%02d.json'%i)) for i in range(80)]
config=dict(native_configuration=runtimepath,output_report=folder.relative_to(root).as_posix()+'/independent_every_original_grid_and_failure.json',
            source_receipts=[dict(path=p,sha256=current.sha(root/p)) for p in paths],citation=runtime['citation'])
cfgpath=root/'configs/reproducibility/fm_cfm_GP6001105_full80_observed_arrays.2026-10-14_v1.json'
with cfgpath.open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
report=current.post_audit(config)
assert report['attempted_original_trajectories']==80 and report['finite_full_grid_trajectories']==79
assert report['failed_trajectory_ids']==[324] and report['independent_checked_grid_coordinates']==40000
assert report['failure_evidence'][0]['initial_state_inside_observed_training_state_range'] is False
assert report['full_original_MSE'] is None and not report['original_scoring_certified_complete']
current.write(root/config['output_report'],report)
for source,name in [(output/'full80_native_failure_observation.json','full80_native_failure_observation.json'),
                    (output/'trajectory_67.json','failed_original_trajectory324.json'),
                    (root/runtime['state_root']/'receipt.json','native_guard_receipt.json'),
                    (root/runtime['state_root']/'status.json','native_terminal_state.json'),
                    (root/runtime['state_root']/'worker.stdout.log','native_worker.stdout.log'),
                    (root/runtime['state_root']/'worker.stderr.log','native_worker.stderr.log')]:
    (folder/name).write_bytes(source.read_bytes())
individual=folder/'individual_trajectory_receipts';individual.mkdir(exist_ok=False)
for position in range(80):
    (individual/('trajectory_%02d.json'%position)).write_bytes((output/('trajectory_%02d.json'%position)).read_bytes())
(folder/'post_observation_verification_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(attempted=80,finite=79,failed=report['failed_trajectory_ids'],checked_coordinates=40000,failure_evidence=report['failure_evidence'])))
