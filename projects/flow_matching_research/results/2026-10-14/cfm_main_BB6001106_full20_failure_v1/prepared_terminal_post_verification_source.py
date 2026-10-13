"""Independent full10000-coordinate audit after the single native20 attempt."""
import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_main_BB6001106_full20_failure_v1 as current

root=current.ROOT
cfgpath='configs/runtime/fm_cfm_main_BB6001106_full20_failure_observation.v1.json';runtime=current.read(root/cfgpath)
base=root/runtime['state_root'];output=Path(runtime['output_root'])
receipt=current.read(base/'receipt.json')
assert receipt['exit_code']==0 and not receipt['resource_guard_aborted'] and receipt['full20_failure_observation_captured']
assert not receipt['original_benchmark_scoring_complete']
relative='projects/flow_matching_research/results/2026-10-14/cfm_main_BB6001106_full20_failure_v1';folder=root/relative
post=dict(native_configuration=cfgpath,output_report=relative+'/independent_every_original_grid_and_failure.json')
pins=[root/cfgpath,root/'scripts/flow_matching/audit_cfm_main_BB6001106_full20_failure_v1.py',
    root/'scripts/flow_matching/audit_cfm_gp600_seed1105_full80_failure_v1.py',
    root/'scripts/flow_matching/audit_cfm_sine600_full80_failure_observation_v1.py',
    root/'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py',root/'scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
    output/'full20_native_failure_observation.json',output/'all_original_grid_attempts.npz',base/'receipt.json']
pins.append(root/'scripts/flow_matching/audit_cfm_main_GP6001104_full20_failure_v1.py')
pins+=sorted(output.glob('trajectory_*.json'))
assert len(list(output.glob('trajectory_*.json')))==20
post['source_receipts']=[dict(path=p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p),sha256=current.sha(p)) for p in pins]
postpath=root/'configs/reproducibility/fm_cfm_main_BB6001106_full20_observed_arrays.2026-10-14_v1.json'
with postpath.open('x',encoding='utf-8') as f:json.dump(post,f,indent=2)
report=current.post_audit(post)
assert report['independent_checked_grid_coordinates']==10000 and report['independent_full_grid_shape']==[20,500,1]
assert not (root/post['output_report']).exists()
current.write(root/post['output_report'],report)
copies={'native_guard_receipt.json':base/'receipt.json','native_terminal_state.json':base/'status.json',
    'native_worker.stdout.log':base/'worker.stdout.log','native_worker.stderr.log':base/'worker.stderr.log',
    'controller.stdout.log':base/'controller.stdout.log','controller.stderr.log':base/'controller.stderr.log',
    'full20_native_failure_observation.json':output/'full20_native_failure_observation.json',
    'post_observation_verification_source.py':Path(__file__)}
for name,p in copies.items():(folder/name).write_bytes(p.read_bytes())
receipts=folder/'all20_original_trajectory_receipts';receipts.mkdir(exist_ok=False)
for p in sorted(output.glob('trajectory_*.json')):(receipts/p.name).write_bytes(p.read_bytes())
print(json.dumps({k:report[k] for k in ['attempted_original_trajectories','finite_full_grid_trajectories','failed_trajectory_ids','independent_checked_grid_coordinates','failure_evidence']}))
