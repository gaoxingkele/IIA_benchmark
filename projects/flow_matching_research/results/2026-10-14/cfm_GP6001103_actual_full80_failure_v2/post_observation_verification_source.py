"""Finish the one native diagnostic only after its actual terminal receipt."""
import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_gp600_seed1103_full80_failure_v2 as current

root=current.ROOT
cfgpath='configs/runtime/fm_cfm_GP6001103_full80_failure_observation.v2.json'
runtime=current.read(root/cfgpath)
base=root/runtime['state_root'];output=Path(runtime['output_root'])
receipt=current.read(base/'receipt.json')
assert receipt['exit_code']==0 and not receipt['resource_guard_aborted'] and receipt['full80_failure_observation_captured']
assert not receipt['original_benchmark_scoring_complete']
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_GP6001103_actual_full80_failure_v2'
postpath='configs/reproducibility/fm_cfm_GP6001103_full80_observed_arrays.2026-10-14_v2.json'
post=dict(native_configuration=cfgpath,output_report=folder.relative_to(root).as_posix()+'/independent_every_original_grid_and_failure.json')
pins=[root/cfgpath,root/'scripts/flow_matching/audit_cfm_gp600_seed1103_full80_failure_v2.py',
    root/'scripts/flow_matching/audit_cfm_gp600_seed1105_full80_failure_v1.py',
    root/'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py',root/'scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
    output/'full80_native_failure_observation.json',output/'all_original_grid_attempts.npz',base/'receipt.json']
pins += sorted(output.glob('trajectory_*.json'))
assert len(list(output.glob('trajectory_*.json')))==80
post['source_receipts']=[dict(path=p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p),sha256=current.sha(p)) for p in pins]
with (root/postpath).open('x',encoding='utf-8') as stream:json.dump(post,stream,indent=2)
report=current.post_audit(post)
assert report['independent_checked_grid_coordinates']==40000
assert not (root/post['output_report']).exists()
current.write(root/post['output_report'],report)
copies={'native_guard_receipt.json':base/'receipt.json','native_terminal_state.json':base/'status.json',
    'native_worker.stdout.log':base/'worker.stdout.log','native_worker.stderr.log':base/'worker.stderr.log',
    'controller.stdout.log':base/'controller.stdout.log','controller.stderr.log':base/'controller.stderr.log',
    'full80_native_failure_observation.json':output/'full80_native_failure_observation.json',
    'post_observation_verification_source.py':Path(__file__)}
for name,p in copies.items():(folder/name).write_bytes(p.read_bytes())
receipts=folder/'all80_original_trajectory_receipts';receipts.mkdir(exist_ok=False)
for p in sorted(output.glob('trajectory_*.json')):(receipts/p.name).write_bytes(p.read_bytes())
print(json.dumps({k:report[k] for k in ['attempted_original_trajectories','finite_full_grid_trajectories','failed_trajectory_ids','independent_checked_grid_coordinates','failure_evidence']}))
