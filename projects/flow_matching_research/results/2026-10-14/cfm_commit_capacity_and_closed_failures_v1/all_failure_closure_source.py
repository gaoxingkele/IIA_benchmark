"""Verify every presently known original saved-model scoring failure exactly.

All models were already observed once; this reads their original complete grids,
failure receipts and hash-bound proofs, without inference or additional training.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as BB80
from scripts.flow_matching import audit_cfm_gp600_seed1103_full80_failure_v2 as GP3
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as GP5
from scripts.flow_matching import audit_cfm_main_GP6001104_full20_failure_v1 as GP20
from scripts.flow_matching import audit_cfm_main_BB6001106_full20_failure_v1 as BB20

root=GP3.ROOT
items=[(BB80.audit,'configs/reproducibility/fm_cfm_full80_observed_arrays.2026-10-14_v2.json'),
    (GP3.post_audit,'configs/reproducibility/fm_cfm_GP6001103_full80_observed_arrays.2026-10-14_v2.json'),
    (GP5.post_audit,'configs/reproducibility/fm_cfm_GP6001105_full80_observed_arrays.2026-10-14_v1.json'),
    (GP20.post_audit,'configs/reproducibility/fm_cfm_main_GP6001104_full20_observed_arrays.2026-10-14_v1.json'),
    (BB20.post_audit,'configs/reproducibility/fm_cfm_main_BB6001106_full20_observed_arrays.2026-10-14_v1.json')]
closed=[]
for call,path in items:
    config=GP3.read(root/path);fresh=call(config);stored=GP3.read(root/config['output_report'])
    for key,value in fresh.items():
        if key!='captured_utc' and stored[key]!=value:raise ValueError('Immutable independent failure proof differs: '+key)
    assert stored['full_original_MSE'] is None and not stored['original_scoring_certified_complete']
    assert stored['additional_ODE_solver_calls']==0 and stored['additional_optimizer_updates']==0
    closed.append(dict(canonical_job_id=stored['canonical_job_id'],configuration=path,
        proof_path=config['output_report'],proof_sha256=GP3.sha(root/config['output_report']),
        full_original_shape=stored['independent_full_grid_shape'],attempted=stored['attempted_original_trajectories'],
        finite_trajectories=stored['finite_full_grid_trajectories'],failed_ids=stored['failed_trajectory_ids'],
        independently_checked_coordinates=stored['independent_checked_grid_coordinates'],full_original_MSE=None,
        all_grid_coordinates_and_failure_accounting_verified=True))
queue=GP3.read(root/'configs/experiments/fm_cfm_ts_original_queue.v1.json');known=[]
for job in queue['jobs']:
    receipt=root/queue['state_root']/'receipts'/(job['id']+'.json')
    if not receipt.exists():continue
    source=GP3.read(receipt)
    if source['complete'] or source['resource_observations']['resource_guard_aborted']:continue
    stderr=(root/queue['state_root']/'logs'/(job['id']+'.stderr.log')).read_text(encoding='utf-8',errors='replace')
    if 'AssertionError: underflow in dt 0.0' in stderr and (Path(job['artifact_directory'])/'model.pt').exists():known.append(job['id'])
ids={r['canonical_job_id'] for r in closed};missing=sorted(set(known)-ids)
assert not ids-set(known)
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_source_slots=len(queue['jobs']),
    known_original_saved_model_scoring_failures=len(known),complete_once_observed_models=len(closed),
    all_present_known_original_saved_scoring_failures_observed=not missing,unobserved_original_saved_scoring_failures=missing,
    records=closed,total_attempted_original_trajectories=sum(r['attempted'] for r in closed),
    total_independently_checked_grid_coordinates=sum(r['independently_checked_coordinates'] for r in closed),
    every_original_failed_metric_remains_missing=True,successful_subsets_never_promoted_as_full_scores=True,
    additional_ODE_solver_calls=0,additional_optimizer_updates=0,all_original_experiments_complete=False)
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_commit_capacity_and_closed_failures_v1'
with (folder/'all_present_original_saved_scoring_failures_closed_observation_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
(folder/'all_failure_closure_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:report[k] for k in ['known_original_saved_model_scoring_failures','complete_once_observed_models','all_present_known_original_saved_scoring_failures_observed','total_attempted_original_trajectories','total_independently_checked_grid_coordinates']}))
