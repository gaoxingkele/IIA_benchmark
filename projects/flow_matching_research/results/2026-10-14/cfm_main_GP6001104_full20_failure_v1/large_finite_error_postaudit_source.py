"""Post-audit every stored coordinate; never infer, train, trim or tune."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays
from scripts.flow_matching.audit_cfm_gp_seed_dispersion_v1 import dispersion

root=cold.ROOT
relative='projects/flow_matching_research/results/2026-10-14/cfm_main_GP6001104_full20_failure_v1'
folder=root/relative
closure=cold.original.read(folder/'new_complete_model_certificate_closure.json')
identity='cfm_ts_original__main_text__gp_paper_literal_e600__sine__seed1106'
record=next(r for r in closure['new_records'] if r['original_canonical_id']==identity)
runtime=cold.original.read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
cold.validate_closed_primary_record(runtime,record)
resultpath=root/record['result_path'];result=cold.original.read(resultpath);job=result['job']
predictions=next(a['path'] for a in result['artifacts'] if Path(a['path']).name=='predictions.npz')
pins=[resultpath,root/job['data_path'],root/job['model_config'],Path(predictions),Path(record['checkpoint_path']),
    root/'scripts/flow_matching/audit_cfm_full_arrays_v1.py',root/'scripts/flow_matching/audit_cfm_gp_seed_dispersion_v1.py',Path(__file__)]
config=dict(canonical_job=job,source_receipts=[dict(path=p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p),sha256=cold.streaming_sha(p)) for p in pins],
    output_report=relative+'/main_GP6001106_complete_finite_large_error_diagnosis.json',
    analysis_boundary='All original20 held-out trajectories and all10000 coordinates remain. Concentration and quantiles diagnose the complete MSE; they do not replace it.',
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ')
cfgpath=root/'configs/reproducibility/fm_cfm_main_GP6001106_stored_grid_diagnosis.2026-10-14_v1.json'
with cfgpath.open('x',encoding='utf-8') as f:json.dump(config,f,indent=2)
with np.load(predictions,allow_pickle=False) as arrays,np.load(root/job['data_path'],allow_pickle=False) as data:
    metrics=audit_arrays(job,result,arrays,data)
    detail=dispersion(arrays['per_trajectory_MSE'],arrays['test_ids'],arrays['initial_states'],arrays['times'],arrays['predictions'],arrays['truth'])
    training=data['observed_states'][data['train_ids']];limits={'min':float(training.min()),'max':float(training.max())}
    for row in detail['largest_error_trajectories']:
        row['initial_inside_observed_training_state_range']=all(limits['min']<=x<=limits['max'] for x in row['initial_state'])
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_job=job,
        full_original_metrics=metrics,all_original_coordinates_checked=int(arrays['predictions'].size),
        original_training_state_range=limits,all_original_grids_finite=True,**detail,
        native_full_model_and_independent_array_proofs_verified=True,
        original_replay_prediction_maximum_difference=record['max_absolute_difference'],
        result_sha256=record['result_sha256'],checkpoint_sha256=record['checkpoint_sha256'],
        additional_optimizer_updates=0,additional_ODE_solver_calls=0,test_based_parameter_selection=False,
        inference_repeatability_is_not_evidence_of_good_performance=True,
        training_root_cause_or_author_code_bug_certified=False,strict_TAB_result=False,
        author_equivalence_certified=False,all_original_experiments_complete=False)
(root/config['output_report']).write_text(json.dumps(report,indent=2)+'\n')
(folder/'large_finite_error_postaudit_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:report[k] for k in ['full_MSE','trajectory_MSE_median','top1_squared_error_fraction','largest_error_trajectories']}))
