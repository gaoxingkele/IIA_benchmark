"""Read current proof chains and full arrays; never run an optimizer or solver."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root=cold.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_GP6001103_actual_full80_failure_v2'
runtime=cold.original.read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
snapshot=cold.original.read(root/'experiments/runs/fm_cfm_completion_replay_queue_v4/verified_full_model_records.json')
records={r['original_canonical_id']:r for r in snapshot['records']}
assert len(records)==len(snapshot['records'])
verify_arrays=FunctionType(cold.original.verify_record.__code__,dict(cold.original.__dict__,sha=cold.streaming_sha))
identities=['cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed'+str(s) for s in (1104,1106,1107)]
identities+=['cfm_ts_original__main_text__gp_paper_literal_e600__sine__seed1103']
closed=[]
for identity in identities:
    record=records[identity]
    cold.validate_closed_primary_record(runtime,record)
    route=next(r for r in runtime['routes'] if r['original_canonical_id']==identity)
    nativepath=root/runtime['child_config_root']/(identity+'.native.json')
    native=cold.original.read(nativepath)
    canonical=next(r for r in native['full_results'] if r['original_canonical_id']==identity)
    assert verify_arrays(runtime,canonical,record)==record
    closed.append(record)
    if identity.endswith('seed1107') or '__main_text__' in identity:
        destination=folder/('new_closed_'+identity);destination.mkdir(exist_ok=False)
        nativepath=root/runtime['child_config_root']/(identity+'.native.json')
        reviewpath=root/runtime['child_config_root']/(identity+'.review.json')
        native=cold.original.read(nativepath);review=cold.original.read(reviewpath)
        copies={
            'native_configuration.json':nativepath,'independent_configuration.json':reviewpath,
            'native_checkpoint_full_replay_report.json':root/native['output_directory']/'checkpoint_replay_audit.json',
            'native_guard_receipt.json':root/native['state_root']/'receipt.json',
            'independent_full_array_proof.json':root/review['output_proof'],
            'independent_guard_receipt.json':root/review['state_root']/'receipt.json',
            'original_result.json':root/record['result_path'],
            'original_source_guard_receipt.json':root/'experiments/runs/fm_cfm_ts_original_v1/receipts'/(identity+'.json'),
        }
        for name,source in copies.items():(destination/name).write_bytes(source.read_bytes())
        (destination/'current_all_array_verification.json').write_text(json.dumps(record,indent=2)+'\n')

failed=[]
for seed in (1103,1105):
    identity='cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed'+str(seed)
    source=root/'experiments/runs/fm_cfm_ts_original_v1'
    receipt=cold.original.read(source/'receipts'/(identity+'.json'))
    progress=cold.original.read(source/'jobs'/identity/'progress.json')
    assert receipt['exit_code']==1 and not receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
    assert progress['phase']=='scoring' and progress['required_test_trajectories']==80
    assert not (source/'jobs'/identity/'result.json').exists()
    failed.append(dict(seed=seed,canonical_job=identity,original_source_scoring_failed=True,
        original_scoring_trajectories_completed=progress['test_trajectories_completed'],
        full_cohort_score=None,resource_abort=False,trained_checkpoint_metadata_verification_pending=seed==1103))

report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=runtime['original_scope'],
    original_CFM_routes=len(runtime['routes']),current_automatic_verified_record_count=len(records),
    current_verified_record_snapshot_sha256=cold.streaming_sha(root/'experiments/runs/fm_cfm_completion_replay_queue_v4/verified_full_model_records.json'),
    this_turn_freshly_closed_models=2,this_turn_revalidated_full_models=4,
    all_checked_source_and_native_arrays=True,all_checked_native_and_independent_guard_proof_chains=True,
    checked_records=closed,appendix_GP600_Sine_original_seeds=[1103,1104,1105,1106,1107],
    appendix_GP600_Sine_complete_original_scores=3,appendix_GP600_Sine_numerical_source_failures=failed,
    appendix_GP600_Sine_five_seed_mean=None,appendix_GP600_Sine_full_cohort_performance_complete=False,
    main_text_and_appendix_d_are_distinct_recipes=True,additional_optimizer_updates=0,additional_solver_calls=0,
    original_author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
(folder/'current_new132_certificates_and_GP600_failure_cohort.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'closed_certificate_post_verification_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:report[k] for k in ['current_automatic_verified_record_count','this_turn_freshly_closed_models','this_turn_revalidated_full_models','appendix_GP600_Sine_complete_original_scores','appendix_GP600_Sine_full_cohort_performance_complete']}))
