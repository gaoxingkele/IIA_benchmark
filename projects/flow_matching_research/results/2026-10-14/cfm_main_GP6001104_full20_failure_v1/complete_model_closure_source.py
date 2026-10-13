"""Check and export every new complete certificate beyond the132 snapshot."""
from pathlib import Path
from datetime import datetime,timezone
from types import FunctionType
import json
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root=cold.ROOT;folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_main_GP6001104_full20_failure_v1'
runtime=cold.original.read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
snapshot=cold.original.read(root/'experiments/runs/fm_cfm_completion_replay_queue_v4/verified_full_model_records.json')
prior=cold.original.read(root/'projects/flow_matching_research/results/2026-10-14/cfm_idle_streaming128_handoff_v3/preserved_actual128_guarded_records.json')
later=cold.original.read(root/'projects/flow_matching_research/results/2026-10-14/cfm_GP6001103_actual_full80_failure_v2/current_new132_certificates_and_GP600_failure_cohort.json')
priorrecords={r['original_canonical_id']:r for r in prior['records']+later['checked_records']}
assert len(priorrecords)==132
priorids=set(priorrecords)
assert all(r in snapshot['records'] for r in priorrecords.values())
new=[r for r in snapshot['records'] if r['original_canonical_id'] not in priorids]
assert new
verify=FunctionType(cold.original.verify_record.__code__,dict(cold.original.__dict__,sha=cold.streaming_sha))
for record in new:
    identity=record['original_canonical_id'];cold.validate_closed_primary_record(runtime,record)
    nativepath=root/runtime['child_config_root']/(identity+'.native.json');native=cold.original.read(nativepath)
    canonical=next(r for r in native['full_results'] if r['original_canonical_id']==identity)
    assert verify(runtime,canonical,record)==record
    reviewpath=root/runtime['child_config_root']/(identity+'.review.json');review=cold.original.read(reviewpath)
    target=folder/('new_closed_'+identity);target.mkdir(exist_ok=False)
    copies={'native_configuration.json':nativepath,'independent_configuration.json':reviewpath,
        'native_checkpoint_full_replay_report.json':root/native['output_directory']/'checkpoint_replay_audit.json',
        'native_guard_receipt.json':root/native['state_root']/'receipt.json',
        'independent_full_array_proof.json':root/review['output_proof'],
        'independent_guard_receipt.json':root/review['state_root']/'receipt.json',
        'original_result.json':root/record['result_path']}
    for name,p in copies.items():(target/name).write_bytes(p.read_bytes())
    (target/'current_all_array_verification.json').write_text(json.dumps(record,indent=2)+'\n')
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),current_automatic_verified_count=len(snapshot['records']),
    original_required_CFM_models=285,original_scope=runtime['original_scope'],
    newly_closed_current_full_models=len(new),all_new_native_guarded_and_independent_proof_chains_and_arrays_verified=True,
    new_records=new,additional_optimizer_updates=0,additional_solver_calls=0,
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
(folder/'new_complete_model_certificate_closure.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'complete_model_closure_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'verified_count':len(snapshot['records']),'newly_closed':len(new),'ids':[r['original_canonical_id'] for r in new]}))
