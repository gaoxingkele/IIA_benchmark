from datetime import datetime,timezone
import json
from pathlib import Path
from scripts.flow_matching import run_pi_causal1106_completed_full_replay_v1 as replay
from scripts.flow_matching import verify_pi_original_full_replay_completion_v1 as verifier

root,read,sha=replay.ROOT,replay.read,replay.digest
runtimepath='configs/runtime/fm_pi_causal1106_completed_full_replay.v1.json'
runtime=read(runtimepath);replay.verify(runtime)
case=runtime['cases'][0]
point=read(case['point_configuration']);completion=read(point['output_directory']+'/independent_full_recovery_completion.json')
nativepath=runtime['configuration_root']+'/'+replay.CANONICAL+'_native_ready.json'
reviewpath=runtime['configuration_root']+'/'+replay.CANONICAL+'_independent.json'
native,review=read(nativepath),read(reviewpath)
reportpath=review['completed_replay_report'];report=read(reportpath)
native_receipt=read(native['state_root']+'/receipt.json')
verifier.verify_terminal(report,native_receipt,native,completion)
proofpath=review['output_directory']+'/independent_completed_native_replay.json'
proof=read(proofpath);guard=read(review['state_root']+'/receipt.json')
assert proof['passed'] and proof['canonical_id']==replay.CANONICAL and proof['seed']==1106
assert proof['configuration_sha256']==sha(reviewpath) and proof['completed_native_replay_report_sha256']==sha(reportpath)
assert proof['all_fifteen_full_saved_and_replay_streams_independently_reread']
assert proof['full_original_model_checkpoint_reinference_completed']
assert guard['exit_code']==0 and guard['passed'] and not guard['resource_guard_aborted']
assert len(report['all_original_score_streams_replayed'])==15
assert all(v['bitwise_equal'] and v['maximum_absolute_error']==0 for v in report['all_original_score_streams_replayed'].values())
assert proof['original_score_sha256']==proof['replay_score_sha256']
assert proof['branch']=='causal_axis_repaired'
assert report['full_original_training_score_points']==13238200 and report['full_original_test_score_points']==87800
assert report['dropped_original_test_tail_points']==41 and report['full_parameter_count']==1930289
assert report['config_sha256']==sha(nativepath)
for pin in proof['source_receipts']:assert sha(pin['path'])==pin['sha256']
folder=root/'projects/flow_matching_research/results/2026-10-15/pi_causal1106_actual_complete_model_replay_closure_v1'
folder.mkdir(exist_ok=False)
sources=dict(native_configuration=nativepath,independent_configuration=reviewpath,completed_native_report=reportpath,
    native_guarded_receipt=native['state_root']+'/receipt.json',independent_full_stream_proof=proofpath,
    independent_guarded_receipt=review['state_root']+'/receipt.json',
    completed_actual_pipeline_state=runtime['state_root']+'/status.json')
state=read(sources['completed_actual_pipeline_state']);assert state['active_pid'] is None and state['verified_ready_original_ids']==[replay.CANONICAL]
for name,path in sources.items():(folder/(name+'.json')).write_bytes((root/path).read_bytes())
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=replay.CANONICAL,
    selected_complete_models=1,required_original_jobs=1140,required_original_training_seeds=[42,1103,1104,1105,1106,1107],
    original_full_training_epochs=3,optimizer_minibatches=1554,full_model_parameters=1930289,
    full_train_score_points=13238200,scored_original_test_points=87800,original_dropped_tail_points=41,
    complete_original_score_streams=15,all_streams_bitwise_identical_to_original_saved_arrays=True,
    full_native_checkpoint_reinference_and_independent_stream_metric_review_complete=True,
    independently_recomputed_metrics=proof['independently_recomputed_metrics'],
    confusion_no_PA=proof['confusion_no_PA'],confusion_author_PA=proof['confusion_author_PA'],
    source_threshold=proof['source_threshold'],original_scope=runtime['original_scope'],
    additional_optimizer_updates=0,additional_training_seed_slots=0,additional_model_inference_by_this_archive=0,
    original_training_resource_guard_receipt_available=False, original_training_guard_history_proven=False,
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
    boundary='Original mirror causal-axis control; source test-selection, pooled test calibration and PA retained. Full local checkpoint/stream reproducibility is certified, not journal equivalence or strict TAB. One new original seed, not two models despite inherited pipeline terminal-state wording.')
closure['inherited_proof_prose_mismatch_disclosure']='Frozen independent-review template prose names source_runtime PSM seed42; actual structured canonical_id, branch, seed1106 and checkpoint/config/result/array digests bind causal_axis_repaired PSM1106. Frozen proof and configurations are preserved unchanged.'
(folder/'one_complete_original_model_and_all_fifteen_streams_closure.json').write_text(json.dumps(closure,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
(folder/'README.md').write_text('Causal-axis PSM seed1106 full3epochs/1554updates/1930289parameters now has successful guarded checkpoint re-inference and independent rereading/recomputation of all15 saved score streams. All13238200 overlapping training score points and87800original test points match bitwise; original41-point dropped tail retained. All three point/native/independent audit guards passed; the original historical training has no persisted resource-usage receipt and its guard history remains unproven. Source PA F1=0.9788686340 versus no-PA F1=0.0191087853 is now independently checked against replayed weights/scores, not just a printed source result. Source test-based model selection and pooled test calibration remain; this is local mirror reproducibility, not journal equivalence or strict TAB. Exactly one selected model is complete; all1140original jobs/six seeds and45/681/273scope remain.\n')
print(json.dumps(dict(complete_models=1,full_bitwise_score_streams=15,independently_recomputed_metrics=proof['independently_recomputed_metrics']),indent=2))
