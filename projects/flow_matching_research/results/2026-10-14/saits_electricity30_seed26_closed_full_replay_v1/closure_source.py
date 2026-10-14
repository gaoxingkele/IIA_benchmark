from datetime import datetime,timezone
import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root,read,sha=cold.ROOT,cold.original.read,cold.streaming_sha
base=root/'projects/flow_matching_research/results/2026-10-14'
folder=base/'saits_electricity30_seed26_closed_full_replay_v1'
folder.mkdir(exist_ok=False)
identity='saits_paper_saits_electricity_author_missing0.3_seed26'
nativepath=root/'configs/reproducibility/fm_saits_completion_replay_queue_v1'/f'{identity}.native.json'
reviewpath=nativepath.with_name(f'{identity}.review.json')
native,review=read(nativepath),read(reviewpath)
reportpath=root/native['artifact_directory']/'checkpoint_replay_audit.json'
proofpath=root/review['output_directory']/'independent_full_replay_batch_verification.json'
report,proof=read(reportpath),read(proofpath)
record=report['records'][0];row=proof['records'][0]
assert report['full_models_replayed']==proof['full_native_models_verified']==1 and proof['passed']
assert report['configuration_sha256']==sha(nativepath) and proof['configuration_sha256']==sha(reviewpath)
assert proof['native_replay_report_sha256']==sha(reportpath)
assert row['id']==identity and row['full_test_shape']==[290,100,370] and row['held_out_coordinates']==2781282
assert row['full_truth_masks_units_groups_equal'] and row['full_group_statistics_independently_recomputed']
for path,key in [(record['result_path'],'result_sha256'),(record['checkpoint_path'],'checkpoint_sha256'),
    (record['replay_arrays'],'replay_arrays_sha256')]: assert sha(path)==record[key]
result=read(record['result_path']);data=root/result['config']['dataset_root'];output=root/result['config']['output_root']
paths=dict(config=root/record['config'],result=Path(record['result_path']),checkpoint=Path(record['checkpoint_path']),
    arrays=Path(record['replay_arrays']),dataset=data/'datasets.h5',resume=output/'resume.pt',
    original_group_statistics=output/'group_statistics.npz',data_audit=data/'data_audit.json')
assert {key:sha(path) for key,path in paths.items()}==row['hashes']
assert row['full_training_completion']['next_epoch']==201 and row['full_training_completion']['author_patience_remaining']==0
sources=dict(native_config=nativepath,independent_config=reviewpath,original_result=Path(record['result_path']),
    native_report=reportpath,independent_proof=proofpath,
    native_guarded_receipt=root/native['state_root']/'receipt.json',
    independent_guarded_receipt=root/review['state_root']/'receipt.json')
for name,path in sources.items(): (folder/(name+'.json')).write_bytes(path.read_bytes())
nr,rr=read(sources['native_guarded_receipt']),read(sources['independent_guarded_receipt'])
assert nr['exit_code']==rr['exit_code']==0 and not nr['resource_guard_aborted'] and not rr['resource_guard_aborted']
assert nr['full_native_model_replay_passed'] and rr['full_native_batch_independent_review_passed']
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),full_original_model_and_early_stop_verified=True,
    checkpoint_and_full_data_and_original_masks_and_current_artifact_SHA256_rechecked=True,
    full_test_grid_cells=10730000,held_out_coordinates=2781282,grouping='10 original months',
    original_maximum_epochs=10000,next_epoch=201,best_epoch=169,early_stop_patience_exhausted=True,
    independently_recomputed_metrics=row['independently_recomputed_metrics'],
    additional_optimizer_updates=0,additional_model_inference=0,author_equivalence_certified=False,
    original_prediction_bitwise_equivalence_claimed=False,strict_TAB_result=False,
    required_SAITS_BRITS_canonical_slots=210,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False,citation='https://arxiv.org/abs/2202.08516')
(folder/'actual_original_saits_electricity30_seed26_complete_closure.json').write_text(json.dumps(closure,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
