from datetime import datetime,timezone
import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

root,read,sha=cold.ROOT,cold.original.read,cold.streaming_sha
base=root/'projects/flow_matching_research/results/2026-10-14'
folder=base/'brits_electricity20_seed30_closed_full_replay_v1'
folder.mkdir(exist_ok=False)
identity='saits_paper_brits_electricity_author_missing0.2_seed30'
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
assert row['id']==identity and row['full_test_shape']==[290,100,370] and row['held_out_coordinates']==1944331
assert row['full_truth_masks_units_groups_equal'] and row['full_group_statistics_independently_recomputed']
for path,key in [(record['result_path'],'result_sha256'),(record['checkpoint_path'],'checkpoint_sha256'),
    (record['replay_arrays'],'replay_arrays_sha256')]: assert sha(path)==record[key]
result=read(record['result_path']);data=root/result['config']['dataset_root'];output=root/result['config']['output_root']
paths=dict(config=root/record['config'],result=Path(record['result_path']),checkpoint=Path(record['checkpoint_path']),
    arrays=Path(record['replay_arrays']),dataset=data/'datasets.h5',resume=output/'resume.pt',
    original_group_statistics=output/'group_statistics.npz',data_audit=data/'data_audit.json')
assert {key:sha(path) for key,path in paths.items()}==row['hashes']
assert row['full_training_completion']['next_epoch']==1001 and row['full_training_completion']['author_patience_remaining']==0
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
    full_test_grid_cells=10730000,held_out_coordinates=1944331,grouping='10 original months',
    original_maximum_epochs=10000,next_epoch=1001,best_epoch=969,early_stop_patience_exhausted=True,
    independently_recomputed_metrics=row['independently_recomputed_metrics'],
    additional_optimizer_updates=0,additional_model_inference=0,author_equivalence_certified=False,
    original_prediction_bitwise_equivalence_claimed=False,strict_TAB_result=False,
    required_SAITS_BRITS_canonical_slots=210,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False,citation='https://arxiv.org/abs/2202.08516')
(folder/'actual_original_seed30_complete_closure.json').write_text(json.dumps(closure,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
(folder/'README.md').write_text('BRITS / Electricity / 20% missing / seed30 completed the frozen 10000 maximum-epoch, patience30 protocol at next_epoch1001 (best validation epoch969). All290x100x370 cells,1944331 held-out coordinates and10 original month groups passed native CPU model replay and independent review at the original2e-6 tolerance. Independent MAE0.9313053601, RMSE1.3734326551, MRE0.4983255148. All current input/model/result/proof hashes were rechecked. Original predictions were unavailable, so no bitwise prediction claim. Historical author preprocessing/settings remain uncertain; this is not strict TAB or completion of all210 tasks.\n')

for name,text in [
    ('cfm_actual145_main_GP400_closure_v1','Three main-text GP Gaussian conditional400 models (seeds1103,1104,1105) completed400 epochs/62400 updates each and all20x500x1 original held-out coordinates. Native model replays were bitwise identical; independent full-array proofs passed. Deduplicated snapshot145/285 includes two supplemental BB executions only once. Main protocol differs from Appendix80 trajectories/250000 updates. Original author equivalence and TAB performance remain unproved.\n'),
    ('cfm_actual146_main_GP400_seed1107_closure_v1','Main-text GP Gaussian conditional400 seed1107 completed400 epochs/62400 updates and all20x500x1 original held-out coordinates. MSE0.006120708138789589, MAE0.05950776648944914, RMSE0.07823495471200574. Native replay was bitwise identical and independent every-coordinate review passed. Deduplicated full snapshot146/285; the second unchanged-guard recurring window released naturally. Seed1106 has a preserved progress-write metadata failure after full training, so this report does not claim a complete five-seed main cohort. The full45/681/273 research scope remains.\n'),
    ('cfm_saved_GP4001106_score_recovery_v1','Observed main GP400 seed1106 failed solely on progress.json publication at6/20 scoring trajectories. Original400-epoch/62400-update model.pt is preserved and SHA256 pinned. A separate execution restores bitwise identical weights and reruns all20x500 grids through the unchanged original scoring body, without fitting, optimizer updates or target preprocessing. Metadata publication alone uses bounded UUID temporary-file retries. Native30GiB commit/16GiB physical admission,16GiB emergency and2GiB RSS/4GiB private caps remain unchanged; same family lock queues on natural source releases. The replay dispatcher retains all285 routes and changes only this execution.26 tests passed. Launch receipts prove real controllers; scoring/replay completion must still be demonstrated.\n')]:
    (base/name/'README.md').write_text(text)

launchpath=base/'cfm_saved_GP4001106_score_recovery_v1/actual_guarded_controller_launch.json'
launch=read(launchpath);handles=[]
for saved in launch['actual_handles']:
    p=psutil.Process(saved['pid']);assert p.create_time()==saved['create_time'] and p.cmdline()==saved['command']
    statepath=root/('experiments/runs/fm_cfm_saved_score_recovery_v1/status.json' if 'saved_full_score_recovery' in saved['module'] else 'experiments/runs/fm_cfm_saved_score_full_replay_v1/status.json')
    handles.append(dict(**saved,private_bytes=p.memory_info().private,state=read(statepath)))
(launchpath.parent/'actual_live_postlaunch_wait_snapshot.json').write_text(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles),indent=2)+'\n')
print(json.dumps(dict(brits_seed30_complete=True,CFM_closed=146,new_live_saved_model_scoring_and_review=len(handles))))
