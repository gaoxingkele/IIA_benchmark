from datetime import datetime, timezone
import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_flow_mismatching_full_image_evaluation_queue_v1 as image
from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as cfm

root=image.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/verified_image_epoch_cases_and_cfm92_v1'
folder.mkdir(parents=True,exist_ok=False)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
runtime=image.read(root/'configs/runtime/fm_flow_mismatching_full_image_evaluation_queue.v1.json')
queue=image.read(root/runtime['queue']);image.validate_queue(queue)
image.verify_receipts(runtime['source_receipts']);image.verify_receipts(queue['source_receipts'])
cases=[c for c in queue['cases'] if c['id'].startswith('table9_epoch_checkpoints__')
       and c['dataset']=='mvtec' and c['seed']==1103 and c['checkpoint_epoch'] in [100,200,300,400,500]]
assert len(cases)==5
rows=[]
for case in cases:
    assert image.complete(case)
    location_path=root/case['output_directory']/'fully_verified_score_result.json'
    location=image.read(location_path);reportpath=Path(location['report']);report=image.read(reportpath)
    candidatepath=Path(report['native_score_candidate']);candidate=image.read(candidatepath)
    terminalpath=reportpath.parent/'terminal.json';terminal=image.read(terminalpath)
    assert terminal['returncode']==0 and not terminal['resource_usage']['resource_guard_aborted']
    assert report['complete_original_test_images']==1725 and report['complete_original_categories']==15
    assert report['max_absolute_heatmap_replay_error']==report['max_absolute_image_score_replay_error']==0.
    assert report['original_author_equivalence_certified'] is False and report['strict_TAB_result'] is False
    target=folder/'original_closed_image_evidence'/case['id'];target.mkdir(parents=True)
    for source,name in [(reportpath,'independent_full_native_image_evaluation.json'),
        (terminalpath,'native_guarded_terminal.json'),(candidatepath,'native_score_candidate.json'),
        (location_path,'fully_verified_score_location.json')]:
        (target/name).write_bytes(source.read_bytes())
    rows.append(dict(case_id=case['id'],seed=1103,checkpoint_epoch=case['checkpoint_epoch'],
        complete_original_test_images=1725,complete_original_categories=15,
        metrics_percent=report['category_macro_percent'],paper_comparison=report['original_paper_comparison'],
        native_report_sha256=image.sha(reportpath),native_guarded_terminal_sha256=image.sha(terminalpath),
        native_candidate_sha256=image.sha(candidatepath),full_checkpoint_sha256=report['full_original_checkpoint_sha256'],
        all_native_models_heatmaps_probe_seeds_and_scores_replayed=True,
        independent_all_category_all_pixel_metric_verification_passed=True,
        max_absolute_native_heatmap_difference=0.,max_absolute_native_score_difference=0.,
        original_author_equivalence_certified=False,strict_TAB_result=False))
rows.sort(key=lambda r:r['checkpoint_epoch'])
write(folder/'verified_MVTec_table9_epoch_cases_and_paper_comparison.json',dict(
    citation='https://arxiv.org/abs/2605.23070',original_evaluation_queue=runtime['queue'],
    original_required_case_seed_evaluations=325,verified_selected_cases=len(rows),records=rows,
    metrics_unit='percentage; paper difference is percentage points; unweighted15-category macro mean',
    required_final_model_training_epochs=2000,these_cases_prove_final2000_epoch_completion=False,
    uncertainty_boundary='Five checkpoints of one training seed, not five independently trained seeds; no between-seed uncertainty is certified.',
    F1_max_boundary='Complete test-label threshold sweep; oracle evaluation only, never strict TAB calibration.',
    local_unreported_settings=report['local_unreported_settings'],
    causal_performance_gap_mechanism_certified=False,original_author_equivalence_certified=False,
    strict_TAB_result=False,all_original_experiments_complete=False))
cfmruntime=cfm.read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
records=cfm.read(root/cfmruntime['state_root']/'verified_full_model_records.json')['records']
old=image.read(root/'projects/flow_matching_research/results/2026-10-14/timegan_sine_collector_recovery_and_cfm_gp_complete_v1/actual_full285_cfm_replay_observation.json')
oldids=set(old['verified_original_ids']);new=[r for r in records if r['original_canonical_id'] not in oldids]
assert len(records)>=92 and len(oldids)==89
routes={r['original_canonical_id']:r for r in cfmruntime['routes']}
newverified=[]
for record in new:
    identity=record['original_canonical_id'];route=routes[identity]
    configpath=root/cfmruntime['child_config_root']/(identity+'.native.json')
    nativeconfig=cfm.read(configpath)
    nativepath=root/nativeconfig['output_directory']/'checkpoint_replay_audit.json'
    nativereceiptpath=root/nativeconfig['state_root']/'receipt.json'
    nativereport,nativereceipt=cfm.read(nativepath),cfm.read(nativereceiptpath)
    cfm.check_native_terminal(nativeconfig,nativereport,nativereceipt)
    canonical=dict(route,result_sha256=cfm.sha(root/route['result_path']))
    verified=cfm.verify_record(nativeconfig,canonical,record)
    reviewconfigpath=root/cfmruntime['child_config_root']/(identity+'.review.json')
    reviewconfig=cfm.read(reviewconfigpath)
    reviewpath=root/reviewconfig['output_proof'];review=cfm.read(reviewpath)
    reviewreceiptpath=root/reviewconfig['state_root']/'receipt.json';reviewreceipt=cfm.read(reviewreceiptpath)
    assert review['passed'] and reviewreceipt['exit_code']==0
    assert reviewreceipt['independent_full_array_review_passed'] and not reviewreceipt['resource_guard_aborted']
    assert review['configuration_sha256']==cfm.sha(reviewconfigpath)
    assert len(review['records'])==1 and review['records'][0]==record
    target=folder/'original_closed_CFM_evidence'/identity;target.mkdir(parents=True)
    for source,name in [(configpath,'native_configuration.json'),(nativepath,'checkpoint_replay_audit.json'),
        (nativereceiptpath,'native_guarded_receipt.json'),(reviewconfigpath,'independent_review_configuration.json'),
        (reviewpath,'independent_full_array_review.json'),(reviewreceiptpath,'independent_guarded_receipt.json')]:
        (target/name).write_bytes(Path(source).read_bytes())
    newverified.append(verified)
write(folder/'new_complete_CFM_native_and_independent_certificates.json',dict(
    required_original_models=285,previously_published_verified_ids=sorted(oldids),
    actual_verified_original_models=len(records),pending_full_certification_models=285-len(records),
    newly_checked_complete_native_and_independent_records=newverified,
    every_new_original_grid_coordinate_and_metric_verified=True,
    additional_training_seed_slots=0,author_equivalence_certified=False,all_original_experiments_complete=False))
locks=image.read(root/'tmp/actual_shared_resource_lock_owners_20261014.json')
for old in locks:
    process=psutil.Process(old['pid'])
    assert process.create_time()==old['create_time'] and process.cmdline()==old['command']
write(folder/'verified_actual_shared_lock_owners.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    lock_owners=locks,actual_training_or_native_replay_children_present=True,
    idle_lock_deadlock_certified=False,workers_stopped=False))
write(folder/'iteration_actual_verified_result_evidence.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    complete_image_checkpoint_cases_verified=5,required_image_case_seed_evaluations=325,
    image_cases_native_images_each=1725,image_cases_native_categories_each=15,
    new_CFM_full_native_and_independent_certificates=len(newverified),total_CFM_full_certificates=len(records),
    required_CFM_original_models=285,native_training_or_replay_workers_stopped=False,
    lower_or_different_paper_effects_preserved=True,author_equivalence_certified=False,
    strict_TAB_result=False,all_original_experiments_complete=False))
print(json.dumps(dict(image_cases=len(rows),CFM_full_certificates=len(records),new_CFM_certificates=len(newverified))))
