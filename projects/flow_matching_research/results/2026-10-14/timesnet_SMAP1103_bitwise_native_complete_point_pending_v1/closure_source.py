"""Archive passed full native scores; preserve the pending standalone point audit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit

root=audit.ROOT
native_path='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.native.2026-10-14_v1.json'
point_path='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.point.2026-10-14_v1.json'
config=audit.read(root/native_path)
queue,job,result,model=audit.binding(config)
proof_path=str(Path(config['output_directory'])/'independent_full_saved_score_verification.json')
proof=audit.read(proof_path)
receipt_path=config['state_root']+'/receipt.json'
receipt=audit.read(root/receipt_path)
assert proof['passed'] and proof['canonical_id']==audit.CANONICAL and proof['mode']=='native'
assert proof['configuration_sha256']==audit.sha(root/native_path)
assert proof['full_native_checkpoint_reinference_passed'] and proof['all_replayed_timestamp_scores_bitwise_equal']
assert proof['all_original_point_core_event_bootstrap_metrics_exact']
assert proof['original_scope']==audit.SCOPE and proof['original_queue_jobs']==35
assert proof['full_entities']==51 and proof['full_test_points']==408160
assert proof['full_validation_points']==26684 and proof['full_train_points']==101565
assert proof['original_scores_sha256']==proof['replay_scores_sha256']==result['scores_sha256']
assert audit.sha(Path(config['output_directory'])/'replayed_scores.npz')==result['scores_sha256']
assert proof['original_checkpoint_sha256']==result['checkpoint_sha256']
assert proof['original_result_sha256']==audit.sha(root/job['output_directory']/'result.json')
assert receipt['exit_code']==0 and receipt['full_saved_score_review_passed'] and not receipt['resource_guard_aborted']
coverage=proof['coverage']
assert len(coverage)==153 and len({(r['entity'],r['split']) for r in coverage})==153
assert {r['split'] for r in coverage}=={'train','validation','test'}
training_windows=sum(r['windows'] for r in coverage if r['split']=='train')
configured_steps_per_epoch=(training_windows+127)//128
point=audit.read(root/point_path)
point_state=audit.read(root/point['state_root']/'status.json')
p=psutil.Process(point_state['pid'])
assert p.create_time()==point_state['create_time']
assert not (root/point['state_root']/'receipt.json').exists()
assert point_state['active_pid'] is None and point_state['state'] in ('waiting_for_memory','waiting_for_resource_lock')
xml=root/'tmp/TimesNet_SMAP1103_native_completed_validation_20261014_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==37
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
base='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_bitwise_native_complete_point_pending_v1'
folder=root/base;folder.mkdir(parents=True,exist_ok=False)
for path,name in [(root/native_path,'native_configuration.json'),(Path(proof_path),'complete_native_and_metric_verification.json'),
                  (root/receipt_path,'successful_native_guarded_receipt.json'),(xml,'validation.xml')]:
    shutil.copyfile(path,folder/name)
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=audit.CANONICAL,
    full_original_local_training_and_full_native_score_replay_passed=True,
    all_full_train_validation_test_scores_bitwise_equal=True,original_and_replay_score_archives_same_sha256=True,
    full_entities=51,full_training_points=101565,full_validation_points=26684,full_test_points=408160,
    complete_score_arrays=153,complete_label_arrays=51,parameters=28132633,
    configured_epochs=3,full_registered_training_windows=training_windows,
    configured_optimizer_steps_per_epoch_from_original_code=configured_steps_per_epoch,
    configured_optimizer_steps_all_epochs_from_original_code=3*configured_steps_per_epoch,
    actual_per_epoch_optimizer_update_trace_available=False,
    training_window_construction='Original local nonoverlapping windows with one end-aligned tail, within each original entity; no training windows or epochs changed by replay.',
    independently_recomputed_primary_metrics=proof['independently_recomputed_primary_metrics'],
    conditional_entity_cluster_f1_ci95=proof['conditional_entity_cluster_f1_ci95'],
    original_event_metrics=proof['original_event_metrics'],
    original_test_calibrated_TAB_core_diagnostic=proof['original_test_calibrated_TAB_core_diagnostic'],
    original_checkpoint_sha256=result['checkpoint_sha256'],original_and_replay_scores_sha256=result['scores_sha256'],
    full_saved_metric_recomputation_inside_native_worker_passed=True,
    separately_registered_point_audit_passed=False,separately_registered_point_audit_pending=True,
    actual_point_controller=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),state=point_state),
    full_registered_two_stage_audit_complete=False,full_TAB_range_metrics_completed=False,
    original_scope=audit.SCOPE,full_original_baseline_queue_jobs=35,full_corrected_data_controls=291,
    required_original_cohort_seeds=[1103,1104,1105,1106,1107],complete_five_seed_aggregate=False,
    additional_optimizer_updates=0,additional_training_seed_slots=0,validation_tests_passed=37,
    historical_author_equivalence_certified=False,full_TAB_training_harness_certified=False,
    all_original_experiments_complete=False,
    source_receipts=[dict(path=path,sha256=audit.sha(root/path)) for path in [native_path,point_path,
        proof_path,receipt_path,job['output_directory']+'/result.json',job['model_config'],job['dataset_manifest_path']]])
(folder/'full_bitwise_native_replay_with_standalone_point_audit_pending.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf-8')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text(
    'Complete original local TimesNet SMAP constant-fit seed1103 checkpoint inference passed. '
    'All153train/validation/test score arrays and51full label arrays across51entities '
    '(101,565train/26,684validation/408,160test points) match bitwise. The complete compressed '
    'score archive SHA256 is also identical. The original28,132,633parameters,float32,CPU2 '
    'andbatch128 were retained; source/raw/config/artifact hashes were rechecked.\n\n'
    'Independent source-metric recomputation inside the native worker passed: validation1% '
    'noPA F1=0.0453082580,P=0.2896551724,R=0.0245762547,AUROC=0.4362539773,AP=0.1213720771. '
    'Conditional entity-cluster F1 CI=[0.0104659048,0.1044334680],200resamples. '
    'Source test-calibrated/test-label-selected grid maxima: plainF1=0.1451872366, '
    'PAF1=0.7774734352; these remain separate oracle diagnostics.\n\n'
    f'The complete original local training uses{training_windows}nonoverlapping/end-aligned '
    f'windows, configured{configured_steps_per_epoch}optimizer steps per epoch and3epochs '
    f'({3*configured_steps_per_epoch}configured steps total). These are derived from the '
    'frozen executed code and complete input geometry, not an independent measured update '
    'trace. No actual per-epoch update trace was saved. This local training/window/selection '
    'protocol does not certify a historical author pipeline or fullTAB-training harness.\n\n'
    '37 relevant checks passed and the actual native guard exited0 without abort. '
    'The separately registered standalone point audit is still alive waiting for original '
    'resource admission; its receipt is not complete. Both original stage obligations '
    'and the pre-existing full291-job TAB range collector remain. All35baseline slots, '
    'five seeds and45-paper/681-method/273-data scope remain required.\n',encoding='utf-8')
print(json.dumps(dict(full_bitwise_native_model_replay_passed=True,standalone_point_still_pending=True,
    full_training_windows=training_windows,configured_optimizer_steps=3*configured_steps_per_epoch,
    full_test_points=408160,validation_tests_passed=37)))
