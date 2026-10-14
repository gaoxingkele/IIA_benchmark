"""Close both frozen full TimesNet audit stages; preserve prior pending snapshots."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit

root=audit.ROOT
previous='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_bitwise_native_complete_point_pending_v1/full_bitwise_native_replay_with_standalone_point_audit_pending.json'
prior=audit.read(root/previous)
assert prior['all_full_train_validation_test_scores_bitwise_equal'] and prior['separately_registered_point_audit_pending']
sources={};proofs={};receipts={}
for mode in ('native','point'):
    config_path='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.'+mode+'.2026-10-14_v1.json'
    config=audit.read(root/config_path);queue,job,result,model=audit.binding(config)
    proof_path=str(Path(config['output_directory'])/'independent_full_saved_score_verification.json')
    receipt_path=config['state_root']+'/receipt.json'
    proof=audit.read(root/proof_path);receipt=audit.read(root/receipt_path)
    assert proof['passed'] and proof['canonical_id']==audit.CANONICAL and proof['mode']==mode
    assert proof['configuration_sha256']==audit.sha(root/config_path)
    assert proof['original_result_sha256']==audit.sha(root/job['output_directory']/'result.json')
    assert proof['original_checkpoint_sha256']==result['checkpoint_sha256']
    assert proof['original_scores_sha256']==result['scores_sha256']
    assert proof['all_original_point_core_event_bootstrap_metrics_exact']
    assert receipt['exit_code']==0 and receipt['full_saved_score_review_passed'] and not receipt['resource_guard_aborted']
    proofs[mode]=proof;receipts[mode]=receipt
    sources.update({mode+'_configuration':root/config_path,mode+'_complete_proof':root/proof_path,
                    mode+'_guarded_receipt':root/receipt_path})
for key in ('full_entities','full_test_points','full_validation_points','full_train_points','coverage',
            'independently_recomputed_primary_metrics','conditional_entity_cluster_f1_ci95',
            'original_event_metrics','original_test_calibrated_TAB_core_diagnostic'):
    assert proofs['point'][key]==proofs['native'][key]
assert proofs['native']['full_native_checkpoint_reinference_passed']
assert proofs['native']['all_replayed_timestamp_scores_bitwise_equal']
assert proofs['native']['replay_scores_sha256']==proofs['native']['original_scores_sha256']
assert audit.sha(Path(audit.read(sources['native_configuration'])['output_directory'])/'replayed_scores.npz')==result['scores_sha256']
base='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_complete_two_stage_replay_v1'
folder=root/base;folder.mkdir(parents=True,exist_ok=False)
for name,path in sources.items():shutil.copyfile(path,folder/(name+'.json'))
closure=deepcopy(prior)
closure.update(captured_utc=datetime.now(timezone.utc).isoformat(),
    separately_registered_point_audit_passed=True,separately_registered_point_audit_pending=False,
    full_registered_two_stage_audit_complete=True,
    actual_point_controller_terminal_state=audit.read(root/audit.read(sources['point_configuration'])['state_root']/'status.json'),
    previous_pending_native_snapshot=previous,previous_pending_native_snapshot_sha256=audit.sha(root/previous),
    current_full_proof_hashes={name:audit.sha(path) for name,path in sources.items()},
    preserved_historical_point_controller_snapshot=closure.pop('actual_point_controller'),
    all_original_experiments_complete=False)
(folder/'complete_two_stage_native_and_independent_saved_score_closure.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf-8')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text(
    'Both original TimesNet SMAP constant-fit seed1103 audit stages are complete. '
    'Standalone full saved-score/metric/raw-data verification and complete native checkpoint '
    're-inference each exited0 without resource abort. All153score arrays and51label arrays '
    '(51entities,101565train/26684validation/408160test points) match the original native '
    'archive bitwise, including compressed archive SHA256. Both stages independently '
    'recomputed exactly the same point/core/event/bootstrap metrics.\n\n'
    'Validation1% noPA: F1=0.0453082580,P=0.2896551724,R=0.0245762547, '
    'AUROC=0.4362539773,AP=0.1213720771. Conditional entity-cluster F1 '
    'CI=[0.0104659048,0.1044334680],200resamples. Original test-calibrated '
    'and test-label-selected diagnostics remain separate. No five-seed aggregate.\n\n'
    'Local configured training:1044nonoverlapping/end-aligned windows,3epochs, '
    '27configured updates derived from frozen code/input geometry. No per-epoch '
    'measured update trace was saved. Local training/window/selection differs from '
    'an unverified historical author harness; this is not author equivalence or '
    'fullTAB training-harness certification. Full291-job range evaluation and all '
    '35baseline/five-seed/45-paper/681-method/273-data obligations remain. '
    'Earlier pending snapshots are preserved unchanged.\n',encoding='utf-8')
print(json.dumps(dict(complete_original_two_stage_audit=True,full_bitwise_score_archive=True,
    full_test_points=408160,all_original_experiments_complete=False)))
