"""Register all original score-audit, full-model and independent review stages."""
import copy
from datetime import datetime,timezone
import json

from scripts.flow_matching.run_pi_source1103_completed_full_replay_v2 import ROOT,read,digest,verify,CANONICAL,RECOVERY,SCOPE
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    native_path='configs/reproducibility/fm_pi_original_source_full_checkpoint_replay.2026-10-11_v1.json'
    point_path='configs/reproducibility/fm_pi_original_source_recovery_completion.2026-10-11_v1.json'
    review_path='configs/reproducibility/fm_pi_original_completed_full_replay_review.2026-10-11_v1.json'
    native,point,review=read(native_path),read(point_path),read(review_path)
    queue_path='configs/experiments/fm_pi_transformer_author_queue.v1.json'
    recovery_path='configs/experiments/fm_pi_source1103_shared_commit_full_retry_queue.v6.json'
    queue,recovery=read(queue_path),read(recovery_path)
    case=next(c for c in recovery['jobs'] if c['job']['id']==RECOVERY)
    job=next(j for j in queue['jobs'] if j['id']==CANONICAL)
    assert job==case['original_job'] and case['original_id']==CANONICAL
    result_path=case['job']['output_directory']+'/result.json';result=read(result_path)
    config=dict(python=review['python'],original_queue=queue_path,recovery_queue=recovery_path,
        required_original_jobs=1140,original_scope=copy.deepcopy(SCOPE),additional_training_seed_slots=0,
        required_original_training_updates=1554,required_full_train_score_points=13238200,
        required_full_test_score_points=87800,required_full_saved_score_streams=15,
        native_template=native_path,review_template=review_path,poll_seconds=5,
        state_root='experiments/runs/fm_pi_source1103_completed_full_replay_v2',
        child_state_root='experiments/runs/fm_pi_source1103_completed_full_replay_children_v2',
        configuration_root='configs/reproducibility/fm_pi_source1103_completed_full_replay_children_v2',
        proof_root='projects/flow_matching_research/results/2026-10-12/pi_source1103_completed_full_replay_v2/completed_models',
        artifact_root='F:/aicoding/IIA_Data/experiments/pi_source1103_completed_full_replay_v2',cases=[],
        boundary='One newly finished original source-runtime PSM seed1103, full3epochs/1554updates, full1930289parameter model, all13238200overlapping training score points and87800original test points with41dropped tails. Full original checkpoint replay and independent all15stream review required. All1140original jobs/six seeds and45paper/681method/273data obligations remain. Source test-selection, pooled test calibration and original PA retained. No new training seed or strictTAB/journal equivalence claim.')
    sources={queue_path,recovery_path,native_path,point_path,review_path,result_path,
        case['job']['output_directory']+'/resource_usage.json',
        'scripts/flow_matching/run_pi_source1103_completed_full_replay_v2.py',
        'scripts/flow_matching/register_pi_source1103_completed_full_replay_v2.py',
        'tests/test_fm_pi_source1103_completed_full_replay_v2.py',
        'scripts/flow_matching/run_pi_causal_completed_pair_replay_v1.py',
        'tests/test_fm_pi_causal_completed_pair_replay_v1.py'}
    for template in (native,point,review):sources.update(pin['path'] for pin in template['source_receipts'])
    sources.update(pin['path'] for pin in job['frozen_files']);sources.update(a['path'] for a in result['artifacts'])
    pins=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    point_config=copy.deepcopy(point)
    point_config.update(recovery_queue=recovery_path,recovery_job_id=RECOVERY,python=review['python'],
        settings=copy.deepcopy(review['settings']),resource_lock=review['resource_lock'],
        output_directory=config['proof_root']+'/'+CANONICAL+'/point',
        state_root=config['child_state_root']+'/'+CANONICAL+'/point',source_receipts=copy.deepcopy(pins),
        boundary=config['boundary'])
    p=config['configuration_root']+'/'+CANONICAL+'_point.json';exclusive_json(ROOT/p,point_config)
    native_config=copy.deepcopy(native)
    native_config.update(recovery_queue=recovery_path,recovery_job_id=RECOVERY,
        independent_completion=point_config['output_directory']+'/independent_full_recovery_completion.json',
        state_root=config['child_state_root']+'/'+CANONICAL+'/native',
        artifact_directory=config['artifact_root']+'/'+CANONICAL+'/native',
        rng_probe_output=point_config['output_directory']+'/original_sampler_rng_probe.json',
        source_receipts=copy.deepcopy(pins),boundary=config['boundary'])
    n=config['configuration_root']+'/'+CANONICAL+'_native_prototype.json';exclusive_json(ROOT/n,native_config)
    config['cases'].append(dict(canonical_id=CANONICAL,point_configuration=p,native_configuration=n))
    sources.update((p,n));config['source_receipts']=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    verify(config)
    path='configs/runtime/fm_pi_source1103_completed_full_replay.v2.json';exclusive_json(ROOT/path,config)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/pi_source1103_completed_full_replay_v2'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=digest(path),source_pins=len(config['source_receipts']),
        canonical_id=CANONICAL,recovery_id=RECOVERY,original_scope=config['original_scope'],required_original_jobs=1140,
        original_required_training_seeds=queue['settings']['seeds'],original_native_result_sha256=digest(result_path),
        required_full_training_updates=1554,required_full_train_score_points=13238200,required_full_test_score_points=87800,
        required_full_saved_score_streams=15,original_native_training_complete=True,
        full_point_score_audit_passed=False,full_native_checkpoint_reinference_passed=False,
        independent_full_stream_review_passed=False,additional_training_seed_slots=0,
        original_native_and_review_guards_cpu_settings_tolerance_unchanged=True,
        new_numeric_results_published=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),
        all_original_jobs=1140,full_training_score_points_required=13238200,full_test_score_points_required=87800)))


if __name__=='__main__':main()
