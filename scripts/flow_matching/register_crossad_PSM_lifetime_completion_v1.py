"""Register the full fixed-weight PSM replay after its proven RSS failure."""
from copy import deepcopy
from datetime import datetime,timezone
import json

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full

NATIVE='configs/reproducibility/fm_crossad_PSM_residual_lifetime_native_replay.2026-10-12_v4.json'
POST='configs/reproducibility/fm_crossad_PSM_residual_lifetime_independent_review.2026-10-12_v4.json'
CONFIG='configs/runtime/fm_crossad_PSM_residual_lifetime_completion.v1.json'
RESULT='projects/flow_matching_research/results/2026-10-12/crossad_PSM_residual_lifetime_replay_v4'


def register():
    import psutil
    from scripts.flow_matching.run_crossad_PSM_lifetime_completion_v1 import verify
    old_path='configs/reproducibility/fm_crossad_recovery_completion_children_v1/crossad_author__release_checkpoint__psm__seed2025__resource_recovery_v1_native.json'
    old=full.read(old_path);failed_path=old['state_root']+'/receipt.json';failed=full.read(failed_path)
    if failed['exit_code']==0 or failed['full_native_model_replay_passed'] or not failed['resource_guard_aborted']:
        raise ValueError('Require the actual failed full PSM replay, not an observation timeout')
    if failed['peak_tree_rss_bytes']<=old['settings']['maximum_worker_rss_bytes']:
        raise ValueError('This lifetime-only change is for the actual native RSS-limit failure')
    point_terminal='experiments/runs/fm_crossad_recovery_completion_children_v1/crossad_author__release_checkpoint__psm__seed2025__resource_recovery_v1/point/receipt.json'
    diagnostics=full.read('experiments/runs/fm_crossad_residual_lifetime_native_diagnostics_v1/status.json')
    diagnostic_terminal='experiments/runs/fm_crossad_residual_lifetime_native_diagnostics_v1/receipt.json'
    if (full.ROOT/diagnostic_terminal).exists():
        tested=full.read(diagnostic_terminal)
        if tested['exit_code']!=0 or not tested['passed'] or tested['resource_guard_aborted']:
            raise ValueError('Failed actual native diagnostics may not admit a full source replay')
        diagnostic_owner=dict(pid=diagnostics['pid'],create_time=diagnostics['create_time'],command=diagnostics['command'])
    else:
        actor=psutil.Process(diagnostics['pid'])
        if actor.create_time()!=diagnostics['create_time'] or actor.cmdline()!=diagnostics['command']:
            raise ValueError('Keep the exact original guarded diagnostic owner')
        diagnostic_owner=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline())
    for path in (NATIVE,POST,CONFIG):
        if (full.ROOT/path).exists():raise FileExistsError('Preserve every existing registered attempt')
    sources={p['path'] for p in old['source_receipts']}
    sources.update([old_path,failed_path,old['independent_point_completion'],point_terminal,
        'src/iia_benchmark/models/fm_crossad_reference_lifetime_v1.py',
        'scripts/flow_matching/replay_crossad_residual_lifetime_v4.py',
        'scripts/flow_matching/run_crossad_PSM_lifetime_completion_v1.py',
        'scripts/flow_matching/register_crossad_PSM_lifetime_completion_v1.py',
        'tests/test_fm_crossad_residual_lifetime_source_v1.py',
        'tests/test_fm_crossad_residual_lifetime_native_v1.py','scripts/flow_matching/cached_resource_snapshot_v1.py'])
    if (full.ROOT/diagnostic_terminal).exists():sources.add(diagnostic_terminal)
    pins=lambda paths:[dict(path=p,sha256=full.digest(p)) for p in sorted(paths)]
    native=deepcopy(old)
    native.update(artifact_directory='F:/aicoding/IIA_Data/experiments/crossad_PSM_residual_lifetime_replay_v4/native',
        state_root='experiments/runs/fm_crossad_PSM_residual_lifetime_replay_v4/native',source_receipts=pins(sources))
    native['boundary']+=' Release only encoder/decoder x_ references after their final residual use. Full128 batches,1,328,459 parameters, all arithmetic/dropout/normalization, scores and original author/SPOT/range/VUS evaluators unchanged. Original RSS3/private4GiB and all memory/commit guards unchanged. Numeric diagnostic success is required before dispatch; no complete result claimed at registration.'
    full.exclusive_configuration(NATIVE,native)
    post={k:deepcopy(native[k]) for k in (*full.GUARDS,'original_scope')}
    post.update(phase='post',native_configuration=NATIVE,
        state_root='experiments/runs/fm_crossad_PSM_residual_lifetime_replay_v4/post',
        output_directory=RESULT+'/independent',source_receipts=pins(sources|{NATIVE}))
    full.exclusive_configuration(POST,post)
    recovery=full.read('configs/runtime/fm_crossad_recovery_completion_handoff.v1.json')
    canonical='crossad_author__release_checkpoint__psm__seed2025'
    route=next(r for r in recovery['recovery_routes'] if r['canonical_id']==canonical)
    config=dict(native_configuration=NATIVE,post_configuration=POST,
        state_root='experiments/runs/fm_crossad_PSM_residual_lifetime_completion_v1',
        original_completion_configuration='configs/runtime/fm_crossad_completion_replay_queue.v1.json',
        preserved_recovery_route=route,canonical_id=canonical,point_terminal=point_terminal,
        diagnostic_terminal=diagnostic_terminal,
        diagnostic_owner=diagnostic_owner,
        required_original_jobs=139,original_scope=full.SCOPE,score_tolerance=full.SCORE_TOLERANCE,
        metric_tolerance=full.METRIC_TOLERANCE,poll_seconds=10,
        source_receipts=pins(sources|{NATIVE,POST,'configs/runtime/fm_crossad_completion_replay_queue.v1.json',
            'configs/runtime/fm_crossad_recovery_completion_handoff.v1.json'}))
    verify(config);full.exclusive_configuration(CONFIG,config)
    full.exclusive_configuration(RESULT+'/actual_same_full_PSM_replay_registration.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        full_native_configuration=NATIVE,full_independent_configuration=POST,original_scope=full.SCOPE,
        required_original_jobs=139,unchanged_other_original_job_recipes=138,
        full_model_parameters=1328459,original_batch_size=128,original_seq_len=192,
        all_raw_points=dict(train=105984,validation=26497,test=87841),
        original_nonoverlap_test_points=87744,original_unscored_tail_points=97,
        original_failed_replay=failed,failed_original_artifacts_and_source_preserved=True,
        independent_original_full_point_certificate_reused=True,
        native_original_arithmetic_and_batch_partition_unchanged=True,
        actual_numeric_diagnostic_and_full_replay_passed=False,
        original_RSS_private_and_commit_guards_unchanged=True,
        no_scientific_training_worker_changed_or_stopped=True,additional_training_seed_slots=0,
        fixed_release_evaluation_is_not_a_new_training_seed=True,all_original_experiments_complete=False))
    return config


if __name__=='__main__':print(register()['state_root'])
