"""Freeze all413 read-only completion obligations; fixture never satisfies one."""
import json

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    queue_path='configs/experiments/fm_shcl_native_script_queue.v1.json'; queue=native.read(native.ROOT/queue_path)
    fixture_path='configs/runtime/fm_shcl_independent_replay_fixture.v1.json'; fixture=native.read(native.ROOT/fixture_path)
    report_path=native.ROOT/fixture['output_root']/'full_observer_fidelity_fixture.json'; report=native.read(report_path)
    if (not report['passed'] or not report['synthetic_operator_only_not_benchmark_performance']
            or report['original_benchmark_optimizer_updates']!=0 or len(report['full_native_frames_replayed'])!=2):
        raise ValueError('Full independent diagnostic replay required, never original-source completion credit')
    reference_path='configs/runtime/fm_timegan_energy_stage_restore_audit.v1.json'; reference=native.read(native.ROOT/reference_path)
    paths=[queue_path,fixture_path,report_path.relative_to(native.ROOT).as_posix(),reference_path,
        'scripts/flow_matching/replay_shcl_native_frames_v1.py','scripts/flow_matching/run_shcl_native_replay_job_v1.py',
        'scripts/flow_matching/register_shcl_full413_replay_v1.py','scripts/flow_matching/run_shcl_full413_completion_v1.py',
        'scripts/flow_matching/run_shcl_native_job_v1.py','scripts/flow_matching/run_shcl_native_script_queue_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py','scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    config=dict(original_queue=queue_path,canonical_job_ids=[j['id'] for j in queue['jobs']],
        original_scope=queue['original_scope'],full_training_commands=49,full_test_commands=364,
        settings=reference['settings'],resource_lock=reference['resource_lock'],
        gpu_settings={k:queue['settings'][k] for k in ('gpu_uuid','minimum_gpu_free_bytes','emergency_gpu_free_bytes')},
        python=queue['python'],environment=queue['environment'],
        state_root='experiments/runs/fm_shcl_full413_completion_v1',
        output_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/independent_full413_replays_v1',
        source_receipts=[dict(path=p,sha256=native.sha(native.ROOT/p)) for p in paths],
        every_original_epoch_model_and_Adam_restore_required=True,every_original_native_score_frame_required=True,
        bitwise_loss_score_label_and_RNG_gate=True,original_native_point_adjustment_code_unchanged=True,
        source_test_prefix_not_strict_full_point_TAB=True,
        boundary='All49 actual original trainings plus all364 original tests remain completion obligations. '
            'Verify every saved epoch model/Adam variable and final native weights, and every complete '
            'saved stochastic test/threshold loss/score/label/RNG frame with exact original scalar/PA '
            'metrics. No extra optimizer step or shortened grid. Independent source replay does not '
            'certify unreleased paper backbones or the separate strict grouped full-point TAB protocol.')
    exclusive(native.ROOT/'configs/runtime/fm_shcl_full413_completion_replay.v1.json',config)
    print(json.dumps(dict(full_training_replays=49,full_test_replays=364,bitwise_gate=True,fixture_slots_credited=0)))


if __name__=='__main__':main()
