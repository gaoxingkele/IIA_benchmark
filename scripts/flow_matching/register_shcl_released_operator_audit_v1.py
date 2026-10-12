"""Freeze full released SHCL operator shapes, tolerance and actual host packages."""
import json

from scripts.flow_matching.audit_shcl_released_operators_v1 import native, profiles, environment
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    model_path = 'configs/models/fm_shcl_released_vae_native.v1.json'
    model = native.read(native.ROOT / model_path); native.verify(model)
    reference = 'configs/runtime/fm_timegan_energy_stage_restore_audit.v1.json'
    resource = native.read(native.ROOT / reference)
    files = [model_path, reference, 'configs/experiments/fm_shcl_released_script_registry.v1.json',
        'scripts/flow_matching/audit_shcl_released_operators_v1.py',
        'scripts/flow_matching/run_shcl_released_operator_audit_v1.py',
        'scripts/flow_matching/register_shcl_released_operator_audit_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/timegan_native_common_v1.py',
        'scripts/flow_matching/run_timegan_full45_completion_replay_v1.py']
    config = dict(model_config=model_path, reference_resource_runtime=reference,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        environment=environment(), python=environment()['executable'], profiles=profiles(model), cpu_threads=2,
        settings=resource['settings'], resource_lock=resource['resource_lock'],
        independent_loss_tolerance=dict(atol=2e-5, rtol=2e-5),
        state_root='experiments/runs/fm_shcl_released_operator_audit_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/shcl_released_native_preparation_v1/operators',
        source_receipts=[dict(path=p, sha256=native.sha(native.ROOT / p)) for p in files],
        boundary='Actual pinned host numerical packages for synthetic full-shape operator audit only. '
            'Original author package/data equivalence and all49train/364test commands remain pending. '
            'No optimizer update or benchmark score is produced; tolerance frozen before execution.')
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_released_operator_audit.v1.json', config)
    print(json.dumps(dict(full_profiles=len(config['profiles']), batch_size=64, environment=config['environment'])))


if __name__ == '__main__':
    main()
