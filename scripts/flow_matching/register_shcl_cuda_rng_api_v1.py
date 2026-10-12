"""Use original native replay guards for an actual CUDA API diagnostic."""
import json

from scripts.flow_matching import run_shcl_training_rng_job_v1 as job
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    native = job.replay.original.native
    path = 'configs/runtime/fm_shcl_all49_training_rng_replay.v2.json'
    previous = native.read(native.ROOT / path)
    job.verify(previous)
    paths = [path, 'scripts/flow_matching/audit_shcl_cuda_rng_api_v1.py',
        'scripts/flow_matching/register_shcl_cuda_rng_api_v1.py', 'tests/test_fm_shcl_cuda_rng_api_v1.py',
        'scripts/flow_matching/shcl_training_rng_replay_v1.py', 'scripts/flow_matching/replay_shcl_native_frames_v1.py',
        'scripts/flow_matching/shcl_native_observer_v1.py', 'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py', 'scripts/flow_matching/run_shcl_native_script_queue_v1.py']
    config = {key: previous[key] for key in ('settings', 'gpu_settings', 'resource_lock', 'python', 'environment', 'original_scope')}
    config.update(full49_reference_configuration=path, diagnostic_only=True, canonical_training_jobs_credited=0,
        state_root='experiments/runs/fm_shcl_native_CUDA_rng_API_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/shcl_native_CUDA_rng_API_v1',
        source_receipts=[dict(path=p, sha256=native.sha(native.ROOT / p)) for p in paths],
        boundary='Actual native Torch1.11 CUDA stream restoration and all six backend flag restoration API diagnostic. '
            'Use a newly captured API fixture, not a claimed original CUDA training checkpoint. CPU and all CUDA draws '
            'must match independent native generators bitwise; invalid device count must fail. No optimizer updates, '
            'model training, source checkpoint replacement, tolerance change or canonical credit. All49 original '
            'saved-state checkpoints, full413 source commands and strict grouped TAB remain required.')
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v1.json', config)
    print(json.dumps(dict(original_full_training_RNG_jobs_retained=49, canonical_credit=0, original_readonly_guards_unchanged=True)))


if __name__ == '__main__':
    main()
