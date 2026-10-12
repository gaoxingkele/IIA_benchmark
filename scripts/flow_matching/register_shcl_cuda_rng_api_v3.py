"""Refine the measured restoration-stage failure without changing native math."""
import copy
import json

from scripts.flow_matching import audit_shcl_cuda_rng_api_v1 as original
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    native = original.job_api.replay.original.native
    path = 'configs/runtime/fm_shcl_native_CUDA_rng_API.v2.json'
    config = copy.deepcopy(native.read(native.ROOT / path))
    original.verify(config)
    base = native.ROOT / config['state_root']
    receipt = native.read(base / 'receipt.json')
    observations = native.read(base / 'native_operation_memory.json')['observations']
    if (not receipt['resource_guard_aborted'] or receipt['passed']
            or observations[-1]['phase'] != 'before_complete_native_rng_restore'
            or not any(row['phase'] == 'after_native_CUDA_RNG_initialization' for row in observations)):
        raise ValueError('Actual RNG-restoration-stage failure and completed initialization required')
    config['state_root'] = config['state_root'].replace('_v2', '_v3')
    config['output_root'] = config['output_root'].replace('_v2', '_v3')
    paths = [path, (base / 'receipt.json').relative_to(native.ROOT).as_posix(),
        (base / 'native_operation_memory.json').relative_to(native.ROOT).as_posix(),
        'scripts/flow_matching/audit_shcl_cuda_rng_api_v3.py', 'scripts/flow_matching/register_shcl_cuda_rng_api_v3.py',
        'tests/test_fm_shcl_cuda_rng_api_v3.py']
    config['source_receipts'] += [dict(path=p, sha256=native.sha(native.ROOT / p)) for p in paths]
    config['reason_for_new_attempt'] = 'Completed native CUDA RNG initialization was observed at3.10GB, then the restoration helper exceeded the original4GiB cap. Add22 suboperation observations to identify the exact failure boundary, preserving all original statements and29 earlier/current observation sites, limits, native environment and all49 original state obligations.'
    original.verify(config)
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v3.json', config)
    print(json.dumps(dict(original_guards_unchanged=True, new_suboperation_observations=22, prior_failures_preserved=True)))


if __name__ == '__main__':
    main()
