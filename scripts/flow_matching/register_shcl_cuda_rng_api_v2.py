"""Freeze operation-level diagnosis after the preserved native private-cap abort."""
import copy
import json

from scripts.flow_matching import audit_shcl_cuda_rng_api_v1 as original
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    native = original.job_api.replay.original.native
    path = 'configs/runtime/fm_shcl_native_CUDA_rng_API.v1.json'
    config = copy.deepcopy(native.read(native.ROOT / path))
    original.verify(config)
    prior = native.ROOT / config['state_root'] / 'receipt.json'
    receipt = native.read(prior)
    if not receipt['resource_guard_aborted'] or receipt['passed']:
        raise ValueError('Actual preserved native private-cap failure required before adding observations')
    config['state_root'] = config['state_root'].replace('_v1', '_v2')
    config['output_root'] = config['output_root'].replace('_v1', '_v2')
    paths = [path, prior.relative_to(native.ROOT).as_posix(), 'scripts/flow_matching/audit_shcl_cuda_rng_api_v2.py',
        'scripts/flow_matching/register_shcl_cuda_rng_api_v2.py', 'tests/test_fm_shcl_cuda_rng_api_v2.py']
    config['source_receipts'] += [dict(path=p, sha256=native.sha(native.ROOT / p)) for p in paths]
    config['reason_for_new_attempt'] = 'Identify the exact native operation boundary of the recorded guard abort using seven memory observations. Preserve the failed v1 attempt, all native computations, every original guard and all49 original state obligations. This is not a pass-until-success retry.'
    original.verify(config)
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v2.json', config)
    print(json.dumps(dict(original_guards_unchanged=True, added_native_operation_observations=7, prior_failure_preserved=True)))


if __name__ == '__main__':
    main()
