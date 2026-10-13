"""Bounded host adoption for all8 retry routes inside unchanged full285 scope.

Adopt only already closed native/full-array proofs, hashing every artifact byte.
No numerical replay or array review is replaced. Native workers retain the
original full-grid implementation, environment, precision and resource guards.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
import ast
from types import FunctionType
import sys

from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as supplement
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT = supplement.ROOT


def adopt_primary(runtime, verified):
    primary = supplement.original.read(ROOT / supplement.PREVIOUS)
    path = ROOT / primary['state_root'] / 'verified_full_model_records.json'
    if not path.exists():
        return
    snapshot = supplement.original.read(path)
    records = snapshot['records']
    if len({r['original_canonical_id'] for r in records}) != len(records):
        raise ValueError('Unique closed primary records required')
    for record in records:
        identity = record['original_canonical_id']
        if identity in verified:
            if verified[identity] != record:
                raise ValueError('Conflicting immutable primary certificate')
            continue
        if identity in runtime['selected_canonical_ids']:
            raise ValueError('Selected retry unexpectedly completed by another execution')
        cold.validate_closed_primary_record(primary, record)
        verified[identity] = record


def namespace():
    # Reuse bounded hashes and the unchanged pure metadata fingerprint. The
    # startup snapshot verifier is replaced below with the complete supplement
    # binding; numerical child functions retain their original code objects.
    values = cold.namespace({'preserved_verified_records': '', 'expected_verified_models': 128})
    original_verify = FunctionType(supplement.original.verify.__code__, values)

    def verify(runtime):
        previous = supplement.original.read(ROOT / supplement.PREVIOUS)
        production = supplement.original.read(ROOT / supplement.PRODUCER)
        selected, recovery = supplement.producer.verify(production)
        if (runtime['previous_full285_configuration'] != supplement.PREVIOUS
                or runtime['producer_configuration'] != supplement.PRODUCER
                or runtime['routes'] != supplement.expected_routes(previous, recovery)
                or runtime['selected_canonical_ids'] != [j['recovery_original_job_id'] for j in selected]):
            raise ValueError('Every285 route and exactly all8 original retries required')
        for key in ['python', 'cpu_threads', 'settings', 'resource_lock', 'environment_lock',
                    'prediction_tolerance', 'original_scope']:
            if runtime[key] != previous[key]:
                raise ValueError('Original numerical worker or guard changed: ' + key)
        routes, template, verified = original_verify(runtime)
        adopt_primary(runtime, verified)
        return routes, template, verified

    values.update(verify=verify, _selected_ready=supplement.selected_ready,
                  _adopt_primary=adopt_primary)
    exec(compile(supplement.controller_ast(), __file__, 'exec'), values)
    return values


def idle_allowed(role, state, output_exists, descendants):
    return cold.idle_allowed(role, state, output_exists, descendants)


def verify_config(config):
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen supplement handoff source changed: ' + pin['path'])
    runtime = supplement.original.read(ROOT / config['original_runtime'])
    values = namespace()
    routes, _, records = values['verify'](runtime)
    snapshot = supplement.original.read(ROOT / config['preserved_verified_records'])['records']
    for record in snapshot:
        if records.get(record['original_canonical_id']) != record:
            raise ValueError('Every existing closed certificate must survive')
    if len(snapshot) != 128 or len(routes) != 285 or runtime['original_scope'] != config['original_scope']:
        raise ValueError('Full research scope,285 original slots and128 closed proofs required')
    return runtime, values, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = supplement.original.read(ROOT / args.config)
    runtime, values, records = verify_config(config)
    if args.probe_only:
        import psutil
        print(dict(original_models=len(runtime['routes']), selected_retry_routes=len(runtime['selected_canonical_ids']),
                   actual_guarded_certificates=len(records), every_artifact_byte_hashed=True,
                   numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                   private_bytes=psutil.Process().memory_info().private))
        return
    base = ROOT / config['handoff_state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = cold.cached_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Supplement handoff already owned')
    try:
        function = FunctionType(cold.handoff.safe_handoff.__code__,
                                dict(cold.handoff.__dict__, handoff_phase_allowed=idle_allowed))
        route = dict(previous_owner=config['previous_owner'], previous_state_path=config['previous_state_path'],
                     runtime_configuration=args.config)
        function('original', route, base, lambda state: supplement.original.write(base / 'status.json', state), 10)
        verify_config(config)
        values['controller'](runtime, config['original_runtime'])
    finally:
        owner.close()


if __name__ == '__main__':
    main()
