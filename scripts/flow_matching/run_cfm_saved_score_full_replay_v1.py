"""Keep all285 routes; replay only the saved-model metadata scoring recovery.

Reuse unchanged native inference and independent full-array review workers.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
from types import FunctionType
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as supplement
from scripts.flow_matching import run_cfm_saved_full_score_recovery_v1 as recovery


def namespace():
    values = cold.namespace({'preserved_verified_records': '', 'expected_verified_models': 128})
    baseverify = FunctionType(cold.original.verify.__code__, values)
    def adopt(runtime, verified):
        primary = cold.original.read(cold.ROOT / runtime['previous_full285_configuration'])
        records = cold.original.read(cold.ROOT / primary['state_root'] / 'verified_full_model_records.json')['records']
        for record in records:
            identity = record['original_canonical_id']
            if identity in verified:
                if verified[identity] != record:
                    raise ValueError('Immutable full original certificate changed')
                continue
            if identity in runtime['selected_canonical_ids']:
                raise ValueError('Recovery unexpectedly already certified by another producer')
            cold.validate_closed_primary_record(primary, record)
            verified[identity] = record
    def verify(runtime):
        config = cold.original.read(cold.ROOT / runtime['producer_configuration'])
        primary = recovery.verify(config)
        expected = [dict(r, job=config['job'], result_path=config['job']['output_directory']+'/result.json')
            if r['original_canonical_id'] == recovery.IDENTITY else r for r in primary['routes']]
        if (runtime['routes'] != expected or runtime['selected_canonical_ids'] != [recovery.IDENTITY]
                or runtime['previous_full285_configuration'] != config['primary_runtime']):
            raise ValueError('Keep every285 route and change only the saved-model scoring execution')
        for key in ['python', 'cpu_threads', 'settings', 'resource_lock', 'environment_lock', 'prediction_tolerance', 'original_scope']:
            if runtime[key] != primary[key]:
                raise ValueError('Original complete native/review policy changed')
        routes, template, verified = baseverify(runtime)
        adopt(runtime, verified)
        return routes, template, verified
    values.update(verify=verify, _selected_ready=supplement.selected_ready, _adopt_primary=adopt)
    exec(compile(supplement.controller_ast(), __file__, 'exec'), values)
    return values


def main():
    FunctionType(cold.original.main.__code__, namespace())()


if __name__ == '__main__':
    main()
