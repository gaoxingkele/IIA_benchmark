"""Native and independent full-grid review of exactly two metadata-failed slots.

Keep full285 scope and adopt closed primary certificates. Every numerical child,
native tolerance and resource cap remains the original implementation.
"""
from types import FunctionType
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as supplement
from scripts.flow_matching import run_cfm_metadata_permission_full_retry_v1 as producer

ROOT=cold.ROOT
PRIMARY='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
PRODUCER='configs/runtime/fm_cfm_metadata_permission_full_retry.v1.json'


def adopt_primary(runtime,verified):
    primary=cold.original.read(ROOT/PRIMARY)
    records=cold.original.read(ROOT/primary['state_root']/'verified_full_model_records.json')['records']
    for record in records:
        identity=record['original_canonical_id']
        if identity in verified:
            if verified[identity]!=record:raise ValueError('Immutable closed primary record changed')
            continue
        if identity in runtime['selected_canonical_ids']:raise ValueError('Selected fresh execution unexpectedly already certified')
        cold.validate_closed_primary_record(primary,record);verified[identity]=record


def namespace():
    values=cold.namespace({'preserved_verified_records':'','expected_verified_models':128})
    baseverify=FunctionType(cold.original.verify.__code__,values)
    def verify(runtime):
        primary=cold.original.read(ROOT/PRIMARY);production=cold.original.read(ROOT/PRODUCER)
        chosen,queue=producer.verify_runtime(production)
        if (runtime['previous_full285_configuration']!=PRIMARY or runtime['producer_configuration']!=PRODUCER
                or runtime['routes']!=supplement.expected_routes(primary,queue)
                or runtime['selected_canonical_ids']!=[j['recovery_original_job_id'] for j in chosen]):
            raise ValueError('Keep every285 route and exactly both scientific-equivalent retries')
        for key in ['python','cpu_threads','settings','resource_lock','environment_lock','prediction_tolerance','original_scope']:
            if runtime[key]!=primary[key]:raise ValueError('Original numerical replay policy changed: '+key)
        routes,template,verified=baseverify(runtime);adopt_primary(runtime,verified)
        return routes,template,verified
    values.update(verify=verify,_selected_ready=supplement.selected_ready,_adopt_primary=adopt_primary)
    exec(compile(supplement.controller_ast(),__file__,'exec'),values)
    return values


def main():
    FunctionType(cold.original.main.__code__,namespace())()


if __name__=='__main__':main()
