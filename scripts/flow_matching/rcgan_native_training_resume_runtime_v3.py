"""Run all134 remaining source epochs only from independently verified366 state."""
from scripts.flow_matching import rcgan_native_training_resume_runtime_v1 as original
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4 import bind as state_binding
from scripts.flow_matching.rcgan_seed1104_resume_binding_v3 import full_resume_ast, cached_resource_api


def bind():
    source = original.ROOT / 'scripts/flow_matching/rcgan_native_training_resume_runtime_v1.py'
    namespace = dict(original.__dict__, state_verify_registration=state_binding()['verify_registration'],
                     _cached_resource_api=cached_resource_api)
    exec(compile(full_resume_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


def run(config, queue, binding):
    return bind()['run'](config, queue, binding)


def verify_result(config, queue):
    return bind()['verify_result'](config, queue)
