"""Execute all416 remaining original native epochs after complete84epoch state proof."""
from scripts.flow_matching import rcgan_native_training_resume_runtime_v1 as original
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v3 import bind as state_binding
from scripts.flow_matching.rcgan_seed1104_resume_binding_v2 import full_resume_ast


def bind():
    path = original.ROOT / 'scripts/flow_matching/rcgan_native_training_resume_runtime_v1.py'
    namespace = dict(original.__dict__, state_verify_registration=state_binding()['verify_registration'])
    exec(compile(full_resume_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


def run(config, queue, binding):
    return bind()['run'](config, queue, binding)


def verify_result(config, queue):
    return bind()['verify_result'](config, queue)
