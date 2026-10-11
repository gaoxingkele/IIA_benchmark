"""Queue full416 remaining epochs of original seed1104 after full native state proof."""
from scripts.flow_matching import run_rcgan_native_state_recovery_v1 as original
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v3 import bind as state_binding
from scripts.flow_matching.rcgan_seed1104_resume_binding_v2 import full_resume_ast


def bind():
    path = original.ROOT / 'scripts/flow_matching/run_rcgan_native_state_recovery_v1.py'
    namespace = dict(original.__dict__, state_verify_registration=state_binding()['verify_registration'],
        __doc__=__doc__)
    exec(compile(full_resume_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
