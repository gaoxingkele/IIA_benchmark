"""Restore all68 full checkpoints and61 actual persisted losses without loosening."""
from scripts.flow_matching import audit_rcgan_checkpoint_resume_state_v1 as original
from scripts.flow_matching.audit_rcgan_seed1106_chain_v1 import verify_registration as verify_chain
from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import bind_original


def bind():
    return bind_original(original, 'audit_rcgan_checkpoint_resume_state_v1.py',
        verify_chain_registration=verify_chain)


def verify_registration(config):
    return bind()['verify_registration'](config)


if __name__ == '__main__':
    bind()['main']()
