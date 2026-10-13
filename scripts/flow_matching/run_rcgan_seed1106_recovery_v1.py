"""Original full500-budget continuation with stricter40GiB resource admission."""
from scripts.flow_matching import run_rcgan_native_state_recovery_v1 as original
from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import bind_original


def bind():
    return bind_original(original, 'run_rcgan_native_state_recovery_v1.py')


if __name__ == '__main__':
    bind()['main']()
