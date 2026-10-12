"""Continue all162 remaining original epochs after complete native state proof."""
from scripts.flow_matching import rcgan_native_training_resume_runtime_v1 as original
from scripts.flow_matching.rcgan_seed1105_resume_binding_v1 import bind_original


def bind():
    return bind_original(original, 'rcgan_native_training_resume_runtime_v1.py')


def run(config, queue, binding):
    return bind()['run'](config, queue, binding)


def verify_result(config, queue):
    return bind()['verify_result'](config, queue)
