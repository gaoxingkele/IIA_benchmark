"""Audit every saved native RNG draw and cumulative image order across68 epochs."""
from scripts.flow_matching import audit_rcgan_random_order_chain_v1 as original
from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import bind_original

nested_tuple = original.nested_tuple


def bind():
    return bind_original(original, 'audit_rcgan_random_order_chain_v1.py')


def verify_registration(config):
    return bind()['verify_registration'](config)


if __name__ == '__main__':
    bind()['main']()
