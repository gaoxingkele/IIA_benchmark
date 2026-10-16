"""Bridge Torchdyn's empty auxiliary-args ABI to unchanged legacy author math."""
from functools import wraps


def empty_args_adapter(original):
    @wraps(original)
    def compatible(self, t, x, args=None):
        if args is not None and (type(args) is not dict or args):
            raise ValueError('Legacy OT-CFM has no auxiliary ODE arguments; never discard nonempty scientific inputs')
        return original(self, t, x)
    compatible._iia_empty_ode_args_adapter = True
    return compatible


def install(owner, name):
    original = getattr(owner, name)
    if not getattr(original, '_iia_empty_ode_args_adapter', False):
        setattr(owner, name, empty_args_adapter(original))
