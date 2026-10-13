"""Reuse cached host telemetry for unstarted GTGAN/TCCM and GRASP reviews.

The immutable v1 handoff and original entry modules remain unchanged. Extend
only the explicit host allowlist; preserve numerical workers and resource locks.
"""
from types import FunctionType
from scripts.flow_matching import run_idle_full_scope_cached_bundle_v1 as previous

EXTRA_HOSTS={
    'scripts.flow_matching.run_gtgan_energy_native_queue_v1',
    'scripts.flow_matching.run_tccm_full_image_capacity_recovery_v1',
    'scripts.flow_matching.audit_registered_grasp_results_v1',
}


def namespace():
    values=dict(previous.__dict__,ALLOWED=previous.ALLOWED|EXTRA_HOSTS,__doc__=__doc__)
    values['verify']=FunctionType(previous.verify.__code__,values)
    return values


def entry_main():
    return FunctionType(previous.main.__code__,namespace())


def main():
    entry_main()()


if __name__=='__main__':main()
