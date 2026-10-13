"""Full285 replay union; select the two entry-corrected scientific executions.

Keep the original native and independent numerical workers, caps and tolerance.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
from types import FunctionType
from scripts.flow_matching import run_cfm_metadata_permission_replay_queue_v1 as previous

PRODUCER = 'configs/runtime/fm_cfm_metadata_entry_full_retry.v2.json'


def namespace():
    return FunctionType(previous.namespace.__code__, dict(previous.__dict__, PRODUCER=PRODUCER))()


def main():
    FunctionType(previous.cold.original.main.__code__, namespace())()


if __name__ == '__main__':
    main()
