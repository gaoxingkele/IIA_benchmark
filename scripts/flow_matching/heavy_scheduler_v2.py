"""Reserve unchanged resource limits without holding a shared lock while idle.

Old frozen experiment controllers remain byte-identical. This scheduler-only
binding replaces their shared-lock entry and redundant headroom wait, while
retaining job dispatch, artifact validation and the existing emergency guard.
"""
from contextlib import contextmanager
from pathlib import Path
import time

from scripts.flow_matching.heavy_resources import (
    exclusive_lock, has_headroom, resource_snapshot,
)


@contextmanager
def resource_slot(path, settings, update=None, poll_seconds=10):
    while True:
        snapshot = resource_snapshot()
        handle = None
        if has_headroom(snapshot, settings):
            handle = exclusive_lock(path)
            if handle is not None:
                # Resources may change between the initial check and ownership.
                try:
                    snapshot = resource_snapshot()
                    ready = has_headroom(snapshot, settings)
                except BaseException:
                    handle.close()
                    raise
                if ready:
                    break
                handle.close()
                handle = None
        if update:
            update(dict(snapshot, state='waiting_for_memory' if not has_headroom(snapshot, settings)
                        else 'waiting_for_heavy_lock', shared_lock_owned=False))
        time.sleep(poll_seconds)
    try:
        yield snapshot
    finally:
        handle.close()


def bind_controller(controller, root, settings):
    """Bind only the shared-heavy critical section; keep lane locks unchanged."""
    old_waited_lock = controller.waited_lock
    shared = (root / 'experiments/runs/fm_heavy_recovery_cpu.lock').resolve()
    reservation = {'snapshot': None}

    @contextmanager
    def selected_lock(path, waiting=None, poll_seconds=10):
        if Path(path).resolve() != shared:
            with old_waited_lock(path, waiting, poll_seconds) as handle:
                yield handle
            return
        with resource_slot(path, settings, waiting, poll_seconds) as snapshot:
            reservation['snapshot'] = snapshot
            try:
                yield snapshot
            finally:
                reservation['snapshot'] = None

    def reserved_headroom(supplied, update=None, poll_seconds=10):
        if supplied != settings or reservation['snapshot'] is None:
            raise RuntimeError('Resource headroom must be obtained by an owned shared slot')
        # The slot has already rechecked headroom while owning the lock. Waiting
        # here again would recreate the idle-owner convoy. The child guard still
        # monitors actual system commit continuously after dispatch.
        return dict(reservation['snapshot'])

    controller.waited_lock = selected_lock
    controller.wait_for_headroom = reserved_headroom


def run_bound_controller(controller):
    import json
    import sys
    queue_argument = sys.argv[sys.argv.index('--queue') + 1]
    queue = json.loads((controller.ROOT / queue_argument).read_text(encoding='utf-8'))
    bind_controller(controller, controller.ROOT, queue['settings'])
    controller.main()
