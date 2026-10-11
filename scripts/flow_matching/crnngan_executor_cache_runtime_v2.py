"""Scope inert executor fetch reuse to each unchanged scientific fetch request."""
from scripts.flow_matching.crnngan_executor_cache_runtime_v1 import normalize_inert_fetches


def scientific_fetch_signature(fetches, graph, path=()):
    if isinstance(fetches, (list, tuple)):
        return (type(fetches).__name__, tuple(scientific_fetch_signature(value, graph, path+(i,))
                                            for i, value in enumerate(fetches)))
    if (getattr(fetches, 'type', None) == 'NoOp' and getattr(fetches, 'graph', None) is graph
            and not fetches.inputs and not fetches.control_inputs):
        return ('inert', path)
    # Names identify the actual Tensor/Operation in this Session's fixed graph.
    # Different curriculum losses, optimizers and freeze modes keep separate caches.
    name = getattr(fetches, 'name', None)
    return ('scientific', path, name if name is not None else id(fetches))


def run(queue, job):
    import tensorflow as tf
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    session_type = tf.Session

    class ScopedExecutorSession(session_type):
        def __init__(self, *args, **kwargs):
            session_type.__init__(self, *args, **kwargs)
            self._iia_inert_fetches_by_request = {}

        def run(self, fetches, *args, **kwargs):
            signature = scientific_fetch_signature(fetches, self.graph)
            cache = self._iia_inert_fetches_by_request.setdefault(signature, {})
            normalized = normalize_inert_fetches(fetches, self.graph, cache)
            return session_type.run(self, normalized, *args, **kwargs)

    tf.Session = ScopedExecutorSession
    try:
        return original.run(queue, job)
    finally:
        tf.Session = session_type
