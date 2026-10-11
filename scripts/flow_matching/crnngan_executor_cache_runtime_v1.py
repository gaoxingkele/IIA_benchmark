"""Bound inert-fetch executor caching without changing the original TF1 graph."""


def normalize_inert_fetches(fetches, graph, cache, path=()):
    if isinstance(fetches, list):
        return [normalize_inert_fetches(item, graph, cache, path+(index,)) for index, item in enumerate(fetches)]
    if isinstance(fetches, tuple):
        return tuple(normalize_inert_fetches(item, graph, cache, path+(index,)) for index, item in enumerate(fetches))
    if (getattr(fetches, 'type', None) == 'NoOp' and getattr(fetches, 'graph', None) is graph
            and not fetches.inputs and not fetches.control_inputs):
        return cache.setdefault(path, fetches)
    return fetches


def run(queue, job):
    import tensorflow as tf
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    session_type = tf.Session

    class BoundedExecutorSession(session_type):
        def __init__(self, *args, **kwargs):
            session_type.__init__(self, *args, **kwargs)
            self._iia_inert_fetches = {}

        def run(self, fetches, *args, **kwargs):
            normalized = normalize_inert_fetches(fetches, self.graph, self._iia_inert_fetches)
            return session_type.run(self, normalized, *args, **kwargs)

    tf.Session = BoundedExecutorSession
    try:
        return original.run(queue, job)
    finally:
        tf.Session = session_type
