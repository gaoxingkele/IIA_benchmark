from types import SimpleNamespace
from scripts.flow_matching.crnngan_executor_cache_runtime_v1 import normalize_inert_fetches


def test_only_dependency_free_inert_targets_are_reused_without_mutating_fetches():
    graph = object()
    cache = {}
    def noop(**kwargs):
        return SimpleNamespace(type='NoOp', graph=graph, inputs=(), control_inputs=(), **kwargs)
    first, later = noop(), noop()
    optimizer = SimpleNamespace(type='NoOp', graph=graph, inputs=(), control_inputs=('real_optimizer_update',))
    loss = object()
    assert normalize_inert_fetches([loss, first, optimizer], graph, cache) == [loss, first, optimizer]
    request = [loss, later, optimizer]
    result = normalize_inert_fetches(request, graph, cache)
    assert result == [loss, first, optimizer]
    assert request[1] is later and result[2] is optimizer
    foreign = SimpleNamespace(type='NoOp', graph=object(), inputs=(), control_inputs=())
    assert normalize_inert_fetches([loss, foreign], graph, cache)[1] is foreign


def test_each_fetch_position_keeps_its_own_inert_target_and_optimizer_objects():
    graph = object()
    cache = {}
    a = SimpleNamespace(type='NoOp', graph=graph, inputs=(), control_inputs=())
    b = SimpleNamespace(type='NoOp', graph=graph, inputs=(), control_inputs=())
    assert normalize_inert_fetches((a, b), graph, cache) == (a, b)
    assert normalize_inert_fetches((b, a), graph, cache) == (a, b)
    assert len(cache) == 2
