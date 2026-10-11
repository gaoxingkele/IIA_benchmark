import ast
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching.crnngan_executor_cache_runtime_v1 import normalize_inert_fetches
from scripts.flow_matching.crnngan_executor_cache_runtime_v2 import scientific_fetch_signature
from scripts.flow_matching.audit_crnngan_scoped_cache_four_epochs_v2 import worker_tree


def op(graph, name, control_inputs=()):
    return SimpleNamespace(type='NoOp', graph=graph, name=name, inputs=(), control_inputs=control_inputs)


def test_empty_fetch_reuse_is_bounded_within_same_actual_scientific_request():
    graph = object()
    loss = SimpleNamespace(name='model/loss:0')
    optimizer = op(graph, 'model/optimizer', control_inputs=(object(),))
    first = [loss, op(graph, 'empty1'), optimizer, op(graph, 'dummy1')]
    second = [loss, op(graph, 'empty2'), optimizer, op(graph, 'dummy2')]
    assert scientific_fetch_signature(first, graph) == scientific_fetch_signature(second, graph)
    cache = {}
    normalize_inert_fetches(first, graph, cache)
    actual = normalize_inert_fetches(second, graph, cache)
    assert actual[0] is loss and actual[2] is optimizer
    assert actual[1] is first[1] and actual[3] is first[3]
    assert len(cache) == 2


def test_curriculum_optimizer_and_freeze_modes_cannot_share_empty_fetch_targets():
    graph = object()
    first = [SimpleNamespace(name='model/loss:0'), op(graph, 'dummy1'),
             op(graph, 'model/optimizer', (object(),))]
    curriculum = [SimpleNamespace(name='model_1/loss:0'), op(graph, 'dummy2'),
                  op(graph, 'model_1/optimizer', (object(),))]
    freeze = [first[0], op(graph, 'dummy3'), op(graph, 'freeze_G')]
    signatures = [scientific_fetch_signature(f, graph) for f in (first, curriculum, freeze)]
    assert len(set(signatures)) == 3
    caches = {}
    for request, signature in zip((first, curriculum, freeze), signatures):
        actual = normalize_inert_fetches(request, graph, caches.setdefault(signature, {}))
        assert actual[1] is request[1]


def test_complete_original_gate_only_changes_backend_import():
    from scripts.flow_matching import audit_crnngan_four_complete_epochs_v1 as original
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    actual = worker_tree().body[0]
    for node in ast.walk(actual):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.crnngan_executor_cache_runtime_v2':
            node.module = 'scripts.flow_matching.crnngan_executor_cache_runtime_v1'
    assert ast.dump(actual) == ast.dump(worker)
