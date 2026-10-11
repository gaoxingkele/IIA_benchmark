import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.crnngan_reviewed_source_v1 import instrumented_main as original_main
from scripts.flow_matching.crnngan_bounded_noop_runtime_v1 import instrumented_main
from scripts.flow_matching.run_crnngan_bounded_noop_queue_v1 import controller_tree


def source_path():
    asset = next(a for a in read(ROOT/'configs/acquisition/fm_adopted_gan_original_sources.v1.json')['assets'] if a['id'] == 'crnngan')
    return Path(asset['original_directory'])/'rnn_gan.py'


def test_no_scientific_statement_or_full_loop_is_removed():
    tree = instrumented_main(source_path())
    tree.body.pop(0)
    replacements = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == '_native_cached_no_op']
    assert len(replacements) == 8
    for n in replacements:
        n.func = ast.Attribute(value=ast.Name(id='tf', ctx=ast.Load()), attr='no_op', ctx=ast.Load())
    assert ast.dump(tree) == ast.dump(original_main(source_path()))


def test_inert_fetch_is_reused_within_one_graph_and_isolated_between_graphs():
    helper = ast.Module(body=[instrumented_main(source_path()).body[0]], type_ignores=[])
    graphs = [SimpleNamespace(), SimpleNamespace()]
    current = [graphs[0]]
    created = []
    def no_op(**kwargs):
        value = object()
        created.append(value)
        return value
    ns = dict(tf=SimpleNamespace(get_default_graph=lambda: current[0], no_op=no_op))
    exec(compile(ast.fix_missing_locations(helper), 'native_noop', 'exec'), ns)
    first = ns['_native_cached_no_op']()
    assert all(ns['_native_cached_no_op']() is first for _ in range(1000))
    current[0] = graphs[1]
    assert ns['_native_cached_no_op']() is not first and len(created) == 2


def test_full_controller_only_changes_worker_binding_and_dispatcher():
    from scripts.flow_matching import run_crnngan_native_original_queue_v2 as old
    tree = controller_tree()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == 'scripts.flow_matching.crnngan_bounded_noop_runtime_v1':
            n.module = 'scripts.flow_matching.crnngan_native_training_runtime_v1'
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = n.value.replace('scripts.flow_matching.run_crnngan_bounded_noop_queue_v1',
                                      'scripts.flow_matching.run_crnngan_native_original_queue_v2')
    assert ast.dump(tree) == ast.dump(ast.parse(Path(old.__file__).read_text(encoding='utf-8')))
