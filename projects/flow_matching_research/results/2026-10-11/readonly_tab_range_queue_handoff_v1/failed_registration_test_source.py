import ast
import copy
import json

import pytest

from scripts.flow_matching.run_readonly_tab_range_queue_v1 import ROOT, OLD_LOCK, bind, readonly_main_ast, stronger_limits


def settings():
    return dict(minimum_available_memory_bytes=16, minimum_commit_headroom_bytes=28,
                emergency_commit_headroom_bytes=16, maximum_worker_rss_bytes=8, maximum_worker_private_bytes=10)


def test_original_reserves_cannot_be_reduced_and_worker_caps_require_headroom():
    original = dict(minimum_available_memory_bytes=8, minimum_commit_headroom_bytes=12, emergency_commit_headroom_bytes=8)
    assert stronger_limits(original, settings()) == settings()
    for field in original:
        changed = settings()
        changed[field] = original[field]-1
        with pytest.raises(ValueError, match='safeguard'):
            stronger_limits(original, changed)
    changed = settings()
    changed['minimum_commit_headroom_bytes'] = 25
    with pytest.raises(ValueError, match='enough commit'):
        stronger_limits(original, changed)


def test_compiled_original_controller_only_changes_one_mutex_literal():
    source = (ROOT / 'scripts/flow_matching/run_tab_ranges_sparse_queue.py').read_text(encoding='utf-8')
    original = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    changed = readonly_main_ast(source, 'experiments/runs/readonly-test.lock')
    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'experiments/runs/readonly-test.lock':
                node.value = OLD_LOCK
            return node
    restored = Restore().visit(copy.deepcopy(changed))
    assert ast.dump(restored) == ast.dump(original)
    with pytest.raises(ValueError, match='Exactly'):
        readonly_main_ast(source.replace(OLD_LOCK, 'unexpected.lock'), 'readonly-test.lock')
    with pytest.raises(ValueError, match='Exactly'):
        readonly_main_ast(source, OLD_LOCK)


def test_all_registered_readonly_lanes_keep_original_scientific_configuration():
    runtime = json.loads((ROOT / 'configs/runtime/fm_readonly_tab_range_queues.v1.json').read_text(encoding='utf-8'))
    assert set(runtime['controllers']) == {'range_main', 'range_reflow', 'range_industrial', 'range_moment'}
    total = 0
    for name in runtime['controllers']:
        main, config, limits, binding = bind(ROOT, runtime, name)
        assert config['cpu_threads'] == 2
        assert binding['queue_path'] == runtime['controllers'][name]['queue']
        assert limits['minimum_commit_headroom_bytes'] >= config['runtime_resources']['minimum_commit_headroom_bytes']
        assert main.__code__.co_filename == str(ROOT / binding['source_path'])
        total += sum(len(json.loads((ROOT / path).read_text(encoding='utf-8'))['jobs']) for path in config['queue_paths'])
    assert total == 3362


def test_frozen_math_queue_or_non_readonly_controller_cannot_enter():
    runtime = json.loads((ROOT / 'configs/runtime/fm_readonly_tab_range_queues.v1.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='four original readonly'):
        bind(ROOT, runtime, 'crossad')
    runtime['controllers']['range_main']['queue'] = 'different-science.json'
    with pytest.raises(ValueError, match='scientific evaluation'):
        bind(ROOT, runtime, 'range_main')
