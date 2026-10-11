import ast
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching.run_preserved_native_gan_retry_v1 import admission_settings, controller_ast


SETTINGS = dict(minimum_available_memory_bytes=24 << 30,
    minimum_commit_headroom_bytes=28 << 30, emergency_commit_headroom_bytes=16 << 30,
    maximum_worker_rss_bytes=10 << 30, maximum_worker_private_bytes=12 << 30)


def test_stricter_admission_preserves_caps_and_input():
    original = deepcopy(SETTINGS)
    config = dict(retry_admission_commit_headroom_bytes=40 << 30)
    result = admission_settings(config, original)
    assert original == SETTINGS
    assert result == dict(SETTINGS, minimum_commit_headroom_bytes=40 << 30)
    assert config == dict(retry_admission_commit_headroom_bytes=40 << 30)


@pytest.mark.parametrize('value', [True, None, '42949672960', -1, 0, 27 << 30, 28 << 30])
def test_no_loosened_or_ambiguous_admission(value):
    with pytest.raises(ValueError):
        admission_settings(dict(retry_admission_commit_headroom_bytes=value), SETTINGS)


@pytest.mark.parametrize('family,filename', [
    ('rcgan', 'run_rcgan_native_state_recovery_v1.py'),
    ('timegan', 'run_timegan_stage_resume_v2.py')])
def test_only_resource_admission_changes_in_actual_controller(family, filename):
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/flow_matching' / filename).read_text(encoding='utf-8')
    expected = next(n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == 'main')
    adapted = controller_ast(source, family)
    changes = 0
    for node in ast.walk(adapted):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_retry_admission_settings':
            assert len(node.args) == 2 and node.args[0].id == 'config'
            changes += 1
    assert changes == 1
    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_retry_admission_settings':
                return node.args[1]
            return self.generic_visit(node)
    assert ast.dump(Restore().visit(adapted.body[0])) == ast.dump(expected)
    compile(adapted, filename, 'exec')


@pytest.mark.parametrize('source', [
    'def main(): pass',
    "def main():\n api['resource_slot'](lock, wrong, update)",
    "def main():\n api['resource_slot'](lock, config['settings'], update)\n api['resource_slot'](lock, config['settings'], update)",
    "def main():\n api['resource_slot'](lock, config['settings'])"])
def test_changed_controller_structure_rejected(source):
    with pytest.raises(ValueError):
        controller_ast(source, 'rcgan')


def test_unsupported_family_rejected():
    with pytest.raises(ValueError):
        controller_ast('def main(): pass', 'unknown')
