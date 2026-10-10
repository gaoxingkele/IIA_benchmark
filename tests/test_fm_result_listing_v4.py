import ast
import pytest
from scripts.flow_matching.capture_result_listing_v4 import readonly_function


def test_readonly_capture_preserves_computation_and_skips_all_publication():
    source = '''def snapshot(root):
    values = root + 3
    report = {"values": values}
    target = dangerous()
    write_json(target, report)
    return report
'''
    namespace = {'dangerous': lambda: pytest.fail('Publication tail executed')}
    exec(compile(readonly_function(source), '<test>', 'exec'), namespace)
    assert namespace['snapshot'](4) == {'values': 7}


def test_changed_report_boundary_rejected():
    with pytest.raises(ValueError, match='report boundary'):
        readonly_function('def snapshot(root):\n    return {}\n')


def test_changed_publication_boundary_rejected():
    with pytest.raises(ValueError, match='publication boundary'):
        readonly_function('def snapshot(root):\n    report = {}\n    return report\n')


def test_actual_snapshot_uses_registered_boundary():
    from pathlib import Path
    source = Path('scripts/flow_matching/summarize_tsad_execution.py').read_text(encoding='utf-8')
    tree = readonly_function(source)
    assert isinstance(tree.body[0].body[-1], ast.Return)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'write_json'
                   for n in ast.walk(tree))
