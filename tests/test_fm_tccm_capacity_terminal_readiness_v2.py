import ast
import json
from pathlib import Path

import pytest

from scripts.flow_matching import run_tccm_full_image_recovery_queue_v2 as queue


def test_report_before_terminal_never_admits_full_training(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.original, 'ROOT', tmp_path)
    output = tmp_path / 'capacity'; output.mkdir()
    config = dict(capacity_cases=[dict(output_directory='capacity')])
    assert queue.capacity_receipts_pending(config)
    (output / 'capacity_probe_result.json').write_text(json.dumps(dict(path='capacity/attempt/report.json')))
    assert queue.capacity_receipts_pending(config)
    attempt = output / 'attempt'; attempt.mkdir()
    (attempt / 'report.json').write_text('{}')
    assert queue.capacity_receipts_pending(config)
    (attempt / 'terminal.json').write_text('{}')
    assert not queue.capacity_receipts_pending(config)
    # Presence only releases the wait; the unchanged strict capacity verifier
    # still checks the actual return code, resource guards and full-model proof.


def test_only_capacity_wait_condition_changes():
    source = Path(queue.original.__file__).read_text(encoding='utf-8')
    expected = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    tree = queue.ready_main_ast(source); compile(tree, '<terminal-ready>', 'exec')
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    assert len(loops) == 1 and ast.unparse(loops[0].test) == '_capacity_receipts_pending(config)'
    loops[0].test = next(n.test for n in ast.walk(expected) if isinstance(n, ast.While))
    assert ast.dump(tree.body[0]) == ast.dump(expected)


def test_changed_wait_structure_rejected():
    with pytest.raises(ValueError):
        queue.ready_main_ast('def main(): pass')
