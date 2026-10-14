"""Collect a corrected test fixture while retaining an immutable audit source.

Only the documented Pi causal1106 seed-list typo is eligible. The replacement
must keep every other AST node, including all scientific rejection checks.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'tests/test_fm_pi_causal1106_completed_full_replay_v1.py'
SUCCESSOR = ROOT / 'tests/test_fm_pi_causal1106_scope_corrected_v2.py'
RUNTIME = ROOT / 'configs/runtime/fm_pi_causal1106_completed_full_replay.v1.json'


def validate_successor(legacy_source, successor_source, frozen_digest):
    if hashlib.sha256(legacy_source).hexdigest() != frozen_digest:
        raise ValueError('Frozen historical validation source changed')
    legacy = ast.parse(legacy_source.decode('utf-8'))
    successor = ast.parse(successor_source.decode('utf-8'))
    target = next(node for node in legacy.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'test_full_original_scope_and_budget_accepted')
    corrections = 0
    for node in ast.walk(target):
        if isinstance(node, ast.List) and ast.literal_eval(node) == [42,1103,1104,1106,1106,1107]:
            node.elts[3] = ast.Constant(value=1105)
            corrections += 1
    if corrections != 1:
        raise ValueError('Exactly the documented historical seed-list typo required')
    for tree in (legacy, successor):
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body.pop(0)
    if ast.dump(legacy, include_attributes=False) != ast.dump(successor, include_attributes=False):
        raise ValueError('The successor must retain every original scientific check')


def verify_registered_successor():
    config = json.loads(RUNTIME.read_text(encoding='utf-8'))
    relative = LEGACY.relative_to(ROOT).as_posix()
    pins = [pin for pin in config['source_receipts'] if pin['path'] == relative]
    if len(pins) != 1:
        raise ValueError('Exactly one original frozen validation source pin required')
    validate_successor(LEGACY.read_bytes(), SUCCESSOR.read_bytes(), pins[0]['sha256'])
