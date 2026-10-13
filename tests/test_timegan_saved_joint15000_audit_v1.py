import ast
from pathlib import Path
import random

from scripts.flow_matching.audit_timegan_sine_saved_joint15000_v1 import tuple_state
from scripts.flow_matching.timegan_joint_checkpoint_observer_v1 import expected_joint_counts


def test_saved_python_state_roundtrip_restores_all_fields():
    source = random.Random(1103)
    original = source.getstate()
    import json
    restored = tuple_state(json.loads(json.dumps(original)))
    target = random.Random(); target.setstate(restored)
    assert target.getstate() == original
    assert target.random() == source.random()


def test_partial_generator_iteration_cannot_be_credited_as_full_state():
    counts = dict(E0=50000, GS=50000, E=30000, G=30000, D=15000)
    assert expected_joint_counts(counts, 15000)
    assert not expected_joint_counts(dict(counts, G=29999), 15000)
    assert not expected_joint_counts(dict(counts, E=30400), 15000)


def test_native_audit_has_no_optimization_or_inference_call():
    tree = ast.parse(Path('scripts/flow_matching/audit_timegan_sine_saved_joint15000_v1.py').read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ['minimize', 'fit', 'backward', 'run'] for n in calls)
    assert sum(isinstance(n.func, ast.Name) and n.func.id == 'restore_all' for n in calls) == 1
