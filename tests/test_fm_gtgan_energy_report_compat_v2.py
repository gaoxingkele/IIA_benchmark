import ast
import json
from pathlib import Path

from scripts.flow_matching.audit_gtgan_energy_native_v2 import write


def test_python38_atomic_writer_preserves_json_values_and_lf_bytes(tmp_path):
    path = tmp_path / 'native' / 'receipt.json'
    payload = dict(passed=True, label='能源', optimizer_updates=17000)
    write(path, payload)
    assert json.loads(path.read_text(encoding='utf-8')) == payload
    assert b'\r' not in path.read_bytes()
    assert not path.with_name(path.name + '.tmp').exists()
    write(path, dict(payload, optimizer_updates=22000))
    assert json.loads(path.read_text())['optimizer_updates'] == 22000


def test_report_fix_does_not_modify_data_processing_or_native_models():
    original = ast.parse(Path('scripts/flow_matching/audit_gtgan_energy_native_v1.py').read_text(encoding='utf-8'))
    fixed = ast.parse(Path('scripts/flow_matching/audit_gtgan_energy_native_v2.py').read_text(encoding='utf-8'))
    for name in ['worker', 'original_loader_module', 'verify']:
        a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in fixed.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b)
