import ast
from pathlib import Path

from scripts.flow_matching import audit_crnngan_original_four_epochs_v1 as original
from scripts.flow_matching.audit_crnngan_original_four_epochs_v2 import worker_tree


def test_native_midi_observation_only_changes_filename_lookup():
    source = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    actual = worker_tree().body[0]
    old = next(n for n in ast.walk(worker) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'glob')
    new = next(n for n in ast.walk(actual) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'glob')
    assert ast.unparse(new.args[0]) == "'out-%s-%d-*.mid' % (root.name, index)"
    new.args[0] = old.args[0]
    assert ast.dump(actual) == ast.dump(worker)


def test_original_source_sample_pattern_matches_released_and_diagnostic_directory_names(tmp_path):
    for basename in ('seed1103', 'crnngan_original_four_epochs_v2'):
        root = tmp_path / basename
        samples = root / 'generated_data'
        samples.mkdir(parents=True)
        for index in range(4):
            path = samples / ('out-%s-%d-2026-10-11-09-25-22.mid' % (basename, index))
            path.write_bytes(b'')
            assert list(samples.glob('out-%s-%d-*.mid' % (root.name, index))) == [path]
