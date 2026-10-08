import ast

import pytest

from iia_benchmark.models.fm_crossad_evaluator_compat import patch_evaluator, prepare_runtime


TEXT = '''from . import pate
class Evaluator:
    def evaluate(self, metrics, **metrics_args):
        results_storage = {}
        pate.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        return results_storage
'''


def test_optional_import_is_conditional_without_changing_requested_metrics():
    patched = patch_evaluator(TEXT)
    tree = ast.parse(patched)
    assert not any(isinstance(n, ast.ImportFrom) for n in tree.body)
    method = tree.body[0].body[0]
    gate = method.body[1]
    assert isinstance(gate, ast.If) and ast.unparse(gate.test) == "'pate' in metrics"
    original_call = ast.parse(TEXT).body[1].body[0].body[1].value
    assert ast.dump(gate.body[1].value) == ast.dump(original_call)


def test_disabled_pate_does_not_require_private_sklearn_dependency():
    namespace = {'__name__': 'broken_pate_package.evaluator', '__package__': 'broken_pate_package'}
    exec(patch_evaluator(TEXT), namespace)
    evaluator = namespace['Evaluator']()
    evaluator.gt, evaluator.anomaly_score = [0, 1], [.1, .9]
    assert evaluator.evaluate(['best_f1', 'auc', 'r_auc', 'vus']) == {}
    with pytest.raises(ImportError):
        evaluator.evaluate(['pate'])


def test_changed_source_anchor_is_rejected():
    with pytest.raises(ValueError, match='anchors'):
        patch_evaluator(TEXT.replace('from . import pate', 'from . import different'))


def test_runtime_copy_preserves_parent_and_rejects_overwrite(tmp_path):
    import hashlib
    import json
    parent, target = tmp_path / 'parent', tmp_path / 'new'
    path = parent / 'ts_ad_evaluation/evaluator.py'
    path.parent.mkdir(parents=True)
    path.write_text(TEXT, encoding='utf-8')
    original = path.read_bytes()
    (parent / 'runtime_manifest.json').write_text(json.dumps({'files': [
        {'path': 'ts_ad_evaluation/evaluator.py', 'runtime_sha256': hashlib.sha256(original).hexdigest()}]}))
    manifest = prepare_runtime(parent, target)
    assert path.read_bytes() == original and not manifest['algorithm_changed']
    assert prepare_runtime(parent, target) == manifest
    (target / 'ts_ad_evaluation/evaluator.py').write_text('user data must not be overwritten')
    with pytest.raises(ValueError, match='frozen differently'):
        prepare_runtime(parent, target)


def test_complete_compatibility_branch_preserves_all_original_experiments():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    read = lambda name: json.loads((root / 'configs/experiments' / name).read_text(encoding='utf-8'))
    old, new = read('fm_crossad_complete_queue.v6.json'), read('fm_crossad_complete_queue.v7.json')
    assert len(old['jobs']) == len(new['jobs']) == 139
    assert new['settings']['author_metrics'] == old['settings']['author_metrics']
    for before, after in zip(old['jobs'], new['jobs']):
        assert {k: v for k, v in before.items() if k not in ('output_directory', 'frozen_files')} == {
            k: v for k, v in after.items() if k not in ('output_directory', 'frozen_files')}
        assert before['output_directory'] != after['output_directory']
    for name in ('minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'):
        assert old['settings'][name] == new['settings'][name]
