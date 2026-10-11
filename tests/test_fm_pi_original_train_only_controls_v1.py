import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching import run_pi_transformer_job as source
from scripts.flow_matching.run_pi_original_train_only_controls_v1 import controls_ast, train_only_threshold


@pytest.mark.parametrize('reference,percentile', [(np.array([]), 99), (np.array([np.nan]), 99),
    (np.array([np.inf]), 99), (np.ones((2,2)), 99), (np.ones(2), -1), (np.ones(2), 101)])
def test_only_full_finite_training_vectors_and_fixed_percentiles_accepted(reference, percentile):
    with pytest.raises(ValueError): train_only_threshold(reference, percentile)


def test_threshold_has_no_test_data_or_labels_input():
    train = np.arange(10, dtype=np.float32)
    assert train_only_threshold(train, 90) == float(np.percentile(train, 90))
    assert list(train_only_threshold.__code__.co_varnames[:train_only_threshold.__code__.co_argcount]) == ['train', 'percentile']


@pytest.mark.parametrize('train_only', [True, False])
def test_original_full_normalization_softmax_temperature_metric_and_PA_AST_preserved(train_only):
    old = next(n for n in ast.parse(Path(source.__file__).read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'score_controls')
    class Undo(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'capture_full_case': return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'train_only_threshold':
                return ast.parse('np.percentile(np.concatenate([train, test]), 100 - anomaly_ratio)', mode='eval').body
            return self.generic_visit(node)
        def visit_Dict(self, node):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'test_scores_used_for_calibration': value.value = True
                if isinstance(key, ast.Constant) and key.value == 'fidelity': value.value = 'Posthoc CPU score control; original author result is recorded separately'
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(controls_ast(train_only)).body[0]) == ast.dump(old)


def values(test_shift=0, flip_labels=False):
    return dict(train_phase=np.array([1,2,3,4,5,6,7,8], dtype=np.float32),
        mismatch_test=np.array([1,2,3,4], dtype=np.float32) + test_shift,
        reconstruction_train=np.array([2,1,3,4,5,2,1,6], dtype=np.float32),
        reconstruction_test=np.array([3,1,4,7], dtype=np.float32) + test_shift,
        labels=np.array([0,1,0,1] if not flip_labels else [1,0,1,0]))


def transformed(data, train_only):
    captured = []
    namespace = dict(vars(source), train_only_threshold=train_only_threshold,
        capture_full_case=lambda factor, temperature, stream, train, test, labels, predicted: captured.append((factor, stream, train.copy(), test.copy())))
    exec(compile(controls_ast(train_only), source.__file__, 'exec'), namespace)
    result = namespace['score_controls'](data, 2, 10, 2, [.5, 1, 2], ['energy','mismatch','fused'])
    return result, captured


def test_changed_test_scores_and_labels_cannot_change_any_train_only_threshold():
    before, _ = transformed(values(), True)
    changed, _ = transformed(values(1000, True), True)
    assert len(before) == len(changed) == 9
    assert [r['threshold'] for r in before] == [r['threshold'] for r in changed]
    assert all(not r['test_scores_used_for_calibration'] and not r['strict_TAB_result'] for r in changed)
    pooled, _ = transformed(values(), False)
    changed_pool, _ = transformed(values(1000, True), False)
    assert [r['threshold'] for r in pooled] != [r['threshold'] for r in changed_pool]


def test_all_normalized_score_vectors_stay_bitwise_identical_when_only_calibration_changes():
    pooled, original = transformed(values(), False)
    adjusted, counterfactual = transformed(values(), True)
    assert len(pooled) == len(adjusted) == len(original) == len(counterfactual) == 9
    for old, new in zip(original, counterfactual):
        assert old[:2] == new[:2]
        assert np.array_equal(old[2], new[2]) and np.array_equal(old[3], new[3])
