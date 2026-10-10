import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.audit_giflow_checkpoint_replay_v1 import ROOT,native_test_function,compare_vectors


def test_extract_complete_actual_native_sampler_without_training_or_EMA_reconstruction():
    source=ROOT/'experiments/runs/flow_matching_campaign/sources/giflow/original/main.py'
    function=native_test_function(source.read_text(encoding='utf-8'))
    text=ast.unparse(function)
    assert 'num_steps = 20' in text and 'delta_t = 1 / num_steps' in text
    assert 'for batch in test_loader:' in text and 'model.eval()' in text
    assert 'losses_mae' in text and 'nonzero_mask' in text and 'inverse_transform' in text
    assert 'ema.apply_shadow' not in text and 'optimizer' not in text
    assert 'ema.restore()' in text


def test_reject_sampler_step_reduction():
    source=(ROOT/'experiments/runs/flow_matching_campaign/sources/giflow/original/main.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError):native_test_function(source.replace('num_steps = 20','num_steps = 19'))


def test_prediction_agreement_cannot_hide_missing_nonfinite_or_different_targets():
    tolerance=dict(rtol=2e-4,atol=1e-3)
    assert compare_vectors(np.array([1.,2.]),np.array([1.,2.]),tolerance)['bitwise_identical']
    assert not compare_vectors(np.array([1.0001,2.]),np.array([1.,2.]),tolerance)['bitwise_identical']
    for actual in [np.array([1.]),np.array([1.,np.nan]),np.array([1.,2.1])]:
        with pytest.raises(ValueError):compare_vectors(actual,np.array([1.,2.]),tolerance)
