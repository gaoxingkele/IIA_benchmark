"""Check adapter equality and checkpoint retention; fixtures are not benchmarks."""
import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_legacy_window_adapter import LegacyWindowAdapter
from iia_benchmark.models.mtsad_detectors import build_detector


@pytest.mark.parametrize('method,parameters', [
    ('timesnet', {'window': 12, 'd_model': 4, 'd_ff': 4, 'e_layers': 1, 'top_k': 2, 'num_kernels': 2, 'dropout': 0., 'batch_size': 2}),
    ('usad', {'window': 12, 'hidden_size': 8, 'latent_size': 4, 'batch_size': 2, 'adversarial': False}),
])
def test_adapter_keeps_scores_weights_and_training_budget(method, parameters):
    torch.set_num_threads(1)
    values = np.random.default_rng(15).normal(size=(4, 12, 2)).astype('float32')
    original = build_detector(method, {**parameters, 'epochs': 2, 'seed': 1103, 'device': 'cpu'}).fit(values)
    adapter = LegacyWindowAdapter(method, parameters, 2, 1103, 'cpu').fit(values)
    np.testing.assert_allclose(adapter.score(values), original.score(values), rtol=1e-6, atol=1e-7)
    assert isinstance(adapter._model, torch.nn.Module)
    assert adapter._model.state_dict()
    assert adapter._delegate.epochs == 2


def test_adapter_does_not_substitute_a_point_detector():
    with pytest.raises(ValueError, match='complete-score window'):
        # Point detector constructor cannot consume window-specific settings.
        LegacyWindowAdapter('pca', {}, 1).fit(np.ones((2, 12, 2)))
