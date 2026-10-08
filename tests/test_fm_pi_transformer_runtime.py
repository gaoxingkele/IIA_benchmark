from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from iia_benchmark.models.fm_pi_transformer_runtime import dfa_sample, prepare
from scripts.flow_matching.register_pi_transformer_author import recipes
from scripts.flow_matching.run_pi_transformer_job import author_pa, score_controls

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/fm_project_acquisition/sources/pi_transformer_author_history_mirror/original'


@pytest.mark.parametrize('count,batch', [(53, 7), (1103, 64)])
def test_streamed_dfa_keeps_exact_source_rng_and_selected_windows(tmp_path, count, batch):
    x = torch.arange(count * 12, dtype=torch.float32).reshape(count, 4, 3)
    loader = DataLoader(TensorDataset(x, torch.zeros(count)), batch_size=batch, shuffle=True)
    torch.manual_seed(42)
    original = torch.cat([inputs for inputs, _ in loader]).float()
    expected = original[torch.randperm(count)[:max(100, count // 10)]]
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    actual = dfa_sample(loader, 'cpu', tmp_path / 'windows.dat')
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not (tmp_path / 'windows.dat').exists()
    assert (tmp_path / 'windows.json').is_file()


def test_pinned_runtime_preserves_source_and_idempotence(tmp_path):
    before = (SOURCE / 'main.py').read_bytes()
    destination = tmp_path / 'source'
    manifest = prepare(SOURCE, destination, 'source_runtime')
    assert manifest == prepare(SOURCE, destination, 'source_runtime')
    generated = (destination / 'main.py').read_text()
    assert 'vali_loss1, vali_loss2 = self.validation(self.test_loader)' in generated
    assert 'loss1.backward(retain_graph=True)' in generated
    assert generated.count('self.optimiser.step()') == 1
    assert '- self.k * series_loss' in generated
    assert 'for j in range(i, 0, -1)' in generated
    assert 'scores.masked_fill(attn_mask.mask, -np.inf)' in (destination / 'model/attn.py').read_text()
    assert (SOURCE / 'main.py').read_bytes() == before
    (destination / 'main.py').write_text('changed')
    with pytest.raises(ValueError):
        prepare(SOURCE, destination, 'source_runtime')


def test_corrected_support_is_causal_for_both_attention_streams(tmp_path):
    destination = tmp_path / 'causal'
    prepare(SOURCE, destination, 'causal_axis_repaired')
    spec = importlib.util.spec_from_file_location('pi_test_attention', destination / 'model/attn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PhaseSyncAttention(10, output_attention=True)
    q = torch.randn(2, 10, 2, 4)
    projection = torch.zeros(2, 10, 2)
    _, series, prior, *_ = model(q, q, q, projection, projection, projection, None)
    future = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(series[:, :, future]) == 0
    assert torch.count_nonzero(prior[:, :, future]) == 0
    assert torch.allclose(series.sum(-1), torch.ones(2, 2, 10), atol=1e-6)
    assert torch.allclose(prior.sum(-1), torch.ones(2, 2, 10), atol=2e-6)


def test_all_training_axes_preserve_other_dataset_hyperparameters():
    import json
    import yaml
    settings = json.loads((ROOT / 'configs/experiments/fm_pi_transformer_author_execution.v1.json').read_text())
    original = yaml.safe_load((SOURCE / 'config.yaml').read_text())['datasets']['PSM']
    items = list(recipes(original, settings))
    assert len(items) == 19
    for name, config, changes in items:
        assert config['data']['win_size'] == 100
        assert config['model']['enc_in'] == 25
        assert config['testing'] == original['testing']
        assert config['training']['lr'] == original['training']['lr']
    assert original['model']['d_model'] == 512
    assert {r[1]['model']['d_model'] for r in items} == {128, 256, 512, 1024}


def test_original_pa_index_zero_boundary_is_retained():
    labels = np.array([1, 1, 1, 0, 1])
    prediction = np.array([0, 0, 1, 0, 1])
    assert np.array_equal(author_pa(labels, prediction), [0, 1, 1, 0, 1])


def test_score_control_grid_keeps_protocol_separate_and_all_axes():
    values = {'labels': np.array([0, 1, 0, 1]), 'train_phase': np.array([0.1, 1, 0.2, 2], np.float32),
              'mismatch_test': np.array([0.3, 3, 0.1, 4], np.float32),
              'reconstruction_train': np.array([0.3, 1, 0.2, 2], np.float32),
              'reconstruction_test': np.array([0.2, 2, 0.1, 3], np.float32)}
    records = score_controls(values, 10, 1, 2, [.25, .5, 1, 2, 4], ['energy', 'mismatch', 'fused'])
    assert len(records) == 15
    assert all(not r['strict_TAB_result'] and r['test_scores_used_for_calibration'] for r in records)
    assert all(np.isfinite(r['threshold']) for r in records)
