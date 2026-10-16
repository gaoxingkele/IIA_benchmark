"""Source/API and sampling semantics; fixtures are not benchmark results."""
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from iia_benchmark.models import dsb_singlecell_source_components_v1 as s

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/fm_foundation_baseline_sources/sources/dsb/original'


def test_full_paper_network_and_both_original_encoder_activations():
    model = s.build_score_network(SOURCE)
    assert sum(p.numel() for p in model.parameters()) == 14261
    assert len(model.net.layers) == 4
    assert isinstance(model.net.activation_fn, torch.nn.SELU)
    assert isinstance(model.t_encoder.activation_fn, torch.nn.LeakyReLU)
    assert isinstance(model.x_encoder.activation_fn, torch.nn.LeakyReLU)
    x = torch.arange(2560, dtype=torch.float32).reshape(512, 5) / 2560
    loss = model(x, torch.full((512, 1), .01)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_original_empty_cache_failure_and_full10000_particle_replacement():
    x = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    assert len(DataLoader(TensorDataset(x), batch_size=10000, drop_last=True)) == 0
    loader = s.endpoint_cache_loader(x, 10000, 42)
    draws = next(iter(loader))[0]
    assert draws.shape == (10000, 5)
    assert len(loader.dataset) == 7 and len(loader) == 1
    assert torch.equal(torch.unique(draws, dim=0), x)
    assert torch.equal(draws, next(iter(s.endpoint_cache_loader(x, 10000, 42)))[0])
    assert not torch.equal(draws, next(iter(s.endpoint_cache_loader(x, 10000, 43)))[0])


@pytest.mark.parametrize('count,left', [(4, 1), (4, 2), (5, 1), (5, 2), (5, 3)])
def test_backward_orientation_excludes_target_and_uses_exact_midpoint(count, left):
    indices = s.endpoint_indices(count, left)
    assert {indices['source_initial_index'], indices['source_final_index']} == {left - 1, left + 1}
    assert left not in [indices['source_initial_index'], indices['source_final_index']]
    assert indices['backward_start_index'] == left - 1
    assert indices['midpoint_path_index'] == 9


@pytest.mark.parametrize('values', [torch.empty(0, 5), torch.zeros(2, 4), torch.full((2, 5), float('nan'))])
def test_invalid_endpoint_populations_cannot_be_silently_replaced(values):
    with pytest.raises(ValueError):
        s.endpoint_cache_loader(values, 10000, 42)
