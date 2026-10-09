import math
import numpy as np
import pytest
import torch

from iia_benchmark.models.grasp_protocol_v2 import (
    gaussian_distance_graph, fit_only_scaler, SeriesWindows,
    GNNVelocity, TransformerVelocity, GRASPProtocolDetector,
)


def test_distance_kernel_is_paper_gaussian_not_pearson():
    values = np.array([[0., 0., 10.], [1., 2., 12.], [2., 4., 15.]])
    a, audit = gaussian_distance_graph(values)
    distance = np.array([[np.linalg.norm(values[:, i] - values[:, j]) for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(audit['distance'], distance)
    np.testing.assert_allclose(audit['kernel'], np.exp(-(distance / distance.std())**2))
    expected = (audit['kernel'] >= .5).astype(float)
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(a, expected)
    assert not a[0, 2]  # Pearson is strongly correlated despite very different levels.


def test_degenerate_graph_and_validation_do_not_change_scaler():
    a, info = gaussian_distance_graph(np.ones((10, 4)))
    np.testing.assert_array_equal(a, np.ones((4, 4)) - np.eye(4))
    assert info['scale'] == 0
    train = np.arange(40).reshape(10, 4).astype(float)
    train[0, 0] = np.nan
    _, _, s1 = fit_only_scaler(train, np.ones((8, 4)))
    _, _, s2 = fit_only_scaler(train, np.full((8, 4), 1e7))
    for key in ('median', 'mean', 'scale'):
        np.testing.assert_array_equal(s1[key], s2[key])


def test_stride_one_windows_have_exact_boundary_coverage():
    x = np.arange(30).reshape(10, 3)
    d = SeriesWindows(x, 4)
    assert len(d) == 7
    np.testing.assert_array_equal(d[0].numpy(), x[:4].T)
    np.testing.assert_array_equal(d[6].numpy(), x[-4:].T)
    with pytest.raises(IndexError):
        d[7]


@pytest.mark.parametrize('architecture', ['gnn', 'transformer'])
def test_paper_architecture_axes_and_gradients(architecture):
    adjacency = np.ones((5, 5)) - np.eye(5)
    if architecture == 'gnn':
        model = GNNVelocity(adjacency)
        assert len(model.blocks) == 2
        assert model.input.out_features == 128
    else:
        model = TransformerVelocity(5)
        assert len(model.blocks) == 2
        assert all(b.self_attn.num_heads == 4 for b in model.blocks)
        assert model.input.out_features == 128
    x = torch.randn(2, 5, 50, requires_grad=True)
    score = model(torch.tensor([.2, .8]), x)
    assert score.shape == x.shape
    score.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize('architecture', ['tsmixer', 'gnn', 'transformer'])
def test_validation_budget_full_updates_and_chunk_independent_scores(architecture):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    train, validation, test = (rng.normal(size=(n, 5)) for n in (15, 10, 13))
    model = GRASPProtocolDetector(window=4, hidden=8, epochs=3, validation_interval=2, batch_size=4,
                                 architecture=architecture, dropout=0, source_samples=2, flow_evaluations=2, device='cpu')
    model.fit(train, validation)
    assert model.epochs_completed_ == 3
    assert model.optimizer_updates_ == 3 * math.ceil((len(train) - 4 + 1) / 4)
    assert [r['epoch'] for r in model.history_ if 'validation_loss' in r] == [2, 3]
    score1 = model.score(test, batch_size=2, split_tag=2)
    score2 = model.score(test, batch_size=5, split_tag=2)
    assert len(score1) == len(test) and np.isfinite(score1).all()
    np.testing.assert_allclose(score1, score2, rtol=2e-6, atol=1e-5)
    assert not np.array_equal(score1, model.score(test, split_tag=3))


def test_validation_values_cannot_select_training_weights():
    torch.set_num_threads(1)
    train = np.random.default_rng(1).normal(size=(14, 5))
    params = dict(window=4, hidden=8, epochs=2, validation_interval=1, batch_size=4, dropout=.1, device='cpu')
    a = GRASPProtocolDetector(**params).fit(train, np.ones((8, 5)))
    b = GRASPProtocolDetector(**params).fit(train, np.full((8, 5), 100.))
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key])
    assert a.history_[-1]['validation_loss'] != b.history_[-1]['validation_loss']


def test_mechanism_ablations_preserve_paper_unweighted_er_and_lin():
    train = np.random.default_rng(1).normal(size=(30, 5))
    train[:, 1] = train[:, 0] + .01
    train[:, 4] = train[:, 3] + .01
    plain = GRASPProtocolDetector(window=4, variant='mean')
    er = GRASPProtocolDetector(window=4, variant='er')
    lin = GRASPProtocolDetector(window=4, variant='lin')
    for m in (plain, er, lin):
        m.prepare(train, train[:8])
        m._build()
    assert np.count_nonzero(plain.adjacency_) == np.count_nonzero(er.adjacency_)
    assert torch.count_nonzero(lin.path.omega) == 0
    assert plain.variant != 'grasp' and er.variant != 'grasp' and lin.variant != 'grasp'


def test_full_attribution_sums_match_window_scores_and_eight_quantile_heatmaps():
    torch.set_num_threads(1)
    rng = np.random.default_rng(44)
    train, validation, test = [rng.normal(size=(n,9)) for n in [14,8,12]]
    model = GRASPProtocolDetector(window=4,hidden=8,epochs=1,validation_interval=1,batch_size=4,
                                 dropout=0,source_samples=2,flow_evaluations=3,device='cpu').fit(train,validation)
    points,details = model.score(test,split_tag=2,return_details=True)
    expected = np.zeros(len(test))
    counts = np.zeros(len(test))
    for i,score in enumerate(details['window_scores']):
        expected[i:i+4]+=score
        counts[i:i+4]+=1
    np.testing.assert_allclose(points,expected/counts)
    np.testing.assert_allclose(details['mode_energy_weighted'].sum(),details['window_scores'].sum(),rtol=2e-6)
    assert details['quantile_heatmap_raw_percent'].shape==(3,8)
    assert details['quantile_heatmap_raw_percent'].sum()==pytest.approx(100)
    assert details['quantile_heatmap_weighted_percent'].sum()==pytest.approx(100)
