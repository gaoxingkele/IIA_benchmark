import torch
import pytest

from iia_benchmark.models.fm_image_evaluation import paired_probes, probe_seed, score_original_image


class ExactGeometryFixture(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__(); self.velocity = velocity; self.calls = []

    def forward(self, images, time, labels=None):
        self.calls.append((time.detach().clone(), None if labels is None else labels.clone()))
        return self.velocity.expand_as(images)


def parameters(aggregation='minimum', percentile=.1, weighted=True):
    return dict(paths=10, times=10, aggregation=aggregation, percentile=percentile,
                weighted=weighted, top_fraction=.01)


def test_probe_prefix_is_exact_across_ablation_order_and_path_counts():
    image = torch.zeros(1, 3, 4, 4)
    small, ids = paired_probes(image, training_seed=1103, dataset='visa', native_image_index=9, paths=5)
    large, more = paired_probes(image, training_seed=1103, dataset='visa', native_image_index=9, paths=20)
    assert torch.equal(small, large[:5]) and ids == more[:5]
    assert probe_seed(1103, 'visa', 9, 0) != probe_seed(1104, 'visa', 9, 0)
    assert probe_seed(1103, 'visa', 9, 0) != probe_seed(1103, 'visa', 10, 0)
    assert probe_seed(1103, 'visa', 9, 0) != probe_seed(1103, 'mvtec', 9, 0)


def test_percentile_point_one_k10_is_exactly_minimum_with_paired_draws():
    image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 10
    model = ExactGeometryFixture(torch.zeros(1, 3, 4, 4)).eval()
    kwargs = dict(labels=torch.tensor([2]), training_seed=1103, dataset='visa', native_image_index=5)
    minimum = score_original_image(model, image, parameters=parameters(), **kwargs)
    percentile = score_original_image(model, image, parameters=parameters('percentile', .1), **kwargs)
    assert torch.equal(minimum[0], percentile[0]) and torch.equal(minimum[1], percentile[1])
    assert len(model.calls) == 20
    for time, labels in model.calls:
        assert labels.tolist() == [2] * 10
        assert torch.all((time > 0) & (time < 1))


def test_score_matches_independent_nested_channel_time_path_equations():
    image = torch.arange(48, dtype=torch.float64).reshape(1, 3, 4, 4) / 7
    model = ExactGeometryFixture(torch.full_like(image, .3)).eval()
    config = parameters('percentile', .3)
    actual = score_original_image(model, image, labels=None, parameters=config,
        training_seed=1103, dataset='mvtec', native_image_index=0)
    probes, _ = paired_probes(image, training_seed=1103, dataset='mvtec', native_image_index=0, paths=10)
    errors = torch.stack([sum((.3 - (image[0, channel] - noise[0, channel])) ** 2
                             for channel in range(3)) for noise in probes])
    selected = errors.sort(dim=0).values[2]
    weight = sum((j / 11) ** 2 for j in range(1, 11)) / 10
    expected = selected * weight
    assert torch.allclose(actual[0][0], expected, rtol=1e-12, atol=1e-12)
    assert actual[1][0] == pytest.approx(float(expected.max() * 2), abs=1e-12)


def test_training_mode_and_missing_native_image_index_rejected():
    image = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError):
        score_original_image(ExactGeometryFixture(image), image, labels=None, parameters=parameters(),
                              training_seed=1103, dataset='visa', native_image_index=0)
    with pytest.raises(ValueError): probe_seed(1103, 'visa', -1, 0)
