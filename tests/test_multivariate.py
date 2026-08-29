import numpy as np
import pytest

from iia_benchmark.models import (
    ConvexHullNOZAlarm,
    MahalanobisAlarm,
    convex_hull_fitness_index,
)


def test_multivariate_models_flag_far_outlier() -> None:
    rng = np.random.default_rng(4)
    normal = rng.normal(0, 0.25, (250, 3))
    points = np.vstack([normal[:5], [5.0, 5.0, 5.0]])
    for model in (MahalanobisAlarm(quantile=0.98), ConvexHullNOZAlarm(0.02)):
        prediction = model.fit(normal).predict(points)
        assert prediction[-1] == 1


def test_convex_hull_dynamic_bounds_contain_center() -> None:
    rng = np.random.default_rng(9)
    normal = rng.normal(size=(300, 3))
    model = ConvexHullNOZAlarm(0.01).fit(normal)
    bounds = model.dynamic_bounds(normal.mean(axis=0))
    assert bounds.shape == (3, 2)
    assert np.all(bounds[:, 0] <= normal.mean(axis=0))
    assert np.all(normal.mean(axis=0) <= bounds[:, 1])


def test_book_figure_3_2_convex_fitness_is_nine_thirteenths() -> None:
    points = np.asarray(
        [
            [0.0, -0.4],
            [-0.2, -0.2],
            [0.0, -0.2],
            [0.2, -0.2],
            [-0.4, 0.0],
            [0.4, 0.0],
            [-0.2, 0.2],
            [0.2, 0.2],
            [0.0, 0.4],
        ]
    )
    result = convex_hull_fitness_index(
        points,
        [0.2, 0.2],
        lower=[-0.5, -0.5],
        upper=[0.5, 0.5],
    )
    assert result.inside_points == 13
    assert result.counting_points == 9
    assert result.fitness == pytest.approx(9.0 / 13.0)


def test_outside_dynamic_bounds_use_closest_normal_projection() -> None:
    rng = np.random.default_rng(17)
    normal = rng.normal(size=(200, 2))
    model = ConvexHullNOZAlarm(0.01).fit(normal)
    outside = np.asarray([10.0, -8.0])
    projected = model.closest_normal_point(outside)
    assert model.predict([projected])[0] == 0
    bounds = model.dynamic_bounds(outside)
    assert np.all(bounds[:, 0] <= projected + 1e-8)
    assert np.all(projected <= bounds[:, 1] + 1e-8)
