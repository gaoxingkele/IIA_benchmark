import numpy as np

from iia_benchmark.models import ConvexHullNOZAlarm, MahalanobisAlarm


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
