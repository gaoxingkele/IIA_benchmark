"""Independent CASIM reproduction for industrial alarm-flood classification.

The implementation follows Faulwasser and Fay (2024), Eqs. (1)--(10): two
alarm-series representations, length-nine MultiRocket kernels, four pooling
operators, an ensemble of calibrated ridge classifiers, and LoOP novelty
detection on the complete posterior vector plus its top-two probability gap.

This is not copied from the authors' Code Ocean artifact.  The artifact is
versioned in the literature registry but was not downloadable in the current
environment, so paper-score equivalence remains a separate validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class CASIMKernel:
    """One fitted multivariate MiniRocket/MultiRocket kernel."""

    weights: np.ndarray
    dilation: int
    padding: bool
    channels: tuple[int, ...]
    biases: tuple[float, float]


def _check_alarm_tensor(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 3:
        raise ValueError("X must have shape (episodes, alarm_variables, time)")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] < 9:
        raise ValueError("X needs at least one episode/channel and nine time samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain only finite values")
    return values


def _canonical_kernels() -> tuple[np.ndarray, ...]:
    """Return the 84 permutations with three weights 2 and six weights -1."""

    result: list[np.ndarray] = []
    for positive_positions in combinations(range(9), 3):
        weights = -np.ones(9, dtype=float)
        weights[list(positive_positions)] = 2.0
        result.append(weights)
    return tuple(result)


def _convolution(
    episode: np.ndarray,
    weights: np.ndarray,
    dilation: int,
    channels: tuple[int, ...],
    padding: bool,
) -> np.ndarray:
    signal = np.sum(episode[np.asarray(channels, dtype=int)], axis=0)
    receptive_field = (len(weights) - 1) * dilation + 1
    if padding:
        amount = receptive_field // 2
        signal = np.pad(signal, (amount, amount), mode="constant")
    output_length = signal.size - receptive_field + 1
    if output_length <= 0:
        return np.empty(0, dtype=float)
    offsets = np.arange(len(weights)) * dilation
    indices = np.arange(output_length)[:, None] + offsets[None, :]
    return signal[indices] @ weights


def _pool_positive(values: np.ndarray) -> np.ndarray:
    positive = values > 0
    count = int(np.sum(positive))
    if count == 0:
        return np.array([0.0, 0.0, -1.0, 0.0])
    runs = np.diff(np.concatenate(([0], positive.astype(int), [0])))
    starts = np.flatnonzero(runs == 1)
    stops = np.flatnonzero(runs == -1)
    return np.array(
        [
            count / values.size,
            float(np.mean(values[positive])),
            float(np.mean(np.flatnonzero(positive))),
            float(np.max(stops - starts)),
        ]
    )


class CASIMFeatureTransformer:
    """Paper-faithful 8-features-per-kernel alarm MultiRocket transform.

    A bias is represented as the negative selected convolution quantile because
    Eq. (2) adds ``b`` before applying the positive-value pooling operators.
    """

    def __init__(self, n_features: int = 672, random_state: int = 0) -> None:
        if n_features < 8 or n_features % 8:
            raise ValueError("n_features must be a positive multiple of eight")
        self.n_features = int(n_features)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray) -> "CASIMFeatureTransformer":
        values = _check_alarm_tensor(X)
        rng = np.random.default_rng(self.random_state)
        n_kernels = self.n_features // 8
        canonical = _canonical_kernels()
        max_channels = min(9, values.shape[1])
        representations = (values, np.diff(values, axis=2))
        specs: list[CASIMKernel] = []

        # The irrational rotation supplies the low-discrepancy quantiles used
        # by the Rocket family while remaining exactly reproducible.
        quantiles = np.mod((np.arange(n_kernels) + 1) * ((sqrt(5.0) - 1.0) / 2.0), 1.0)
        for index in range(n_kernels):
            weights = canonical[index % len(canonical)].copy()
            n = values.shape[2] if index % 2 == 0 else values.shape[2] - 1
            max_power = max(0, int(np.floor(np.log2(max(1.0, n / 8.0)))))
            dilation = 2 ** (index % (max_power + 1))
            padding = index % 2 == 0
            channel_count = int(rng.integers(1, max_channels + 1))
            channels = tuple(
                sorted(int(item) for item in rng.choice(values.shape[1], channel_count, replace=False))
            )
            biases: list[float] = []
            for representation in representations:
                sample = representation[int(rng.integers(representation.shape[0]))]
                raw = _convolution(sample, weights, dilation, channels, padding)
                if raw.size == 0:
                    raise ValueError("time dimension is too short for the selected dilation")
                biases.append(-float(np.quantile(raw, quantiles[index])))
            specs.append(CASIMKernel(weights, dilation, padding, channels, tuple(biases)))

        self.kernels_ = tuple(specs)
        raw_features = self._transform_raw(values)
        scale = np.std(raw_features, axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
        self.scale_ = scale
        self.n_channels_in_ = values.shape[1]
        self.n_time_in_ = values.shape[2]
        return self

    def _transform_raw(self, X: np.ndarray) -> np.ndarray:
        representations = (X, np.diff(X, axis=2))
        output = np.empty((X.shape[0], self.n_features), dtype=float)
        for row in range(X.shape[0]):
            cursor = 0
            for kernel in self.kernels_:
                for representation_index, representation in enumerate(representations):
                    z = _convolution(
                        representation[row],
                        kernel.weights,
                        kernel.dilation,
                        kernel.channels,
                        kernel.padding,
                    )
                    output[row, cursor : cursor + 4] = _pool_positive(
                        z + kernel.biases[representation_index]
                    )
                    cursor += 4
        return output

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "kernels_"):
            raise RuntimeError("fit must be called before transform")
        values = _check_alarm_tensor(X)
        if values.shape[1:] != (self.n_channels_in_, self.n_time_in_):
            raise ValueError("CASIM requires the fitted channel count and zero-padded window length")
        return self._transform_raw(values) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def casim_loop_features(probabilities: np.ndarray) -> np.ndarray:
    """Build LoOP inputs from all class probabilities and Eq. (9)'s margin."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must be a 2-D array with at least two classes")
    if np.any(probs < 0) or not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6):
        raise ValueError("each posterior vector must be nonnegative and sum to one")
    ordered = np.sort(probs, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return np.column_stack((probs, margin))


def _probability_smote(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_count: int,
    k_loop: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    generated = [np.asarray(probabilities, dtype=float)]
    generated_labels = [np.asarray(labels)]
    for label in np.unique(labels):
        group = probabilities[labels == label]
        if group.shape[0] >= target_count:
            continue
        synthetic: list[np.ndarray] = []
        while group.shape[0] + len(synthetic) < target_count:
            source_index = int(rng.integers(group.shape[0]))
            source = group[source_index]
            if group.shape[0] == 1:
                neighbor = source
            else:
                distances = np.linalg.norm(group - source, axis=1)
                candidates = np.argsort(distances)[1 : 1 + min(max(1, k_loop // 2), group.shape[0] - 1)]
                neighbor = group[int(rng.choice(candidates))]
            point = source + rng.random() * (neighbor - source)
            point = np.maximum(point, 0.0)
            total = float(np.sum(point))
            synthetic.append(point / total if total > 0 else source)
        generated.append(np.asarray(synthetic))
        generated_labels.append(np.repeat(label, len(synthetic)))
    return np.vstack(generated), np.concatenate(generated_labels)


class LocalOutlierProbability:
    """LoOP novelty detector of Kriegel et al. (2009)."""

    def __init__(self, n_neighbors: int = 10, extent: float = 3.0) -> None:
        if n_neighbors < 1 or extent <= 0:
            raise ValueError("n_neighbors and extent must be positive")
        self.n_neighbors = int(n_neighbors)
        self.extent = float(extent)

    @staticmethod
    def _distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2))

    def fit(self, X: np.ndarray) -> "LocalOutlierProbability":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("LoOP needs at least two 2-D training samples")
        self.X_ = values
        self.k_ = min(self.n_neighbors, values.shape[0] - 1)
        distances = self._distances(values, values)
        np.fill_diagonal(distances, np.inf)
        neighbors = np.argpartition(distances, self.k_ - 1, axis=1)[:, : self.k_]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        self.pdist_ = self.extent * np.sqrt(np.mean(neighbor_distances**2, axis=1))
        expected_pdist = np.mean(self.pdist_[neighbors], axis=1)
        plof = self.pdist_ / np.maximum(expected_pdist, np.finfo(float).eps) - 1.0
        self.nplof_ = self.extent * sqrt(float(np.mean(plof**2)))
        self.nplof_ = max(self.nplof_, np.finfo(float).eps)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_"):
            raise RuntimeError("fit must be called before predict_proba")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.X_.shape[1]:
            raise ValueError("X has an incompatible LoOP feature shape")
        distances = self._distances(values, self.X_)
        neighbors = np.argpartition(distances, self.k_ - 1, axis=1)[:, : self.k_]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        pdist = self.extent * np.sqrt(np.mean(neighbor_distances**2, axis=1))
        expected_pdist = np.mean(self.pdist_[neighbors], axis=1)
        plof = pdist / np.maximum(expected_pdist, np.finfo(float).eps) - 1.0
        scores = np.array([max(0.0, erf(value / (self.nplof_ * sqrt(2.0)))) for value in plof])
        return np.clip(scores, 0.0, 1.0)


class CASIMClassifier:
    """CASIM ridge ensemble with Platt calibration and LoOP rejection."""

    def __init__(
        self,
        n_features: int = 672,
        n_classifiers: int = 10,
        alphas: tuple[float, ...] | None = None,
        k_loop: int = 10,
        loop_extent: float = 3.0,
        random_state: int = 0,
    ) -> None:
        if n_classifiers < 1:
            raise ValueError("n_classifiers must be positive")
        self.n_features = int(n_features)
        self.n_classifiers = int(n_classifiers)
        self.alphas = tuple(np.logspace(-3, 3, 10)) if alphas is None else alphas
        self.k_loop = int(k_loop)
        self.loop_extent = float(loop_extent)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CASIMClassifier":
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
            from sklearn.model_selection import StratifiedKFold
        except ImportError as exc:  # pragma: no cover - exercised in minimal installations
            raise ImportError("CASIMClassifier requires the project's 'ml' extra") from exc

        values = _check_alarm_tensor(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
            raise ValueError("y must contain one label per episode")
        self.classes_, counts = np.unique(labels, return_counts=True)
        if self.classes_.size < 2 or np.min(counts) < 2:
            raise ValueError("CASIM needs at least two classes and two samples per class")
        calibration_folds = min(5, int(np.min(counts)))
        splitter = StratifiedKFold(calibration_folds, shuffle=True, random_state=self.random_state)

        transformers: list[CASIMFeatureTransformer] = []
        classifiers: list[CalibratedClassifierCV] = []
        for ensemble_index in range(self.n_classifiers):
            seed = self.random_state + ensemble_index
            transformer = CASIMFeatureTransformer(self.n_features, seed)
            features = transformer.fit_transform(values)
            selector = RidgeClassifierCV(alphas=np.asarray(self.alphas, dtype=float), cv=None)
            selector.fit(features, labels)
            classifier = CalibratedClassifierCV(
                RidgeClassifier(alpha=float(selector.alpha_)),
                method="sigmoid",
                cv=splitter,
            )
            classifier.fit(features, labels)
            transformers.append(transformer)
            classifiers.append(classifier)
        self.transformers_ = tuple(transformers)
        self.classifiers_ = tuple(classifiers)

        probabilities = self.predict_proba(values)
        correct = self.classes_[np.argmax(probabilities, axis=1)] == labels
        if np.sum(correct) < 2:
            raise RuntimeError("fewer than two correctly classified samples are available for LoOP")
        rng = np.random.default_rng(self.random_state)
        balanced_probs, balanced_labels = _probability_smote(
            probabilities[correct], labels[correct], self.k_loop + 1, self.k_loop, rng
        )
        self.loop_training_labels_ = balanced_labels
        self.loop_ = LocalOutlierProbability(self.k_loop, self.loop_extent).fit(
            casim_loop_features(balanced_probs)
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classifiers_"):
            raise RuntimeError("fit must be called before predict_proba")
        values = _check_alarm_tensor(X)
        merged = np.zeros((values.shape[0], self.classes_.size), dtype=float)
        for transformer, classifier in zip(self.transformers_, self.classifiers_):
            probabilities = classifier.predict_proba(transformer.transform(values))
            positions = {label: index for index, label in enumerate(classifier.classes_)}
            merged += probabilities[:, [positions[label] for label in self.classes_]]
        return merged / self.n_classifiers

    def outlier_probability(self, X: np.ndarray) -> np.ndarray:
        return self.loop_.predict_proba(casim_loop_features(self.predict_proba(X)))

    def predict(self, X: np.ndarray, novelty_threshold: float = 0.5) -> np.ndarray:
        if not 0 <= novelty_threshold <= 1:
            raise ValueError("novelty_threshold must be between zero and one")
        probabilities = self.predict_proba(X)
        labels = self.classes_[np.argmax(probabilities, axis=1)].astype(object)
        labels[self.outlier_probability(X) >= novelty_threshold] = -1
        return labels


class CASIMExpandingWindowClassifier:
    """Train one CASIM stage for every requested expanding-window prefix."""

    def __init__(self, window_lengths: tuple[int, ...], **casim_parameters: object) -> None:
        if not window_lengths or any(length < 9 for length in window_lengths):
            raise ValueError("window_lengths must contain values of at least nine")
        self.window_lengths = tuple(sorted(set(int(item) for item in window_lengths)))
        self.casim_parameters = casim_parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CASIMExpandingWindowClassifier":
        values = _check_alarm_tensor(X)
        if self.window_lengths[-1] > values.shape[2]:
            raise ValueError("the largest window exceeds the available episode length")
        self.models_ = {
            length: CASIMClassifier(**self.casim_parameters).fit(values[:, :, :length], y)
            for length in self.window_lengths
        }
        return self

    def predict_evolution(
        self, X: np.ndarray, novelty_threshold: float = 0.5
    ) -> dict[int, np.ndarray]:
        values = _check_alarm_tensor(X)
        return {
            length: model.predict(values[:, :, :length], novelty_threshold)
            for length, model in self.models_.items()
        }
