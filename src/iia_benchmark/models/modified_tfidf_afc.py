"""Modified TF-IDF and LSTM pipeline for online alarm-flood analysis.

Rahaman, Alinezhad, and Chen (2025) expose a position-weighted n-gram TF-IDF,
spectral clustering, kernel-PCA fault isolation, an LSTM classifier, and an
optimized alert threshold.  The full equations and VAM data are access
controlled, so position decay and model hyperparameters remain explicit local
choices rather than paper-score claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Hashable, Sequence

import numpy as np


Token = Hashable
NGram = tuple[Token, ...]


@dataclass(frozen=True)
class NGramSelection:
    ngram_size: int
    silhouette: float
    scores: dict[int, float]


@dataclass(frozen=True)
class AlertThresholdResult:
    threshold: float
    balanced_accuracy: float
    false_alarm_rate: float
    miss_rate: float


class ModifiedTFIDFVectorizer:
    """Position-weighted alarm n-gram TF-IDF representation.

    Earlier n-grams receive weight ``exp(-position_decay * relative_position)``.
    This is an auditable realization of the paper's confirmed position-based
    weighting; the exact publisher equation remains a validation gate.
    """

    def __init__(self, ngram_size: int = 2, position_decay: float = 1.0) -> None:
        if ngram_size <= 0 or position_decay < 0:
            raise ValueError("ngram_size must be positive and position_decay nonnegative")
        self.ngram_size = int(ngram_size)
        self.position_decay = float(position_decay)

    def _ngrams(self, sequence: Sequence[Token]) -> tuple[NGram, ...]:
        tokens = tuple(sequence)
        if len(tokens) < self.ngram_size:
            return ()
        return tuple(
            tokens[index : index + self.ngram_size]
            for index in range(len(tokens) - self.ngram_size + 1)
        )

    def fit(self, sequences: Sequence[Sequence[Token]]) -> "ModifiedTFIDFVectorizer":
        documents = [self._ngrams(sequence) for sequence in sequences]
        if not documents:
            raise ValueError("at least one alarm sequence is required")
        vocabulary = sorted(set(chain.from_iterable(documents)), key=repr)
        if not vocabulary:
            raise ValueError("sequences are shorter than ngram_size")
        self.vocabulary_ = tuple(vocabulary)
        self.vocabulary_index_ = {term: index for index, term in enumerate(vocabulary)}
        document_frequency = np.asarray(
            [sum(term in set(document) for document in documents) for term in vocabulary],
            dtype=float,
        )
        self.idf_ = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
        return self

    def transform(self, sequences: Sequence[Sequence[Token]]) -> np.ndarray:
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("fit must be called before transform")
        output = np.zeros((len(sequences), len(self.vocabulary_)), dtype=float)
        for row_index, sequence in enumerate(sequences):
            terms = self._ngrams(sequence)
            if not terms:
                continue
            relative = np.arange(len(terms), dtype=float) / max(1, len(terms) - 1)
            position_weights = np.exp(-self.position_decay * relative)
            normalizer = float(np.sum(position_weights))
            for term, weight in zip(terms, position_weights):
                column = self.vocabulary_index_.get(term)
                if column is not None:
                    output[row_index, column] += weight / normalizer
        return output * self.idf_[None, :]

    def fit_transform(self, sequences: Sequence[Sequence[Token]]) -> np.ndarray:
        return self.fit(sequences).transform(sequences)


class SpectralAlarmFloodClusterer:
    """Spectral clustering over cosine affinity between TF-IDF flood vectors."""

    def __init__(self, n_clusters: int, random_state: int = 0) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least two")
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)

    @staticmethod
    def affinity(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        affinity = np.clip(normalized @ normalized.T, 0.0, 1.0)
        # A negligible floor keeps the graph connected and makes the spectral
        # decomposition deterministic even for perfectly separated corpora.
        affinity = np.maximum(affinity, np.finfo(float).eps)
        np.fill_diagonal(affinity, 1.0)
        return affinity

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("install iia-benchmark[ml] for spectral clustering") from exc
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] < self.n_clusters:
            raise ValueError("X must be a matrix with at least n_clusters rows")
        self.labels_ = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=self.random_state,
            n_init=20,
        ).fit_predict(self.affinity(values))
        return self.labels_.copy()


def optimize_ngram_size(
    sequences: Sequence[Sequence[Token]],
    candidates: Sequence[int],
    n_clusters: int,
    position_decay: float = 1.0,
    random_state: int = 0,
) -> NGramSelection:
    """Select n by the unsupervised silhouette of paper-style spectral clusters."""

    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("install iia-benchmark[ml] for n-gram optimization") from exc
    unique = tuple(sorted(set(int(item) for item in candidates)))
    if not unique or unique[0] <= 0:
        raise ValueError("candidates must contain positive n-gram sizes")
    scores: dict[int, float] = {}
    for ngram_size in unique:
        try:
            features = ModifiedTFIDFVectorizer(ngram_size, position_decay).fit_transform(sequences)
            labels = SpectralAlarmFloodClusterer(n_clusters, random_state).fit_predict(features)
            if len(set(labels.tolist())) < 2:
                score = -1.0
            else:
                score = float(silhouette_score(features, labels, metric="cosine"))
        except ValueError:
            score = -1.0
        scores[ngram_size] = score
    selected = max(unique, key=lambda item: (scores[item], -item))
    return NGramSelection(selected, scores[selected], scores)


def optimize_alert_threshold(scores: np.ndarray, abnormal: np.ndarray) -> AlertThresholdResult:
    """Choose a risk threshold by balanced accuracy on a frozen calibration set."""

    values = np.asarray(scores, dtype=float)
    labels = np.asarray(abnormal, dtype=bool)
    if values.ndim != 1 or labels.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("scores and abnormal must be equal finite vectors")
    if not np.any(labels) or np.all(labels):
        raise ValueError("threshold calibration needs normal and abnormal examples")
    candidates = np.unique(
        np.concatenate(
            ([np.nextafter(values.min(), -np.inf)], values, [np.nextafter(values.max(), np.inf)])
        )
    )
    best: AlertThresholdResult | None = None
    for threshold in candidates:
        predicted = values >= threshold
        false_alarm_rate = float(np.mean(predicted[~labels]))
        miss_rate = float(np.mean(~predicted[labels]))
        balanced = 1.0 - 0.5 * (false_alarm_rate + miss_rate)
        result = AlertThresholdResult(float(threshold), balanced, false_alarm_rate, miss_rate)
        if best is None or (result.balanced_accuracy, -result.false_alarm_rate) > (
            best.balanced_accuracy,
            -best.false_alarm_rate,
        ):
            best = result
    assert best is not None
    return best


class KernelPCAFaultIsolator:
    """Kernel-PCA reconstruction risk with calibrated early-alert threshold."""

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: float | None = None,
        normal_quantile: float = 0.99,
    ) -> None:
        if n_components <= 0 or not 0 < normal_quantile < 1:
            raise ValueError("invalid kernel PCA component count or quantile")
        self.n_components = int(n_components)
        self.kernel = kernel
        self.gamma = gamma
        self.normal_quantile = float(normal_quantile)

    def fit(self, X_normal: np.ndarray) -> "KernelPCAFaultIsolator":
        try:
            from sklearn.decomposition import KernelPCA
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("install iia-benchmark[ml] for kernel PCA") from exc
        values = np.asarray(X_normal, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("X_normal must contain at least two feature rows")
        components = min(self.n_components, values.shape[0] - 1, values.shape[1])
        self.model_ = KernelPCA(
            n_components=components,
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=True,
            eigen_solver="arpack" if components < values.shape[0] else "auto",
        ).fit(values)
        normal_scores = self.score_samples(values)
        self.threshold_ = float(np.quantile(normal_scores, self.normal_quantile))
        return self

    def residuals(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before scoring")
        values = np.asarray(X, dtype=float)
        reconstructed = self.model_.inverse_transform(self.model_.transform(values))
        return values - reconstructed

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        residuals = self.residuals(X)
        return np.sum(residuals * residuals, axis=1)

    def calibrate_threshold(
        self, X: np.ndarray, abnormal: np.ndarray
    ) -> AlertThresholdResult:
        result = optimize_alert_threshold(self.score_samples(X), abnormal)
        self.threshold_ = result.threshold
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit must be called before prediction")
        return self.score_samples(X) >= self.threshold_

    def influential_features(self, X: np.ndarray, top_k: int = 3) -> np.ndarray:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return np.argsort(np.abs(self.residuals(X)), axis=1)[:, ::-1][:, :top_k]


class TFIDFLSTMAlarmFloodClassifier:
    """Five-stage input/LSTM/dense/softmax/classification early classifier."""

    def __init__(
        self,
        ngram_size: int = 2,
        position_decay: float = 1.0,
        hidden_size: int = 16,
        epochs: int = 100,
        learning_rate: float = 0.02,
        random_state: int = 0,
    ) -> None:
        if hidden_size <= 0 or epochs <= 0 or learning_rate <= 0:
            raise ValueError("hidden_size, epochs and learning_rate must be positive")
        self.ngram_size = int(ngram_size)
        self.position_decay = float(position_decay)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

    def _trajectory(self, sequence: Sequence[Token]) -> np.ndarray:
        tokens = tuple(sequence)
        if not tokens:
            raise ValueError("alarm sequences must be nonempty")
        prefixes = [tokens[: end + 1] for end in range(len(tokens))]
        return self.vectorizer_.transform(prefixes)

    def _tensor(self, sequences: Sequence[Sequence[Token]]):
        torch = self._torch
        trajectories = [self._trajectory(sequence) for sequence in sequences]
        lengths = np.asarray([item.shape[0] for item in trajectories], dtype=int)
        padded = np.zeros(
            (len(trajectories), int(np.max(lengths)), len(self.vectorizer_.vocabulary_)),
            dtype=np.float32,
        )
        for index, trajectory in enumerate(trajectories):
            padded[index, : trajectory.shape[0]] = trajectory
        return torch.tensor(padded), torch.tensor(lengths - 1, dtype=torch.long)

    def fit(
        self, sequences: Sequence[Sequence[Token]], y: np.ndarray
    ) -> "TFIDFLSTMAlarmFloodClassifier":
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("install iia-benchmark[deep] for LSTM classification") from exc
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != len(sequences):
            raise ValueError("y must contain one label per alarm sequence")
        self.classes_ = np.asarray(sorted(set(labels.tolist()), key=repr))
        if self.classes_.size < 2:
            raise ValueError("at least two classes are required")
        self.vectorizer_ = ModifiedTFIDFVectorizer(
            self.ngram_size, self.position_decay
        ).fit(sequences)
        self._torch = torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        class Network(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.output = nn.Linear(hidden_size, output_size)

            def forward(self, data, final_indices):
                states, _ = self.lstm(data)
                rows = torch.arange(states.shape[0], device=states.device)
                return self.output(states[rows, final_indices])

        self.network_ = Network(
            len(self.vectorizer_.vocabulary_), self.hidden_size, len(self.classes_)
        )
        features, final_indices = self._tensor(sequences)
        targets = torch.tensor(
            [int(np.flatnonzero(self.classes_ == label)[0]) for label in labels],
            dtype=torch.long,
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        losses: list[float] = []
        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.network_(features, final_indices), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.training_loss_ = np.asarray(losses)
        return self

    def predict_proba(self, sequences: Sequence[Sequence[Token]]) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("fit must be called before prediction")
        torch = self._torch
        features, final_indices = self._tensor(sequences)
        self.network_.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.network_(features, final_indices), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, sequences: Sequence[Sequence[Token]]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(sequences), axis=1)]

    def predict_evolution(
        self,
        sequences: Sequence[Sequence[Token]],
        prefix_lengths: Sequence[int],
    ) -> dict[int, np.ndarray]:
        lengths = tuple(sorted(set(int(item) for item in prefix_lengths)))
        if not lengths or lengths[0] <= 0:
            raise ValueError("prefix lengths must be positive")
        return {
            length: self.predict([tuple(sequence)[:length] for sequence in sequences])
            for length in lengths
        }
