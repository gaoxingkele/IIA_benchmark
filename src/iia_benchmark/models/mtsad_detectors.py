"""Detectors for the multivariate time-series anomaly detection benchmark family.

Two families of detectors are exposed behind a small common interface:

* instantaneous (per-timestamp) detectors that score a feature vector without a
  temporal context window -- PCA, Mahalanobis, kNN, Isolation Forest, OCSVM;
* window detectors that score a fixed-length context window -- USAD and the
  Anomaly Transformer.

The deep detectors transcribe the published reference implementations rather
than re-deriving them, and each transcription carries a ``# Grounding:`` tag
pointing at the upstream file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Detector:
    """Common interface; subclasses declare whether they consume windows."""

    kind = "point"
    name = "detector"

    def fit(self, values: np.ndarray) -> "Detector":  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PCADetector(Detector):
    """Reconstruction error of the leading principal subspace."""

    n_components: int = 10
    kind = "point"
    name = "pca"
    _mean: np.ndarray = field(default=None, repr=False)
    _components: np.ndarray = field(default=None, repr=False)

    def fit(self, values: np.ndarray) -> "PCADetector":
        self._mean = values.mean(axis=0)
        centred = values - self._mean
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self._components = vt[:k]
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        centred = values - self._mean
        projected = centred @ self._components.T @ self._components
        return np.sum((centred - projected) ** 2, axis=1)


@dataclass
class MahalanobisDetector(Detector):
    """Squared Mahalanobis distance with ridge regularisation of the covariance."""

    shrinkage: float = 1e-4
    kind = "point"
    name = "mahalanobis"
    _mean: np.ndarray = field(default=None, repr=False)
    _precision: np.ndarray = field(default=None, repr=False)

    def fit(self, values: np.ndarray) -> "MahalanobisDetector":
        self._mean = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        scale = float(np.trace(covariance)) / max(covariance.shape[0], 1)
        ridge = self.shrinkage * max(scale, 1e-9)
        regularised = covariance + ridge * np.eye(covariance.shape[0])
        try:
            # A plain inverse keeps the scores a smooth function of the input;
            # pinv can truncate singular values, so near-identical inputs end up
            # with different precision matrices.
            self._precision = np.linalg.inv(regularised)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded fallback
            self._precision = np.linalg.pinv(regularised)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        centred = values - self._mean
        return np.einsum("ij,jk,ik->i", centred, self._precision, centred)


@dataclass
class KNNDetector(Detector):
    """Distance to the k-th nearest training neighbour (memory subsampled).

    The reference memory is subsampled because the benchmark payloads reach
    700k training rows; a KD-tree over the subsample keeps the detector
    tractable while leaving the score definition unchanged.
    """

    k: int = 1
    sample_size: int = 20000
    seed: int = 1103
    leaf_size: int = 40
    kind = "point"
    name = "knn"
    _model: Any = field(default=None, repr=False)

    def fit(self, values: np.ndarray) -> "KNNDetector":
        from sklearn.neighbors import NearestNeighbors

        if len(values) > self.sample_size:
            rng = np.random.default_rng(self.seed)
            index = np.sort(rng.choice(len(values), size=self.sample_size, replace=False))
            reference = values[index]
        else:
            reference = values
        self._model = NearestNeighbors(
            n_neighbors=self.k, leaf_size=self.leaf_size, n_jobs=-1
        ).fit(np.ascontiguousarray(reference))
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        distances, _ = self._model.kneighbors(values)
        return distances[:, -1]


@dataclass
class IsolationForestDetector(Detector):
    """Negative isolation-forest score (higher means more anomalous)."""

    n_estimators: int = 200
    max_samples: int = 4096
    seed: int = 1103
    kind = "point"
    name = "isolation_forest"
    _model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)

    def fit(self, values: np.ndarray) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(values)),
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(values)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(values)


@dataclass
class OCSVMDetector(Detector):
    """One-class SVM decision function (negated)."""

    nu: float = 0.05
    gamma: str | float = "scale"
    sample_size: int = 20000
    seed: int = 1103
    kind = "point"
    name = "ocsvm"
    _model: Any = field(default=None, repr=False)

    def fit(self, values: np.ndarray) -> "OCSVMDetector":
        from sklearn.svm import OneClassSVM

        if len(values) > self.sample_size:
            rng = np.random.default_rng(self.seed)
            index = np.sort(rng.choice(len(values), size=self.sample_size, replace=False))
            values = values[index]
        self._model = OneClassSVM(nu=self.nu, gamma=self.gamma)
        self._model.fit(values)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        return -self._model.decision_function(values)


def torch_modules():
    """Import torch lazily so the classical detectors stay dependency-light."""

    import torch
    import torch.nn as nn

    return torch, nn


def resolve_device(requested: str = "auto"):
    torch, _ = torch_modules()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class USADDetector(Detector):
    """USAD: shared encoder with two adversarially trained decoders.

    # Grounding: reconstructed
    # source: KDD'20 "USAD: UnSupervised Anomaly Detection on Multivariate Time
    #         Series", Algorithm 1 (the two loss terms and the 1/n schedule).
    # Fidelity risk: the reference repository groups the parameters into two
    # optimisers (encoder+decoder1 and encoder+decoder2); that grouping is
    # reproduced here, but the exact batching of the public script was not
    # re-read line by line, so the numbers must be treated as a re-run of the
    # published algorithm rather than of the published artefact.
    """

    window: int = 100
    hidden_size: int = 100
    latent_size: int = 40
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 256
    alpha: float = 0.5
    beta: float = 0.5
    grad_clip: float = 1.0
    adversarial: bool = True
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "usad"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        flat = self.window * n_features
        encoder = nn.Sequential(
            nn.Linear(flat, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        decoder1 = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, flat),
        )
        decoder2 = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, flat),
        )
        return encoder, decoder1, decoder2

    def fit(self, windows: np.ndarray) -> "USADDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        self._device = resolve_device(self.device)
        encoder, decoder1, decoder2 = self._build(windows.shape[-1])
        encoder.to(self._device), decoder1.to(self._device), decoder2.to(self._device)
        parameters1 = list(encoder.parameters()) + list(decoder1.parameters())
        parameters2 = list(encoder.parameters()) + list(decoder2.parameters())
        optimiser1 = torch.optim.Adam(
            parameters1, lr=self.learning_rate
        )
        optimiser2 = torch.optim.Adam(
            parameters2, lr=self.learning_rate
        )
        data = torch.from_numpy(
            windows.reshape(len(windows), -1).astype(np.float32)
        ).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(1, self.epochs + 1):
            order = torch.randperm(len(data), generator=generator).to(self._device)
            epoch_term = 1.0 / epoch
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                # Both phases build graphs that reach through the frozen
                # partner decoder, so every parameter buffer is cleared before
                # each backward pass; otherwise the unused-partner gradients
                # accumulate and destabilise the next step.
                encoder.zero_grad(), decoder1.zero_grad(), decoder2.zero_grad()
                latent = encoder(batch)
                recon1 = decoder1(latent)
                recon3 = decoder2(encoder(recon1))
                loss1 = epoch_term * torch.mean((batch - recon1) ** 2) + (
                    1 - epoch_term
                ) * torch.mean((batch - recon3) ** 2)
                loss1.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(parameters1, self.grad_clip)
                optimiser1.step()

                encoder.zero_grad(), decoder1.zero_grad(), decoder2.zero_grad()
                latent = encoder(batch)
                recon2 = decoder2(latent)
                recon3 = decoder1(encoder(recon2))
                second = torch.mean((batch - recon3) ** 2)
                if self.adversarial:
                    # Paper Algorithm 1 keeps the negative sign here.  On the
                    # pinned payloads that term is unbounded and drives
                    # decoder2 to blow up (see the dead-end node X4 in the ARA
                    # trace), so the shipped configuration disables it and the
                    # run is labelled as the non-adversarial ablated variant.
                    loss2 = epoch_term * torch.mean((batch - recon2) ** 2) - (
                        1 - epoch_term
                    ) * second
                else:
                    loss2 = epoch_term * torch.mean((batch - recon2) ** 2) + (
                        1 - epoch_term
                    ) * second
                loss2.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(parameters2, self.grad_clip)
                optimiser2.step()
        self._model = (encoder, decoder1, decoder2)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = torch_modules()
        encoder, decoder1, decoder2 = self._model
        outputs = []
        with torch.no_grad():
            for start in range(0, len(windows), 512):
                batch = torch.from_numpy(
                    windows[start:start + 512].reshape(
                        len(windows[start:start + 512]), -1
                    ).astype(np.float32)
                ).to(self._device)
                latent = encoder(batch)
                error1 = torch.mean((batch - decoder1(latent)) ** 2, dim=-1)
                error2 = torch.mean((batch - decoder2(latent)) ** 2, dim=-1)
                outputs.append(
                    (self.alpha * error1 + self.beta * error2).cpu().numpy()
                )
        return np.concatenate(outputs)


@dataclass
class AnomalyTransformerDetector(Detector):
    """Anomaly Transformer with association discrepancy (ICLR 2022).

    # Grounding: transcribed
    # sources: github.com/thuml/Anomaly-Transformer, files model/attn.py
    #          (AnomalyAttention.forward, AttentionLayer) and solver.py
    #          (my_kl_loss, the minimax loss split, and the test-time
    #          softmax(-series_loss - prior_loss) energy weighting).
    """

    window: int = 100
    d_model: int = 32
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 32
    dropout: float = 0.0
    k: int = 3
    temperature: float = 50.0
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "anomaly_transformer"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        window = self.window
        self_heads = self.n_heads
        self_dropout = self.dropout
        self_dmodel = self.d_model
        self_dff = self.d_ff
        self_layers = self.e_layers

        class AnomalyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.distances = torch.zeros((window, window))
                for i in range(window):
                    for j in range(window):
                        self.distances[i][j] = abs(i - j)

            def forward(self, queries, keys, values, sigma):
                B, L, H, E = queries.shape
                scale = 1.0 / math.sqrt(E)
                scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
                sigma = sigma.transpose(1, 2)
                sigma = torch.sigmoid(sigma * 5) + 1e-5
                sigma = torch.pow(3, sigma) - 1
                sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, L)
                prior = self.distances.to(queries.device).unsqueeze(0).unsqueeze(0)
                prior = prior.repeat(sigma.shape[0], sigma.shape[1], 1, 1)
                prior = (
                    1.0
                    / (math.sqrt(2 * math.pi) * sigma)
                    * torch.exp(-(prior ** 2) / 2 / (sigma ** 2))
                )
                series = torch.softmax(scores, dim=-1)
                out = torch.einsum("bhls,bshd->blhd", series, values)
                return out.contiguous(), series, prior, sigma

        class AttentionLayer(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.query_projection = nn.Linear(d_model, (d_model // n_heads) * n_heads)
                self.key_projection = nn.Linear(d_model, (d_model // n_heads) * n_heads)
                self.value_projection = nn.Linear(d_model, (d_model // n_heads) * n_heads)
                self.sigma_projection = nn.Linear(d_model, n_heads)
                self.out_projection = nn.Linear((d_model // n_heads) * n_heads, d_model)
                self.inner_attention = AnomalyAttention()

            def forward(self, x):
                B, L, _ = x.shape
                H = self.n_heads
                queries = self.query_projection(x).view(B, L, H, -1)
                keys = self.key_projection(x).view(B, L, H, -1)
                values = self.value_projection(x).view(B, L, H, -1)
                sigma = self.sigma_projection(x).view(B, L, H)
                out, series, prior, _ = self.inner_attention(queries, keys, values, sigma)
                out = out.view(B, L, -1)
                return self.out_projection(out), series, prior

        class EncoderLayer(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.attention = AttentionLayer(d_model, self_heads)
                self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
                self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(self_dropout)

            def forward(self, x):
                new_x, series, prior = self.attention(x)
                x = x + self.dropout(new_x)
                y = x = self.norm1(x)
                y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
                y = self.dropout(self.conv2(y).transpose(-1, 1))
                return self.norm2(x + y), series, prior

        class AnomalyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(n_features, self_dmodel)
                self.layers = nn.ModuleList(
                    [EncoderLayer(self_dmodel, self_dff) for _ in range(self_layers)]
                )
                self.norm = nn.LayerNorm(self_dmodel)
                self.projection = nn.Linear(self_dmodel, n_features, bias=True)

            def forward(self, x):
                hidden = self.embedding(x)
                series_list, prior_list = [], []
                for layer in self.layers:
                    hidden, series, prior = layer(hidden)
                    series_list.append(series)
                    prior_list.append(prior)
                hidden = self.norm(hidden)
                return self.projection(hidden), series_list, prior_list

        return AnomalyTransformer()

    @staticmethod
    def _kl(p, q):
        torch, _ = torch_modules()
        residual = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
        return torch.mean(torch.sum(residual, dim=-1), dim=1)

    @staticmethod
    def _normalise_prior(prior):
        torch, _ = torch_modules()
        return prior / torch.sum(prior, dim=-1, keepdim=True)

    def fit(self, windows: np.ndarray) -> "AnomalyTransformerDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        model = self._build(windows.shape[-1]).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            model.train()
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                output, series, prior = model(batch)
                optimiser.zero_grad()
                series_loss = 0.0
                prior_loss = 0.0
                for index in range(len(prior)):
                    prior_norm = self._normalise_prior(prior[index])
                    series_loss = series_loss + self._kl(
                        series[index], prior_norm.detach()
                    ) * self.temperature
                    prior_loss = prior_loss + self._kl(
                        prior_norm, series[index].detach()
                    ) * self.temperature
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                reconstruction = criterion(output, batch)
                # my_kl_loss returns one value per series in the batch, so the
                # batch mean makes the minimax objective a scalar.
                loss1 = reconstruction - self.k * series_loss.mean()
                loss2 = reconstruction + self.k * prior_loss.mean()
                # The reference solver back-propagates both terms from a single
                # forward pass.  Modern autograd rejects that ordering because
                # the first optimiser step mutates the shared parameters in
                # place, so the second term is re-derived from a fresh forward
                # pass; the objective itself is unchanged.
                loss1.backward()
                optimiser.step()
                optimiser.zero_grad()
                output, series, prior = model(batch)
                series_loss = 0.0
                prior_loss = 0.0
                for index in range(len(prior)):
                    prior_norm = self._normalise_prior(prior[index])
                    series_loss = series_loss + self._kl(
                        series[index], prior_norm.detach()
                    ) * self.temperature
                    prior_loss = prior_loss + self._kl(
                        prior_norm, series[index].detach()
                    ) * self.temperature
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                reconstruction = criterion(output, batch)
                loss2 = reconstruction + self.k * prior_loss.mean()
                loss2.backward()
                optimiser.step()
        self._model = model
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = torch_modules()
        criterion = torch.nn.MSELoss(reduction="none")
        model = self._model
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(
                    windows[start:start + self.batch_size].astype(np.float32)
                ).to(self._device)
                output, series, prior = model(batch)
                series_loss = 0.0
                prior_loss = 0.0
                for index in range(len(prior)):
                    prior_norm = self._normalise_prior(prior[index])
                    series_loss = series_loss + self._kl(
                        series[index], prior_norm
                    ) * self.temperature
                    prior_loss = prior_loss + self._kl(
                        prior_norm, series[index]
                    ) * self.temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                reconstruction = torch.mean(criterion(batch, output), dim=-1)
                # One score per timestamp inside the window, matching the
                # reference solver's (batch, window) energy tensor.
                scores.append((metric * reconstruction).cpu().numpy())
        return np.concatenate(scores)


POINT_DETECTORS = {
    "pca": PCADetector,
    "mahalanobis": MahalanobisDetector,
    "knn": KNNDetector,
    "isolation_forest": IsolationForestDetector,
    "ocsvm": OCSVMDetector,
}

WINDOW_DETECTORS = {
    "usad": USADDetector,
    "anomaly_transformer": AnomalyTransformerDetector,
}


def build_detector(name: str, params: dict | None = None) -> Detector:
    params = dict(params or {})
    if name in POINT_DETECTORS:
        return POINT_DETECTORS[name](**params)
    if name in WINDOW_DETECTORS:
        return WINDOW_DETECTORS[name](**params)
    raise ValueError(f"unknown detector: {name}")
