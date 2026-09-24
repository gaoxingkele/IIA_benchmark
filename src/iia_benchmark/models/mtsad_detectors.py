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

@dataclass
class DCdetectorDetector(Detector):
    """DCdetector: dual attention contrastive representation learning.

    # Grounding: transcribed
    # sources: github.com/DAMO-DI-ML/KDD2023-DCdetector
    #   - model/DCdetector.py (multi-scale patching, RevIN, encoder wiring)
    #   - model/attn.py (DAC_structure dual attention + the upsampling/reduce)
    #   - model/embed.py (TokenEmbedding conv + sinusoidal PositionalEmbedding)
    #   - solver.py (my_kl_loss, the ``prior_loss - series_loss`` objective and
    #     the test-time ``softmax(-series_loss - prior_loss)`` energy)
    # Deviations: einops ``repeat``/``reduce`` are expressed with
    # repeat_interleave and view/mean, which is shape-identical; training runs
    # one objective per batch because the reference solver also has no
    # reconstruction term.
    """

    window: int = 100
    patch_sizes: tuple[int, ...] = (3, 5, 7)
    d_model: int = 256
    n_heads: int = 1
    e_layers: int = 3
    dropout: float = 0.0
    temperature: float = 50.0
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "dcdetector"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        window = self.window
        patch_sizes = tuple(int(size) for size in self.patch_sizes)
        for size in patch_sizes:
            if window % size:
                raise ValueError(
                    f"window {window} must be divisible by patch size {size}"
                )
        heads = self.n_heads
        d_model = self.d_model
        dropout = self.dropout

        class PositionalEmbedding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model).float()
                position = torch.arange(0, max_len).float().unsqueeze(1)
                div_term = (
                    torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
                ).exp()
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return self.pe[:, : x.size(1)]

        class TokenEmbedding(nn.Module):
            def __init__(self, c_in, d_model):
                super().__init__()
                self.tokenConv = nn.Conv1d(
                    in_channels=c_in,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                    bias=False,
                )
                nn.init.kaiming_normal_(
                    self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

            def forward(self, x):
                return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        class DataEmbedding(nn.Module):
            def __init__(self, c_in, d_model, dropout):
                super().__init__()
                self.value_embedding = TokenEmbedding(c_in, d_model)
                self.position_embedding = PositionalEmbedding(d_model)
                self.dropout = nn.Dropout(p=dropout)

            def forward(self, x):
                return self.dropout(
                    self.value_embedding(x) + self.position_embedding(x)
                )

        class DACStructure(nn.Module):
            def forward(self, q_patch, q_num, k_patch, k_num, patch_index, channel):
                patch = patch_sizes[patch_index]
                batch_channel, _, _, _ = q_patch.shape
                scale = 1.0 / math.sqrt(q_patch.shape[-1])
                scores_patch = (
                    torch.einsum("blhe,bshe->bhls", q_patch, k_patch) * scale
                )
                series_patch = torch.softmax(scores_patch, dim=-1)
                scores_num = torch.einsum("blhe,bshe->bhls", q_num, k_num) * scale
                series_num = torch.softmax(scores_num, dim=-1)

                # Upsampling: a patch-level score covers patch x patch entries.
                series_patch = series_patch.repeat_interleave(patch, dim=2)
                series_patch = series_patch.repeat_interleave(patch, dim=3)
                repeat_num = window // patch
                series_num = series_num.repeat(1, 1, repeat_num, repeat_num)

                batch = batch_channel // channel
                series_patch = series_patch.view(batch, channel, *series_patch.shape[1:])
                series_num = series_num.view(batch, channel, *series_num.shape[1:])
                return series_patch.mean(dim=1), series_num.mean(dim=1)

        class AttentionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                d_keys = d_model // heads
                self.patch_query_projection = nn.Linear(d_model, d_keys * heads)
                self.patch_key_projection = nn.Linear(d_model, d_keys * heads)
                self.value_projection = nn.Linear(d_model, d_keys * heads)
                self.out_projection = nn.Linear(d_keys * heads, d_model)
                self.norm = nn.LayerNorm(d_model)
                self.inner_attention = DACStructure()

            def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, channel):
                batch, length, _ = x_patch_size.shape
                q_patch = self.patch_query_projection(x_patch_size).view(
                    batch, length, heads, -1
                )
                k_patch = self.patch_key_projection(x_patch_size).view(
                    batch, length, heads, -1
                )
                batch, length, _ = x_patch_num.shape
                q_num = self.patch_query_projection(x_patch_num).view(
                    batch, length, heads, -1
                )
                k_num = self.patch_key_projection(x_patch_num).view(
                    batch, length, heads, -1
                )
                return self.inner_attention(
                    q_patch, q_num, k_patch, k_num, patch_index, channel
                )

        class DCdetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_patch_size = nn.ModuleList(
                    [DataEmbedding(size, d_model, dropout) for size in patch_sizes]
                )
                self.embedding_patch_num = nn.ModuleList(
                    [
                        DataEmbedding(window // size, d_model, dropout)
                        for size in patch_sizes
                    ]
                )
                self.embedding_window_size = DataEmbedding(n_features, d_model, dropout)
                self.layers = nn.ModuleList([AttentionLayer() for _ in range(self_layers)])
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                batch, _, channels = x.shape
                mean = x.mean(dim=1, keepdim=True)
                std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
                x = (x - mean) / std
                x_ori = self.embedding_window_size(x)
                series_all: list = []
                prior_all: list = []
                for patch_index, patch in enumerate(patch_sizes):
                    patch_size_input = x.permute(0, 2, 1)
                    patch_num_input = x.permute(0, 2, 1)
                    patch_size_input = patch_size_input.reshape(
                        batch * channels, window // patch, patch
                    )
                    patch_size_input = self.embedding_patch_size[patch_index](
                        patch_size_input
                    )
                    patch_num_input = (
                        patch_num_input.reshape(batch * channels, patch, window // patch)
                    )
                    patch_num_input = self.embedding_patch_num[patch_index](
                        patch_num_input
                    )
                    for layer in self.layers:
                        series, prior = layer(
                            patch_size_input, patch_num_input, x_ori, patch_index, channels
                        )
                        series_all.append(series)
                        prior_all.append(prior)
                return series_all, prior_all

        self_layers = self.e_layers
        return DCdetector()

    @staticmethod
    def _kl(p, q):
        torch, _ = torch_modules()
        residual = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
        return torch.mean(torch.sum(residual, dim=-1), dim=1)

    def _losses(self, series, prior, *, detach_partner: bool):
        torch, _ = torch_modules()
        series_loss = 0.0
        prior_loss = 0.0
        for index in range(len(prior)):
            normalised = prior[index] / torch.sum(
                prior[index], dim=-1, keepdim=True
            )
            if detach_partner:
                series_loss = series_loss + self._kl(
                    series[index], normalised.detach()
                ) * self.temperature
                prior_loss = prior_loss + self._kl(
                    normalised, series[index].detach()
                ) * self.temperature
            else:
                series_loss = series_loss + self._kl(series[index], normalised.detach())
                prior_loss = prior_loss + self._kl(normalised, series[index].detach())
        return series_loss / len(prior), prior_loss / len(prior)

    def fit(self, windows: np.ndarray) -> "DCdetectorDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        model = self._build(windows.shape[-1]).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                optimiser.zero_grad()
                series, prior = model(batch)
                series_loss, prior_loss = self._losses(
                    series, prior, detach_partner=False
                )
                loss = (prior_loss - series_loss).mean()
                loss.backward()
                optimiser.step()
        self._model = model
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = torch_modules()
        model = self._model
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(
                    windows[start:start + self.batch_size].astype(np.float32)
                ).to(self._device)
                series, prior = model(batch)
                series_loss, prior_loss = self._losses(
                    series, prior, detach_partner=True
                )
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                scores.append(metric.cpu().numpy())
        return np.concatenate(scores)


@dataclass
class TimesNetDetector(Detector):
    """TimesNet reconstruction detector (ICLR 2023).

    # Grounding: transcribed
    # sources: github.com/thuml/Time-Series-Library
    #   - models/TimesNet.py (FFT_for_Period, TimesBlock, anomaly_detection)
    #   - layers/Conv_Blocks.py (Inception_Block_V1)
    #   - layers/Embed.py (DataEmbedding: TokenEmbedding + PositionalEmbedding,
    #     with the x_mark=None branch used by the anomaly-detection path)
    #   - exp/exp_anomaly_detection.py (MSE objective, train+test percentile
    #     threshold, mean-over-channel energy)
    # Deviation: the reference forecast path also predicts future steps; the
    # anomaly-detection path uses pred_len = 0, which is what is transcribed.
    """

    window: int = 100
    d_model: int = 64
    d_ff: int = 64
    e_layers: int = 2
    top_k: int = 5
    num_kernels: int = 6
    dropout: float = 0.1
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "timesnet"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        window = self.window
        self_top_k = self.top_k
        self_kernels = self.num_kernels
        self_d_model = self.d_model
        self_d_ff = self.d_ff
        self_dropout = self.dropout
        self_layers = self.e_layers

        def fft_for_period(x, k):
            xf = torch.fft.rfft(x, dim=1)
            frequency_list = abs(xf).mean(0).mean(-1)
            frequency_list[0] = 0
            _, top_list = torch.topk(frequency_list, k)
            top_list = top_list.detach().cpu().numpy()
            period = x.shape[1] // top_list
            return period, abs(xf).mean(-1)[:, top_list]

        class InceptionBlockV1(nn.Module):
            def __init__(self, in_channels, out_channels, num_kernels=6):
                super().__init__()
                self.num_kernels = num_kernels
                self.kernels = nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels, out_channels, kernel_size=2 * i + 1, padding=i
                        )
                        for i in range(num_kernels)
                    ]
                )
                for module in self.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_out", nonlinearity="relu"
                        )
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)

            def forward(self, x):
                res = torch.stack(
                    [kernel(x) for kernel in self.kernels], dim=-1
                ).mean(-1)
                return res

        class TimesBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    InceptionBlockV1(self_d_model, self_d_ff, self_kernels),
                    nn.GELU(),
                    InceptionBlockV1(self_d_ff, self_d_model, self_kernels),
                )

            def forward(self, x):
                batch, length_in, channels = x.size()
                period_list, period_weight = fft_for_period(x, self_top_k)
                results = []
                for index in range(self_top_k):
                    period = period_list[index]
                    if length_in % period != 0:
                        padded_length = (length_in // period + 1) * period
                        padding = torch.zeros(
                            [x.shape[0], padded_length - length_in, x.shape[2]]
                        ).to(x.device)
                        out = torch.cat([x, padding], dim=1)
                    else:
                        padded_length = length_in
                        out = x
                    out = out.reshape(
                        batch, padded_length // period, period, channels
                    ).permute(0, 3, 1, 2).contiguous()
                    out = self.conv(out)
                    out = out.permute(0, 2, 3, 1).reshape(batch, -1, channels)
                    results.append(out[:, :length_in, :])
                stacked = torch.stack(results, dim=-1)
                weights = torch.softmax(period_weight, dim=1)
                weights = weights.unsqueeze(1).unsqueeze(1).repeat(
                    1, length_in, channels, 1
                )
                return torch.sum(stacked * weights, -1) + x

        class PositionalEmbedding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model).float()
                position = torch.arange(0, max_len).float().unsqueeze(1)
                div_term = (
                    torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
                ).exp()
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return self.pe[:, : x.size(1)]

        class TokenEmbedding(nn.Module):
            def __init__(self, c_in, d_model):
                super().__init__()
                self.tokenConv = nn.Conv1d(
                    in_channels=c_in,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                    bias=False,
                )
                nn.init.kaiming_normal_(
                    self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

            def forward(self, x):
                return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.value_embedding = TokenEmbedding(n_features, self_d_model)
                self.position_embedding = PositionalEmbedding(self_d_model)
                self.embedding_dropout = nn.Dropout(p=self_dropout)
                self.blocks = nn.ModuleList(
                    [TimesBlock() for _ in range(self_layers)]
                )
                self.layer_norm = nn.LayerNorm(self_d_model)
                self.projection = nn.Linear(self_d_model, n_features, bias=True)

            def forward(self, x):
                means = x.mean(1, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
                x = x / stdev
                enc_out = self.embedding_dropout(
                    self.value_embedding(x) + self.position_embedding(x)
                )
                for block in self.blocks:
                    enc_out = self.layer_norm(block(enc_out))
                dec_out = self.projection(enc_out)
                dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
                    1, window, 1
                )
                return dec_out + means[:, 0, :].unsqueeze(1).repeat(1, window, 1)

        return Model()

    def fit(self, windows: np.ndarray) -> "TimesNetDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        model = self._build(windows.shape[-1]).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                optimiser.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
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
                output = model(batch)
                # Reference: mean squared error over channels, per timestamp.
                scores.append(torch.mean(criterion(batch, output), dim=-1).cpu().numpy())
        return np.concatenate(scores)


@dataclass
class ITransformerDetector(Detector):
    """iTransformer reconstruction detector (ICLR 2024).

    # Grounding: transcribed
    # sources: github.com/thuml/Time-Series-Library
    #   - models/iTransformer.py (inverted embedding, encoder, projection back to
    #     the time axis, anomaly-detection normalisation)
    #   - layers/Embed.py (DataEmbedding_inverted)
    #   - layers/Transformer_EncDec.py (EncoderLayer, Encoder)
    #   - layers/SelfAttention_Family.py (FullAttention, AttentionLayer)
    # Status: diagnostic model. The release publishes an anomaly-detection script
    # only for MSL, and no point-adjusted number for this model exists in this
    # artifact's corpus, so it is used to test whether the residual MSL/SWaT gaps
    # follow the attention family rather than to produce a verdict row.
    """

    window: int = 100
    d_model: int = 128
    d_ff: int = 128
    e_layers: int = 3
    n_heads: int = 8
    factor: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "itransformer"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        window = self.window
        heads = self.n_heads
        d_model = self.d_model
        d_ff = self.d_ff
        dropout = self.dropout
        activation = nn.GELU if self.activation == "gelu" else nn.ReLU
        layers = self.e_layers

        class FullAttention(nn.Module):
            def forward(self, queries, keys, values, attn_mask):
                batch, length, head, embed = queries.shape
                scale = 1.0 / math.sqrt(embed)
                scores = torch.einsum("blhe,bshe->bhls", queries, keys)
                attn = torch.softmax(scale * scores, dim=-1)
                out = torch.einsum("bhls,bshd->blhd", attn, values)
                return out.contiguous(), attn

        class AttentionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                d_keys = d_model // heads
                self.query_projection = nn.Linear(d_model, d_keys * heads)
                self.key_projection = nn.Linear(d_model, d_keys * heads)
                self.value_projection = nn.Linear(d_model, d_keys * heads)
                self.out_projection = nn.Linear(d_keys * heads, d_model)
                self.inner_attention = FullAttention()

            def forward(self, x):
                batch, length, _ = x.shape
                queries = self.query_projection(x).view(batch, length, heads, -1)
                keys = self.key_projection(x).view(batch, length, heads, -1)
                values = self.value_projection(x).view(batch, length, heads, -1)
                out, attn = self.inner_attention(queries, keys, values, None)
                out = out.view(batch, length, -1)
                return self.out_projection(out), attn

        class EncoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = AttentionLayer()
                self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
                self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                self.activation = activation()

            def forward(self, x):
                new_x, attn = self.attention(x)
                x = x + self.dropout(new_x)
                y = x = self.norm1(x)
                y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
                y = self.dropout(self.conv2(y).transpose(-1, 1))
                return self.norm2(x + y), attn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # DataEmbedding_inverted: one linear map per variate over time.
                self.value_embedding = nn.Linear(window, d_model)
                self.embedding_dropout = nn.Dropout(p=dropout)
                self.layers = nn.ModuleList([EncoderLayer() for _ in range(layers)])
                self.norm = nn.LayerNorm(d_model)
                self.projection = nn.Linear(d_model, window, bias=True)

            def forward(self, x):
                means = x.mean(1, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
                x = x / stdev
                _, length, channels = x.shape
                enc_out = self.embedding_dropout(
                    self.value_embedding(x.permute(0, 2, 1))
                )
                for layer in self.layers:
                    enc_out, _ = layer(enc_out)
                enc_out = self.norm(enc_out)
                dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :channels]
                dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, length, 1)
                return dec_out + means[:, 0, :].unsqueeze(1).repeat(1, length, 1)

        return Model()

    def fit(self, windows: np.ndarray) -> "ITransformerDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        model = self._build(windows.shape[-1]).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                optimiser.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
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
                output = model(batch)
                scores.append(torch.mean(criterion(batch, output), dim=-1).cpu().numpy())
        return np.concatenate(scores)


@dataclass
class TranADDetector(Detector):
    """TranAD: two-phase transformer with focus-score self-conditioning (VLDB 2022).

    # Grounding: transcribed
    # sources: github.com/imperial-qore/TranAD
    #   - src/models.py (PositionalEncoding, the TranAD module: one encoder, two
    #     decoders, the fcn head, and the phase-2 conditioning on squared phase-1
    #     error)
    #   - main.py:backprop (window construction, the (1/n) / (1 - 1/n) weighting
    #     between the two phases, AdamW with weight decay 1e-5 and a StepLR(5, 0.9)
    #     schedule, and the test-time score l(z1, elem) at the window's last step)
    # Deviations recorded here: inputs are float32 rather than the release's
    # float64, and the harness's own window grid is used, so the first window is
    # not the release's repeated-first-row padding; both are monotone-equivalent
    # for the ranking the metrics consume.
    # The release thresholds these scores with POT (src/pot.py, SPOT with per
    # dataset lm_d/lr_d); this harness reports the percentile reference protocol
    # and a separate POT script, because mixing the two would confound model and
    # threshold.
    """

    window: int = 10
    d_ff: int = 16
    dropout: float = 0.1
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    scheduler_step: int = 5
    scheduler_gamma: float = 0.9
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "tranad"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        window = self.window
        d_ff = self.d_ff
        dropout = self.dropout

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout, max_len=5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

            def forward(self, x):
                return self.dropout(x + self.pe[: x.size(0), :])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_feats = n_features
                self.n_window = window
                self.n = n_features * window
                self.pos_encoder = PositionalEncoding(2 * n_features, dropout)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=2 * n_features,
                    nhead=n_features,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 1)
                decoder_layer1 = nn.TransformerDecoderLayer(
                    d_model=2 * n_features,
                    nhead=n_features,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, 1)
                decoder_layer2 = nn.TransformerDecoderLayer(
                    d_model=2 * n_features,
                    nhead=n_features,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, 1)
                self.fcn = nn.Sequential(
                    nn.Linear(2 * n_features, n_features), nn.Sigmoid()
                )

            def encode(self, src, c, tgt):
                src = torch.cat((src, c), dim=2)
                src = src * math.sqrt(self.n_feats)
                src = self.pos_encoder(src)
                memory = self.transformer_encoder(src)
                return tgt.repeat(1, 1, 2), memory

            def forward(self, src, tgt):
                # Phase 1 - without anomaly scores.
                c = torch.zeros_like(src)
                x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
                # Phase 2 - with the phase-1 squared error as the focus score.
                c = (x1 - src) ** 2
                x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
                return x1, x2

        return Model()

    def fit(self, windows: np.ndarray) -> "TranADDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        n_features = windows.shape[-1]
        model = self._build(n_features).to(self._device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, self.scheduler_step, self.scheduler_gamma
        )
        criterion = torch.nn.MSELoss(reduction="none")
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for epoch in range(self.epochs):
            term = 1.0 / (epoch + 1)
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                window = batch.permute(1, 0, 2)
                element = window[-1, :, :].view(1, batch.shape[0], n_features)
                x1, x2 = model(window, element)
                loss = term * criterion(x1, element) + (1 - term) * criterion(
                    x2, element
                )
                optimiser.zero_grad()
                torch.mean(loss).backward(retain_graph=True)
                optimiser.step()
            scheduler.step()
        self._model = model
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = torch_modules()
        criterion = torch.nn.MSELoss(reduction="none")
        model = self._model
        model.eval()
        n_features = windows.shape[-1]
        scores = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(
                    windows[start:start + self.batch_size].astype(np.float32)
                ).to(self._device)
                window = batch.permute(1, 0, 2)
                element = window[-1, :, :].view(1, batch.shape[0], n_features)
                _, x2 = model(window, element)
                loss = criterion(x2, element)[0]
                # One score per window, from the window's final timestamp.  The
                # release sums or means over features depending on the branch;
                # either is a monotone rescaling and leaves the ranking intact.
                scores.append(torch.mean(loss, dim=-1).cpu().numpy())
        return np.concatenate(scores)


@dataclass
class CATCHDetector(Detector):
    """CATCH: channel-aware frequency-patching detector (ICLR 2025).

    # Grounding: transcribed
    # sources: github.com/decisionintelligence/CATCH, ts_benchmark/baselines/catch
    #   - models/CATCH_model.py (RevIN, FFT, real/imag patching, the channel mask
    #     generator, the two flatten heads, the complex reassembly and ircom)
    #   - layers/cross_channel_Transformer.py (Trans_C, c_Transformer, c_Attention)
    #   - layers/channel_mask.py (gumbel-softmax Bernoulli mask, diagonal forced)
    #   - utils/ch_discover_loss.py (DynamicalContrastiveLoss)
    #   - utils/fre_rec_loss.py (frequency_loss and frequency_criterion)
    #   - CATCH.py (the two optimisers, the rec + dc_lambda*dc + auxi_lambda*auxi
    #     objective, the periodic mask-optimiser step, and the test score
    #     temp_score + score_lambda * freq_score)
    # Deviations: einops rearrange calls are expressed with reshape/permute; the
    # release's early stopping on its own validation split is not reproduced, the
    # harness trains the fixed epoch budget and evaluates on the pinned splits.
    """

    window: int = 192
    patch_size: int = 16
    patch_stride: int = 8
    inference_patch_size: int = 32
    inference_patch_stride: int = 1
    cf_dim: int = 64
    d_model: int = 128
    d_ff: int = 256
    head_dim: int = 64
    n_heads: int = 8
    e_layers: int = 3
    dropout: float = 0.2
    head_dropout: float = 0.1
    regular_lambda: float = 0.5
    temperature: float = 0.07
    auxi_lambda: float = 0.005
    dc_lambda: float = 0.005
    score_lambda: float = 0.05
    mask_update_every: int = 100
    learning_rate: float = 1e-4
    mask_learning_rate: float = 1e-5
    epochs: int = 3
    batch_size: int = 128
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "catch"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def _build(self, n_features: int):
        torch, nn = torch_modules()
        seq_len = self.window
        patch_size = self.patch_size
        patch_stride = self.patch_stride
        cf_dim = self.cf_dim
        d_model = self.d_model
        d_ff = self.d_ff
        head_dim = self.head_dim
        heads = self.n_heads
        depth = self.e_layers
        dropout = self.dropout
        head_dropout = self.head_dropout
        regular_lambda = self.regular_lambda
        temperature = self.temperature
        patch_num = int((seq_len - patch_size) / patch_stride + 1)

        class RevIN(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                self.eps = 1e-5
                self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

            def forward(self, x, mode):
                if mode == "norm":
                    self._mean = x.mean(dim=1, keepdim=True).detach()
                    self._stdev = torch.sqrt(
                        x.var(dim=1, keepdim=True, unbiased=False) + self.eps
                    ).detach()
                    x = (x - self._mean) / self._stdev
                    return x * self.affine_weight + self.affine_bias
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
                return x * self._stdev + self._mean

        class ChannelMaskGenerator(nn.Module):
            def __init__(self, input_size, n_vars):
                super().__init__()
                self.generator = nn.Sequential(
                    nn.Linear(input_size * 2, n_vars, bias=False), nn.Sigmoid()
                )
                with torch.no_grad():
                    self.generator[0].weight.zero_()
                self.n_vars = n_vars

            def forward(self, x):
                distribution = self.generator(x)
                sampled = self._bernoulli_gumbel_rsample(distribution)
                eye = torch.eye(self.n_vars, device=x.device)
                inverse_eye = 1 - eye
                return torch.einsum("bcd,cd->bcd", sampled, inverse_eye) + eye

            def _bernoulli_gumbel_rsample(self, distribution_matrix):
                batch, channels, dim = distribution_matrix.shape
                flat = distribution_matrix.reshape(batch * channels * dim, 1)
                r_flat = 1 - flat
                log_flat = torch.log(flat / r_flat)
                log_r_flat = torch.log(r_flat / flat)
                both = torch.concat([log_flat, log_r_flat], dim=-1)
                resampled = torch.nn.functional.gumbel_softmax(both, hard=True)
                return resampled[..., 0].reshape(batch, channels, dim)

        class DynamicalContrastiveLoss(nn.Module):
            def __init__(self, k, temperature):
                super().__init__()
                self.temperature = temperature
                self.k = k

            def forward(self, scores, attn_mask, norm_matrix):
                batch = scores.shape[0]
                n_vars = scores.shape[-1]
                cosine = (scores / norm_matrix).mean(1)
                pos_scores = torch.exp(cosine / self.temperature) * attn_mask
                all_scores = torch.exp(cosine / self.temperature)
                clustering = -torch.log(
                    pos_scores.sum(dim=-1) / all_scores.sum(dim=-1)
                )
                eye = (
                    torch.eye(attn_mask.shape[-1])
                    .unsqueeze(0)
                    .repeat(batch, 1, 1)
                    .to(attn_mask.device)
                )
                regular = 1.0 / (n_vars * (n_vars - 1)) * torch.norm(
                    eye.reshape(batch, -1) - attn_mask.reshape(batch, -1), p=1, dim=-1
                )
                return (clustering.mean(1) + self.k * regular).mean()

        class CAttention(nn.Module):
            def __init__(self, dim, heads, dim_head, dropout, regular_lambda, temperature):
                super().__init__()
                self.heads = heads
                self.d_k = math.sqrt(dim_head)
                inner = dim_head * heads
                self.attend = nn.Softmax(dim=-1)
                self.to_q = nn.Linear(dim, inner)
                self.to_k = nn.Linear(dim, inner)
                self.to_v = nn.Linear(dim, inner)
                self.to_out = nn.Sequential(
                    nn.Linear(inner, dim), nn.Dropout(dropout)
                )
                self.dynamical = DynamicalContrastiveLoss(regular_lambda, temperature)

            def forward(self, x, attn_mask=None):
                h = self.heads
                batch, tokens, _ = x.shape
                q = self.to_q(x).reshape(batch, tokens, h, -1)
                k = self.to_k(x).reshape(batch, tokens, h, -1)
                v = self.to_v(x).reshape(batch, tokens, h, -1)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                scale = 1.0 / self.d_k
                scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
                q_norm = torch.norm(q, dim=-1, keepdim=True)
                k_norm = torch.norm(k, dim=-1, keepdim=True)
                norm_matrix = torch.einsum("bhid,bhjd->bhij", q_norm, k_norm)
                dc_loss = None
                if attn_mask is not None:
                    large_negative = -math.log(1e10)
                    attention_mask = torch.where(
                        attn_mask == 0, large_negative, 0
                    )
                    masked_scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                    dc_loss = self.dynamical(scores, attn_mask, norm_matrix)
                else:
                    masked_scores = scores
                attn = self.attend(masked_scores * scale)
                out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
                out = out.permute(0, 2, 1, 3).reshape(batch, tokens, -1)
                return self.to_out(out), attn, dc_loss

        class CTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                PreNorm(
                                    cf_dim,
                                    CAttention(
                                        cf_dim,
                                        heads,
                                        head_dim,
                                        dropout,
                                        regular_lambda,
                                        temperature,
                                    ),
                                ),
                                PreNorm(cf_dim, FeedForward(cf_dim, d_ff, dropout)),
                            ]
                        )
                        for _ in range(depth)
                    ]
                )

            def forward(self, x, attn_mask=None):
                total = 0
                attn = None
                for attn_layer, ff in self.layers:
                    x_n, attn, dc_loss = attn_layer(x, attn_mask=attn_mask)
                    total = total + dc_loss
                    x = x_n + x
                    x = ff(x) + x
                return x, attn, total / len(self.layers)

        class TransC(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_patch_embedding = nn.Sequential(
                    nn.Linear(patch_size * 2, cf_dim), nn.Dropout(dropout)
                )
                self.dropout = nn.Dropout(dropout)
                self.transformer = CTransformer()
                self.mlp_head = nn.Linear(cf_dim, d_model * 2)

            def forward(self, x, attn_mask=None):
                x = self.to_patch_embedding(x)
                x, attn, dc_loss = self.transformer(x, attn_mask)
                x = self.dropout(x)
                return self.mlp_head(x), dc_loss

        class FlattenHead(nn.Module):
            def __init__(self, nf, seq_len, head_dropout):
                super().__init__()
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear1 = nn.Linear(nf, nf)
                self.linear2 = nn.Linear(nf, nf)
                self.linear3 = nn.Linear(nf, nf)
                self.linear4 = nn.Linear(nf, seq_len)
                self.dropout = nn.Dropout(head_dropout)

            def forward(self, x):
                x = self.flatten(x)
                x = torch.nn.functional.relu(self.linear1(x)) + x
                x = torch.nn.functional.relu(self.linear2(x)) + x
                x = torch.nn.functional.relu(self.linear3(x)) + x
                return self.linear4(x)

        class PreNorm(nn.Module):
            def __init__(self, dim, fn):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.fn = fn

            def forward(self, x, **kwargs):
                return self.fn(self.norm(x), **kwargs)

        class FeedForward(nn.Module):
            def __init__(self, dim, hidden_dim, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, dim),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return self.net(x)

        class CATCHModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.revin_layer = RevIN(n_features)
                self.norm = nn.LayerNorm(patch_size)
                self.mask_generator = ChannelMaskGenerator(patch_size, n_features)
                self.frequency_transformer = TransC()
                head_nf_f = d_model * 2 * patch_num
                self.head_f1 = FlattenHead(head_nf_f, seq_len, head_dropout)
                self.head_f2 = FlattenHead(head_nf_f, seq_len, head_dropout)
                self.ircom = nn.Linear(seq_len * 2, seq_len)
                self.get_r = nn.Linear(d_model * 2, d_model * 2)
                self.get_i = nn.Linear(d_model * 2, d_model * 2)

            def forward(self, z):
                z = self.revin_layer(z, "norm")
                z = z.permute(0, 2, 1)
                z = torch.fft.fft(z)
                z1, z2 = z.real, z.imag
                z1 = z1.unfold(dimension=-1, size=patch_size, step=patch_stride)
                z2 = z2.unfold(dimension=-1, size=patch_size, step=patch_stride)
                z1 = z1.permute(0, 2, 1, 3)
                z2 = z2.permute(0, 2, 1, 3)
                batch_size, local_patch_num, c_in, local_patch_size = z1.shape
                z1 = z1.reshape(batch_size * local_patch_num, c_in, local_patch_size)
                z2 = z2.reshape(batch_size * local_patch_num, c_in, local_patch_size)
                z_cat = torch.cat((z1, z2), -1)
                channel_mask = self.mask_generator(z_cat)
                z, dc_loss = self.frequency_transformer(z_cat, channel_mask)
                z1 = self.get_r(z)
                z2 = self.get_i(z)
                z1 = z1.reshape(batch_size, local_patch_num, c_in, -1)
                z2 = z2.reshape(batch_size, local_patch_num, c_in, -1)
                z1 = z1.permute(0, 2, 1, 3)
                z2 = z2.permute(0, 2, 1, 3)
                z1 = self.head_f1(z1)
                z2 = self.head_f2(z2)
                complex_z = torch.complex(z1, z2)
                z = torch.fft.ifft(complex_z)
                z = self.ircom(torch.cat((z.real, z.imag), -1))
                z = z.permute(0, 2, 1)
                return self.revin_layer(z, "denorm"), complex_z.permute(0, 2, 1), dc_loss

        return CATCHModel()

    def _frequency_metric(self, outputs, targets, *, keep_dim, dim):
        torch, _ = torch_modules()
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = torch.fft.fft(outputs, dim=1)
        residual = frequency_outputs - torch.fft.fft(targets, dim=1)
        return residual.abs().mean(dim=dim, keepdim=keep_dim)

    def _frequency_score(self, outputs, targets):
        """frequency_criterion: per-timestamp frequency residual, inferred patching."""
        torch, _ = torch_modules()
        patch = self.inference_patch_size
        stride = self.inference_patch_stride
        window = self.window
        patch_num = int((window - patch) / stride + 1)
        padding_length = window - (patch + (patch_num - 1) * stride)
        output_patch = outputs.unfold(dimension=1, size=patch, step=stride)
        batch, n_patch, channels, local = output_patch.shape
        output_patch = output_patch.reshape(batch * n_patch, local, channels)
        y_patch = targets.unfold(dimension=1, size=patch, step=stride)
        y_patch = y_patch.reshape(batch * n_patch, local, channels)
        main = self._frequency_metric(output_patch, y_patch, keep_dim=True, dim=1)
        main = main.repeat(1, patch, 1).reshape(batch, n_patch, patch, channels)
        end_point = patch + (patch_num - 1) * stride - 1
        main_loss = torch.zeros(
            (batch, n_patch, window - padding_length, channels), device=outputs.device
        )
        for index in range(n_patch):
            start = index * stride
            main_loss[:, index, start:start + patch, :] = main[:, index]
        non_zero = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero
        if padding_length > 0:
            padding_loss = self._frequency_metric(
                outputs[:, -padding_length:, :],
                targets[:, -padding_length:, :],
                keep_dim=True,
                dim=1,
            )
            padding_loss = padding_loss.repeat(1, padding_length, 1)
            return torch.concat([main_loss, padding_loss], dim=1)
        return main_loss

    def fit(self, windows: np.ndarray) -> "CATCHDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device)
        model = self._build(windows.shape[-1]).to(self._device)
        mask_parameters = list(model.mask_generator.parameters())
        mask_ids = {id(parameter) for parameter in mask_parameters}
        main_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in mask_ids
        ]
        optimiser = torch.optim.Adam(main_parameters, lr=self.learning_rate)
        optimiser_mask = torch.optim.Adam(
            mask_parameters, lr=self.mask_learning_rate
        )
        criterion = torch.nn.MSELoss()
        data = torch.from_numpy(windows.astype(np.float32)).to(self._device)
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator).to(self._device)
            for step, start in enumerate(range(0, len(data), self.batch_size)):
                batch = data[order[start:start + self.batch_size]]
                norm_input = model.revin_layer(batch, "norm")
                output, output_complex, dc_loss = model(batch)
                rec_loss = criterion(output, batch)
                auxi_loss = self._frequency_metric(
                    output_complex, norm_input, keep_dim=False, dim=None
                )
                loss = (
                    rec_loss
                    + self.dc_lambda * dc_loss
                    + self.auxi_lambda * auxi_loss
                )
                optimiser.zero_grad()
                optimiser_mask.zero_grad()
                loss.backward()
                optimiser.step()
                if (step + 1) % self.mask_update_every == 0:
                    optimiser_mask.step()
                    optimiser_mask.zero_grad()
        self._model = model
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Reference score: time residual plus score_lambda times frequency residual."""

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
                output, _, _ = model(batch)
                temp_score = torch.mean(criterion(batch, output), dim=-1)
                freq_score = torch.mean(
                    self._frequency_score(batch, output), dim=-1
                )
                scores.append(
                    (temp_score + self.score_lambda * freq_score).cpu().numpy()
                )
        return np.concatenate(scores)


@dataclass
class GCADDetector(Detector):
    """GCAD: anomaly detection from the perspective of Granger causality (AAAI 2025).

    # Grounding: transcribed
    # sources: github.com/Tc99m/GCAD
    #   - models/common.py (RevIN with affine parameters, the TSMixer ResBlock with
    #     its temporal and feature linear paths)
    #   - models/tsmixer.py (TSMixerRevIN: RevIN, stacked residual blocks, a linear
    #     map from the input window to pred_len, then denormalisation)
    #   - test.py (the causality matrix: |d loss_j / d x| per output channel,
    #     symmetrised through the upper/lower triangle difference, thresholded at
    #     sparse_th and averaged; the test score: the mean relative deviation of a
    #     window's causality matrix from the training one, smoothed by a moving
    #     average of three)
    # Deviations recorded rather than hidden:
    #   * the release's dataloader forecasts the pred_len steps that follow the
    #     window; this harness only hands a detector its windows, so the target is
    #     the tail of the window itself and the model predicts it from the head.
    #     The causal graph is computed from the same forecast loss either way, but
    #     the two are not identical objectives.
    #   * the release samples test windows with a stride of 5 (SWaT) or 10 (PSM);
    #     the harness uses one stride for training and testing.
    #   * the release early-stops on a validation split with patience 2; the
    #     harness trains a fixed budget.
    """

    window: int = 30
    pred_len: int = 1
    n_block: int = 3
    ff_dim: int = 1024
    dropout: float = 0.0
    sparse_th: float = 0.005
    sample_p: float = 0.2
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 128
    smoothing_window: int = 3
    target_mode: str = "next_steps"
    seed: int = 1103
    device: str = "auto"
    kind = "window"
    name = "gcad"
    _model: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)
    _train_causal: Any = field(default=None, repr=False)

    def _build(self, n_features: int, input_len: int):
        torch, nn = torch_modules()
        window = input_len
        pred_len = self.pred_len
        n_block = self.n_block
        ff_dim = self.ff_dim
        dropout = self.dropout

        class RevIN(nn.Module):
            def __init__(self, num_features, eps=1e-5, affine=True):
                super().__init__()
                self.num_features = num_features
                self.eps = eps
                self.affine = affine
                if affine:
                    self.affine_weight = nn.Parameter(torch.ones(num_features))
                    self.affine_bias = nn.Parameter(torch.zeros(num_features))

            def forward(self, x, mode, target_slice=None):
                if mode == "norm":
                    self._get_statistics(x)
                    x = self._normalize(x)
                elif mode == "denorm":
                    x = self._denormalize(x, target_slice)
                else:
                    raise NotImplementedError
                return x

            def _get_statistics(self, x):
                dims = tuple(range(1, x.ndim - 1))
                self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
                self.stdev = torch.sqrt(
                    torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps
                ).detach()

            def _normalize(self, x):
                x = (x - self.mean) / self.stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
                return x

            def _denormalize(self, x, target_slice=None):
                if self.affine:
                    x = x - self.affine_bias[target_slice]
                    x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
                x = x * self.stdev[:, :, target_slice]
                return x + self.mean[:, :, target_slice]

        class ResBlock(nn.Module):
            def __init__(self, input_shape, dropout, ff_dim):
                super().__init__()
                self.norm1 = nn.BatchNorm1d(input_shape[0] * input_shape[1])
                self.linear1 = nn.Linear(input_shape[0], input_shape[0])
                self.dropout1 = nn.Dropout(dropout)
                self.norm2 = nn.BatchNorm1d(input_shape[0] * input_shape[1])
                self.linear2 = nn.Linear(input_shape[-1], ff_dim)
                self.dropout2 = nn.Dropout(dropout)
                self.linear3 = nn.Linear(ff_dim, input_shape[-1])
                self.dropout3 = nn.Dropout(dropout)

            def forward(self, x):
                inputs = x
                x = self.norm1(torch.flatten(x, 1, -1)).reshape(x.shape)
                x = torch.transpose(x, 1, 2)
                x = torch.nn.functional.relu(self.linear1(x))
                x = torch.transpose(x, 1, 2)
                x = self.dropout1(x)
                res = x + inputs
                x = self.norm2(torch.flatten(res, 1, -1)).reshape(res.shape)
                x = torch.nn.functional.relu(self.linear2(x))
                x = self.dropout2(x)
                x = self.linear3(x)
                x = self.dropout3(x)
                return x + res

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.rev_norm = RevIN(n_features)
                self.res_blocks = nn.ModuleList(
                    [ResBlock((window, n_features), dropout, ff_dim)
                     for _ in range(n_block)]
                )
                self.linear = nn.Linear(window, pred_len)

            def forward(self, x):
                x = self.rev_norm(x, "norm")
                for res_block in self.res_blocks:
                    x = res_block(x)
                x = torch.transpose(x, 1, 2)
                x = self.linear(x)
                x = torch.transpose(x, 1, 2)
                return self.rev_norm(x, "denorm", slice(None))

        return Model()

    def _causal_matrix(self, model, batch_x, batch_y):
        """|d loss_j / d x| per output channel, symmetrised and thresholded."""
        torch, _ = torch_modules()
        criterion = torch.nn.MSELoss(reduction="sum")
        model.zero_grad()
        batch_x = batch_x.clone().detach().requires_grad_(True)
        outputs = model(batch_x).float()
        for features in range(outputs.shape[-1]):
            model.zero_grad()
            loss_i = criterion(outputs[:, :, features], batch_y[:, :, features])
            loss_i.backward(retain_graph=True)
            grad_i = torch.abs(batch_x.grad)
            batch_x.grad = None
            grad_i = grad_i.unsqueeze(3)
            grad_causal_mat = (
                grad_i if features == 0 else torch.cat([grad_causal_mat, grad_i], dim=3)
            )
        # (batch, input_channels, input_window, output_channels) -> mean over window
        causal = torch.mean(grad_causal_mat, dim=1)
        upper = torch.triu(causal, diagonal=0)
        lower_transposed = torch.tril(causal, diagonal=-1).transpose(1, 2)
        result = torch.triu(upper - lower_transposed, diagonal=0)
        result_upper = torch.where(result < 0, torch.zeros_like(result), result)
        result_lower = torch.where(
            result < 0, torch.abs(result), torch.zeros_like(result)
        ).transpose(1, 2)
        causal = result_upper + result_lower
        zero = torch.zeros_like(causal)
        return torch.where(causal < self.sparse_th, zero, causal)

    def _windows_and_targets(self, windows: np.ndarray):
        """Return (input windows, forecast targets) for the configured mode.

        ``next_steps`` follows the release: the input is the whole window and the
        target is the ``pred_len`` steps that follow it, which the harness can
        recover from the next window whenever windows overlap by ``window - 1``
        (stride 1).  ``window_tail`` exists only for SMD, whose stride of 100
        makes the following steps unavailable through the window interface; it
        forecasts the window's own tail instead and is recorded as a deviation.
        """

        if self.target_mode == "window_tail":
            head = windows[:, : self.window - self.pred_len, :]
            tail = windows[:, self.window - self.pred_len :, :]
            return head, tail
        if len(windows) <= self.pred_len:
            raise ValueError("not enough windows to recover the forecast target")
        head = windows[: len(windows) - self.pred_len]
        tail = windows[self.pred_len :, self.window - self.pred_len :, :]
        return head, tail

    def fit(self, windows: np.ndarray) -> "GCADDetector":
        torch, _ = torch_modules()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._device = resolve_device(self.device)
        head, tail = self._windows_and_targets(windows)
        model = self._build(windows.shape[-1], head.shape[1]).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        data_x = torch.from_numpy(head.astype(np.float32))
        data_y = torch.from_numpy(tail.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data_x), generator=generator)
            for start in range(0, len(data_x), self.batch_size):
                index = order[start:start + self.batch_size]
                batch_x = data_x[index].to(self._device)
                batch_y = data_y[index].to(self._device)
                optimiser.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimiser.step()
        # Training causality matrix: sampled batches, as in the release.
        model.eval()
        collected = []
        with torch.enable_grad():
            for start in range(0, len(data_x), self.batch_size):
                if rng.random() > self.sample_p and collected:
                    continue
                index = torch.arange(start, min(start + self.batch_size, len(data_x)))
                batch_x = data_x[index].to(self._device)
                batch_y = data_y[index].to(self._device)
                collected.append(self._causal_matrix(model, batch_x, batch_y))
        self._train_causal = torch.mean(torch.cat(collected, dim=0), dim=0).detach() + 1e-4
        self._model = model
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        torch, _ = torch_modules()
        model = self._model
        model.eval()
        head, tail = self._windows_and_targets(windows)
        data_x = torch.from_numpy(head.astype(np.float32))
        data_y = torch.from_numpy(tail.astype(np.float32))
        scores = []
        with torch.enable_grad():
            for start in range(0, len(data_x), self.batch_size):
                batch_x = data_x[start:start + self.batch_size].to(self._device)
                batch_y = data_y[start:start + self.batch_size].to(self._device)
                causal = self._causal_matrix(model, batch_x, batch_y)
                relative = torch.abs(causal - self._train_causal) / self._train_causal
                scores.append(torch.mean(relative, dim=(1, 2)).detach().cpu().numpy())
        raw = np.concatenate(scores)
        window = self.smoothing_window
        if window > 1 and len(raw) > window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(raw, kernel, mode="valid")
            pad = (len(raw) - len(smoothed)) // 2
            raw = np.concatenate(
                [np.full(pad, smoothed[0]), smoothed, np.full(len(raw) - len(smoothed) - pad, smoothed[-1])]
            )
        return raw


WINDOW_DETECTORS = {
    "usad": USADDetector,
    "anomaly_transformer": AnomalyTransformerDetector,
    "dcdetector": DCdetectorDetector,
    "timesnet": TimesNetDetector,
    "itransformer": ITransformerDetector,
    "tranad": TranADDetector,
    "catch": CATCHDetector,
    "gcad": GCADDetector,
}


def build_detector(name: str, params: dict | None = None) -> Detector:
    params = dict(params or {})
    if name in POINT_DETECTORS:
        return POINT_DETECTORS[name](**params)
    if name in WINDOW_DETECTORS:
        return WINDOW_DETECTORS[name](**params)
    raise ValueError(f"unknown detector: {name}")
