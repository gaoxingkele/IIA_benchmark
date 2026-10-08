"""Paper-grounded comparator cores and window adapters, with explicit gaps.

DT-LA: doi:10.1109/ACCESS.2025.3594473, Eqs. 3--14.
SHCL: doi:10.1016/j.ins.2025.122680, Eqs. 1--14.
KGL: doi:10.1145/3778450.3778526, Eqs. 1--13.
These are independent reconstructions, not author code or certified replications.
Architecture choices absent/ambiguous in the papers are recorded in model configs.
No thresholds, test labels, point adjustment or dataset paths enter these models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def modified_reverse_huber(deviation: torch.Tensor, delta: float) -> torch.Tensor:
    """DT-LA Eq. 6, including its published discontinuity at delta != 1."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    return torch.where(deviation <= delta, deviation.square(), delta / deviation.clamp_min(1e-12))


def scaled_softmax(scores: torch.Tensor) -> torch.Tensor:
    """DT-LA Eq. 14: softmax over timestamps times absolute magnitude."""
    return scores.softmax(dim=-1) * scores.abs()


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        y, weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x)), weights


class DTLACore(nn.Module):
    """Primary encoder -> linear input reconstruction -> secondary encoder."""
    def __init__(self, channels: int, d_model: int = 512, heads: int = 8,
                 layers: int = 3, delta: float = 1.0, sparse_weight: float = 0.5,
                 latent_loss: str = "mrh", score_mode: str = "scaled_softmax"):
        super().__init__()
        if latent_loss not in {"mrh", "l2"} or score_mode not in {"input", "latent", "softmax", "scaled_softmax"}:
            raise ValueError("unknown DT-LA loss/score ablation")
        self.embedding = nn.Linear(channels, d_model)
        self.primary = nn.ModuleList([AttentionBlock(d_model, heads) for _ in range(layers)])
        self.secondary_embedding = nn.Linear(channels, d_model)
        self.secondary = nn.ModuleList([AttentionBlock(d_model, heads) for _ in range(layers)])
        self.projection = nn.Linear(d_model, channels)
        self.delta, self.sparse_weight = delta, sparse_weight
        self.latent_loss, self.score_mode = latent_loss, score_mode

    def forward(self, x):
        z = self.embedding(x)
        maps = []
        for block in self.primary:
            z, weights = block(z)
            maps.append(weights)
        reconstruction = self.projection(z)
        zhat = self.secondary_embedding(reconstruction)
        for block in self.secondary:
            zhat, _ = block(zhat)
        return reconstruction, z, zhat, maps

    def loss(self, x):
        reconstruction, z, zhat, maps = self(x)
        deviation = torch.linalg.vector_norm(z - zhat, dim=-1)
        latent = modified_reverse_huber(deviation, self.delta) if self.latent_loss == "mrh" else deviation.square()
        # Eq. 9 sums keys and queries. Average batch/heads/layers; see config.
        entropy = torch.stack([-(m * m.clamp_min(1e-12).log()).sum((-1, -2)).mean() for m in maps]).mean()
        return (x - reconstruction).square().sum(-1).mean() + latent.mean() + self.sparse_weight * entropy

    def score(self, x):
        reconstruction, z, zhat, _ = self(x)
        inp = (x - reconstruction).square().sum(-1)
        lat = (z - zhat).square().sum(-1)
        if self.score_mode == "input":
            return inp
        if self.score_mode == "latent":
            return lat
        weights = lat.softmax(-1) if self.score_mode == "softmax" else scaled_softmax(lat)
        return inp * weights


def hierarchical_masks(length: int, branch: str, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """SHCL Eqs. 2/4; pair-even retains indices 2,3 modulo four."""
    index = torch.arange(length, device=device)
    if branch == "point":
        even = index % 2 == 0
    elif branch == "pair":
        even = index % 4 >= 2
    else:
        raise ValueError("branch must be point or pair")
    return even, ~even


def gaussian_kl(mu_even, logvar_even, mu_odd, logvar_odd):
    """SHCL Eq. 12 directed Gaussian KL; returns per-timestamp values."""
    return 0.5 * (logvar_odd - logvar_even +
                  (logvar_even.exp() + (mu_even - mu_odd).square()) / logvar_odd.exp() - 1).sum(-1)


class SHCLTransformerCore(nn.Module):
    """Five SHCL paper backbones under one THM/loss/score contract.

    Name preserved for the initial Transformer API. Shared weights within each
    point/pair branch; exact unpublished backbone widths are config choices.
    """
    def __init__(self, channels: int, d_model: int = 128, heads: int = 8,
                 layers: int = 3, branches: tuple[str, ...] = ("point", "pair"),
                 masking: str = "thm", objective: str = "heterogeneity",
                 backbone: str = "transformer"):
        super().__init__()
        if masking not in {"thm", "random"} or objective not in {"heterogeneity", "reconstruction"}:
            raise ValueError("unsupported SHCL ablation")
        if not branches or not set(branches) <= {"point", "pair"}:
            raise ValueError("invalid branches")
        if backbone not in {"transformer", "vae", "cnn", "rnn", "lstm"}:
            raise ValueError("unknown SHCL backbone")
        self.branches, self.masking, self.objective, self.backbone = branches, masking, objective, backbone
        self.embedding = nn.Linear(channels, d_model)
        def encoder():
            if backbone == "transformer":
                return nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    d_model, heads, 4 * d_model, dropout=0, batch_first=True), layers)
            if backbone == "cnn":
                return nn.Sequential(*[SHCLCNNBlock(d_model) for _ in range(layers)])
            if backbone == "vae":
                return SHCLVAERepresentation(d_model)
            cls = nn.RNN if backbone == "rnn" else nn.LSTM
            return cls(d_model, d_model, num_layers=layers, batch_first=True)
        self.encoders = nn.ModuleDict({b: encoder() for b in branches})
        self.stats = nn.ModuleDict({b: nn.Linear(d_model, 2 * d_model) for b in branches})
        self.reconstruction = nn.ModuleDict({b: nn.Linear(d_model, channels) for b in branches})

    def _represent(self, h, branch):
        if self.backbone == "vae":
            return self.encoders[branch](h)
        representation = self.encoders[branch](h)
        if self.backbone in {"rnn", "lstm"}:
            representation = representation[0]
        mu, logvar = self.stats[branch](representation).chunk(2, -1)
        return representation, mu, logvar.clamp(-12, 12)

    def forward(self, x):
        h = self.embedding(x)
        # Standard sinusoidal positional encoding: choice not fixed in paper.
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, h.shape[-1], 2, device=x.device, dtype=x.dtype) * (-np.log(10000.0) / h.shape[-1]))
        pe = torch.zeros_like(h[0])
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq[:pe[:, 1::2].shape[-1]])
        h = h + pe
        results = {}
        for branch in self.branches:
            even, odd = hierarchical_masks(x.shape[1], branch, x.device)
            if self.masking == "random":
                # Frozen random mask for repeatable train/test ablation. The
                # paper does not publish the random/adaptive mask algorithm.
                generator = torch.Generator(device="cpu").manual_seed(1103)
                even = (torch.rand(x.shape[1], generator=generator) >= 0.5).to(x.device)
                odd = ~even
            e = self._represent(h * even[None, :, None], branch)
            o = self._represent(h * odd[None, :, None], branch)
            results[branch] = (e, o)
        return results

    def _values(self, x):
        outputs = self(x)
        result = torch.zeros(x.shape[:2], device=x.device)
        for branch, (even, odd) in outputs.items():
            if self.objective == "reconstruction":
                result = result + 0.5 * ((self.reconstruction[branch](even[0]) - x).square().mean(-1) +
                                         (self.reconstruction[branch](odd[0]) - x).square().mean(-1))
            else:
                result = result + (even[0] - odd[0]).square().mean(-1) + gaussian_kl(even[1], even[2], odd[1], odd[2])
        return result

    def loss(self, x):
        return self._values(x).mean()

    def score(self, x):
        return self._values(x)


class SHCLCNNBlock(nn.Module):
    """Depthwise temporal convolution + channel mixing (paper Eq. 17)."""
    def __init__(self, width):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.pointwise(F.gelu(self.depthwise(x.transpose(1, 2)))).transpose(1, 2)
        return self.norm(x + y)


class SHCLVAERepresentation(nn.Module):
    """Per-time VAE representation following the released MLP branch layout.

    Exact publisher release is separately preserved. Inference uses mean latent
    for deterministic continuous scoring, an explicitly declared protocol choice.
    """
    def __init__(self, width):
        super().__init__()
        self.hidden = nn.Linear(width, 256)
        self.mean = nn.Linear(256, width)
        self.logvariance = nn.Linear(256, width)
        self.decoder = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, width), nn.Sigmoid())

    def forward(self, x):
        hidden = F.relu(self.hidden(x))
        mu, logvar = self.mean(hidden), self.logvariance(hidden).clamp(-12, 12)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decoder(z), mu, logvar


class BSplineActivation(nn.Module):
    """Learnable per-output cubic B-splines; KGL Eq. 7, Cox-de Boor basis."""
    def __init__(self, width: int, intervals: int = 8, degree: int = 3, bound: float = 3.0):
        super().__init__()
        self.degree = degree
        self.register_buffer("knots", torch.linspace(-bound, bound, intervals + 1 + 2 * degree))
        n_basis = len(self.knots) - degree - 1
        self.coefficients = nn.Parameter(torch.randn(width, n_basis) * 0.05)

    def basis(self, x):
        x = x.unsqueeze(-1)
        k = self.knots
        basis = ((x >= k[:-1]) & (x < k[1:])).to(x.dtype)
        for degree in range(1, self.degree + 1):
            count = len(k) - degree - 1
            left = (x - k[:count]) / (k[degree:degree + count] - k[:count])
            right = (k[degree + 1:degree + 1 + count] - x) / (k[degree + 1:degree + 1 + count] - k[1:count + 1])
            basis = left * basis[..., :count] + right * basis[..., 1:count + 1]
        return basis

    def forward(self, x):
        return (self.basis(x) * self.coefficients).sum(-1)


class KGLCore(nn.Module):
    """KGL GAT->B-spline activation->global pooling->LSTM core.

    Paper specifies neither output/training objective nor graph construction.
    This implementation explicitly chooses dense sensor graph, mean pooling,
    next-step MSE prediction. It is a runnable architecture reconstruction only.
    """
    def __init__(self, channels: int, d_model: int = 64, heads: int = 4,
                 use_gat: bool = True, use_kan: bool = True, use_lstm: bool = True):
        super().__init__()
        self.use_gat, self.use_lstm = use_gat, use_lstm
        self.channels, self.heads, self.width = channels, heads, d_model
        self.sensor_embed = nn.Linear(1, heads * d_model)
        self.attn_left = nn.Parameter(torch.randn(heads, d_model) * 0.05)
        self.attn_right = nn.Parameter(torch.randn(heads, d_model) * 0.05)
        self.activation = BSplineActivation(heads * d_model) if use_kan else nn.ReLU()
        width = heads * d_model
        self.temporal = nn.LSTM(width, width, batch_first=True) if use_lstm else nn.Identity()
        self.output = nn.Linear(width, channels)

    def forward(self, x):
        b, t, sensors = x.shape
        h = self.sensor_embed(x[..., None]).reshape(b, t, sensors, self.heads, self.width)
        if self.use_gat:
            left = (h * self.attn_left).sum(-1).permute(0, 1, 3, 2)
            right = (h * self.attn_right).sum(-1).permute(0, 1, 3, 2)
            weights = F.leaky_relu(left[..., :, None] + right[..., None, :], 0.2).softmax(-1)
            h = torch.einsum("bthij,btjhd->btihd", weights, h)
        h = self.activation(h.reshape(b, t, sensors, -1)).mean(2)
        if self.use_lstm:
            h, _ = self.temporal(h)
        return self.output(h)

    def loss(self, x):
        if x.shape[1] < 2:
            raise ValueError("prediction objective needs at least two timestamps")
        return F.mse_loss(self(x[:, :-1]), x[:, 1:])

    def score(self, x):
        scores = (self(x[:, :-1]) - x[:, 1:]).square().mean(-1)
        # First point has no predecessor inside window: explicit zero.
        return F.pad(scores, (1, 0))


@dataclass
class ComparatorWindowDetector:
    method: str
    options: dict[str, Any] = field(default_factory=dict)
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-4
    seed: int = 1103
    device: str = "cpu"
    kind: str = field(default="window", init=False)
    _model: Any = field(default=None, repr=False, init=False)

    @property
    def name(self):
        return self.method

    def _input(self, windows):
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
            raise ValueError("finite nonempty [batch,time,channel] windows required")
        return torch.from_numpy(values)

    def fit(self, windows):
        data = self._input(windows)
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("positive epochs and batch_size required")
        constructors = {"dt_la": DTLACore, "shcl_transformer": SHCLTransformerCore, "kgl": KGLCore}
        if self.method not in constructors:
            raise ValueError("unknown comparator")
        # CPU default leaves ongoing GPU queues untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._model = constructors[self.method](data.shape[-1], **self.options).to(self.device)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
            self._model.train()
            for _ in range(self.epochs):
                for ids in torch.randperm(len(data)).split(self.batch_size):
                    batch = data[ids].to(self.device)
                    optimizer.zero_grad()
                    loss = self._model.loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite comparator loss")
                    loss.backward()
                    optimizer.step()
        return self

    def score(self, windows):
        if self._model is None:
            raise RuntimeError("fit before score")
        data = self._input(windows)
        self._model.eval()
        with torch.no_grad():
            return np.concatenate([self._model.score(batch.to(self.device)).cpu().numpy()
                                   for batch in data.split(self.batch_size)])
