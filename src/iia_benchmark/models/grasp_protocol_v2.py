"""GRASP paper-protocol reconstruction; unresolved conventions stay explicit.

Primary source arXiv:2609.36765, pp.3/6/8/23/24/25. This module is a
successor, and does not modify frozen fm_native.py or existing experiment jobs.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fm_native import GraphSpectralPath, MixerVelocity, normalized_laplacian


def gaussian_distance_graph(features, threshold=.5):
    """Gaussian kernel on Euclidean channel trajectories, an explicit choice.

    The paper specifies the kernel and std(distance matrix), but not its distance
    estimator. Include both triangles and diagonal in population std (ddof=0).
    Identical trajectories use the continuous zero-distance kernel limit.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all() or not 0 < threshold <= 1:
        raise ValueError('Finite time-by-channel features and threshold in (0,1] required')
    # Direct differences avoid cancellation in large, near-identical trajectories.
    distance = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for i in range(x.shape[1]):
        for j in range(i):
            distance[i, j] = distance[j, i] = np.linalg.norm(x[:, i] - x[:, j])
    scale = float(distance.std(ddof=0))
    kernel = np.ones_like(distance) if scale == 0 else np.exp(-(distance / scale)**2)
    adjacency = (kernel >= threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 0)
    return adjacency, {'distance': distance, 'kernel': kernel, 'scale': scale,
                       'estimator': 'euclidean_channel_training_trajectory_local_choice',
                       'std_population': 'all_matrix_entries_including_zero_diagonal'}


def fit_only_scaler(train, validation, normalization='train_zscore'):
    train, validation = np.asarray(train, dtype=np.float64), np.asarray(validation, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError('Train/validation feature shapes differ')
    clean = np.where(np.isfinite(train), train, np.nan)
    median = np.nanmedian(clean, axis=0)
    if not np.isfinite(median).all():
        raise ValueError('A fitting channel has no finite observations')
    repaired = []
    repairs = []
    for values in (train, validation):
        values = values.copy()
        bad = ~np.isfinite(values)
        repairs.append(int(bad.sum()))
        values[bad] = np.broadcast_to(median, values.shape)[bad]
        repaired.append(values)
    mean, std = repaired[0].mean(0), repaired[0].std(0)
    safe = np.where(std > 0, std, 1.)
    if normalization == 'none':
        mean, safe = np.zeros_like(mean), np.ones_like(safe)
    elif normalization != 'train_zscore':
        raise ValueError('Unknown normalization')
    normalized = [((v - mean) / safe).astype(np.float32) for v in repaired]
    if not all(np.isfinite(v).all() for v in normalized):
        raise ValueError('Normalization overflow')
    return normalized, repaired[0], {'median': median, 'mean': mean, 'scale': safe,
                                     'constant_channels': np.flatnonzero(std == 0), 'repairs': repairs,
                                     'normalization': normalization}


class SeriesWindows(torch.utils.data.Dataset):
    """Stride-one windows with no materialized repeated training array."""
    def __init__(self, values, window):
        self.values = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        self.window = window
        if self.values.ndim != 2 or len(self.values) < window or window < 1 or not torch.isfinite(self.values).all():
            raise ValueError('Finite sequence at least one full window required')

    def __len__(self):
        return len(self.values) - self.window + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.values[index:index + self.window].T.contiguous()


def _embedding(time, frequencies):
    t = time.reshape(-1, 1) * frequencies[None, :]
    return torch.cat((t.sin(), t.cos()), dim=-1)


class _ResidualGCN(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, propagation):
        message = torch.einsum('cd,bdh->bch', propagation, self.norm(x))
        return x + self.dropout(F.relu(self.linear(message)))


class GNNVelocity(nn.Module):
    """Appendix E.6: channel nodes, two residual GCN blocks, hidden 128.

    Concatenation along per-channel temporal features, pre-LayerNorm, ReLU,
    self-loop GCN normalization and output projection are local conventions.
    """
    def __init__(self, adjacency, window=50, hidden=128, blocks=2, dropout=.1):
        super().__init__()
        a = torch.as_tensor(adjacency, dtype=torch.float32)
        a = a + torch.eye(len(a))
        degree = a.sum(1).rsqrt()
        self.register_buffer('propagation', degree[:, None] * a * degree[None, :])
        self.register_buffer('frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 32))
        self.input = nn.Linear(window + 64, hidden)
        self.blocks = nn.ModuleList([_ResidualGCN(hidden, dropout) for _ in range(blocks)])
        self.output = nn.Linear(hidden, window)

    def forward(self, time, x):
        e = _embedding(time, self.frequencies)[:, None, :].expand(-1, x.shape[1], -1)
        hidden = self.input(torch.cat((x, e), dim=-1))
        for block in self.blocks:
            hidden = block(hidden, self.propagation)
        return self.output(hidden)


class TransformerVelocity(nn.Module):
    """Appendix E.6: channel tokens, two encoders and four heads, hidden 128.

    Input/output projection, pre-LayerNorm, ReLU, no positional encoding, and
    FFN width 128 are explicit local choices; the original source is unavailable.
    """
    def __init__(self, channels, window=50, hidden=128, blocks=2, dropout=.1, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError('Hidden dimension must be divisible by attention heads')
        self.register_buffer('frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 32))
        self.input = nn.Linear(window + 64, hidden)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden,
                                                              dropout=dropout, activation='relu',
                                                              batch_first=True, norm_first=True)
                                     for _ in range(blocks)])
        self.output = nn.Linear(hidden, window)

    def forward(self, time, x):
        e = _embedding(time, self.frequencies)[:, None, :].expand(-1, x.shape[1], -1)
        hidden = self.input(torch.cat((x, e), dim=-1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class GRASPProtocolDetector:
    """Per-entity, stride-one GRASP with a fixed full training/validation budget.

    fit accepts only fitting and validation features, never test labels. The
    controller performs the chronological 80/20 split before passing them here.
    No early stopping is invented: last epoch is used, and validation loss is
    logged every 50 epochs. Tests can use a separately declared small budget.
    """
    def __init__(self, window=50, hidden=128, blocks=2, dropout=.1, tau=2.,
                 graph_threshold=.5, variant='grasp', architecture='tsmixer', epochs=1500,
                 validation_interval=50, batch_size=256, learning_rate=.001,
                 source_samples=5, flow_evaluations=10, normalization='train_zscore',
                 graph_features='raw_repaired_fit', seed=1103, device='auto'):
        if variant not in {'grasp', 'mean', 'er', 'lin'} or architecture not in {'tsmixer', 'gnn', 'transformer'}:
            raise ValueError('Unsupported variant/architecture')
        if graph_features not in {'raw_repaired_fit', 'normalized_fit'}:
            raise ValueError('Graph features must be a disclosed fitting-only choice')
        if min(window, hidden, blocks, epochs, validation_interval, batch_size, source_samples, flow_evaluations) < 1:
            raise ValueError('Positive budgets required')
        self.window, self.hidden, self.blocks, self.dropout = window, hidden, blocks, dropout
        self.tau, self.graph_threshold, self.variant, self.architecture = tau, graph_threshold, variant, architecture
        self.epochs, self.validation_interval, self.batch_size = epochs, validation_interval, batch_size
        self.learning_rate, self.source_samples, self.flow_evaluations = learning_rate, source_samples, flow_evaluations
        self.normalization, self.graph_features, self.seed = normalization, graph_features, seed
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)

    def prepare(self, train, validation):
        arrays, raw, scaler = fit_only_scaler(train, validation, self.normalization)
        graph_input = raw if self.graph_features == 'raw_repaired_fit' else arrays[0]
        adjacency, graph = gaussian_distance_graph(graph_input, self.graph_threshold)
        if self.variant == 'er':
            # G(n,m), a disclosed exact-sparsity Erdős–Rényi convention.
            i, j = np.triu_indices(len(adjacency), 1)
            count = int(np.count_nonzero(adjacency[i, j]))
            selected = np.random.default_rng(self.seed).choice(len(i), count, replace=False)
            adjacency = np.zeros_like(adjacency)
            adjacency[i[selected], j[selected]] = adjacency[j[selected], i[selected]] = 1
        self.scaler_, self.graph_audit_, self.adjacency_ = scaler, graph, adjacency
        self.fit_windows_, self.validation_windows_ = SeriesWindows(arrays[0], self.window), SeriesWindows(arrays[1], self.window)
        return arrays

    def _build(self):
        self.path = GraphSpectralPath(normalized_laplacian(self.adjacency_), 0 if self.variant == 'lin' else self.tau).to(self.device)
        channels = len(self.adjacency_)
        if self.architecture == 'gnn':
            model = GNNVelocity(self.adjacency_, self.window, self.hidden, self.blocks, self.dropout)
        elif self.architecture == 'transformer':
            model = TransformerVelocity(channels, self.window, self.hidden, self.blocks, self.dropout)
        else:
            model = MixerVelocity(channels, self.window, self.hidden, self.blocks, self.dropout)
        self.model = model.to(self.device)

    def _fixed_validation_loss(self):
        self.model.eval()
        rng = torch.Generator(device=self.device).manual_seed(self.seed + 1000000)
        total = 0.
        loader = torch.utils.data.DataLoader(self.validation_windows_, batch_size=self.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for target in loader:
                target = target.to(self.device)
                source = torch.randn(target.shape, device=self.device, generator=rng)
                time = torch.rand(len(target), device=self.device, generator=rng)
                state, velocity = self.path(source, target, time)
                total += float(F.mse_loss(self.model(time, state), velocity)) * len(target)
        return total / len(self.validation_windows_)

    def fit(self, train, validation, callback=None):
        self.prepare(train, validation)
        cuda_devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self._build()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            loader = torch.utils.data.DataLoader(self.fit_windows_, batch_size=self.batch_size, shuffle=True, num_workers=0)
            self.history_, self.optimizer_updates_ = [], 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                total = 0.
                for target in loader:
                    target = target.to(self.device)
                    source = torch.randn_like(target)
                    time = torch.rand(len(target), device=self.device)
                    state, velocity = self.path(source, target, time)
                    loss = F.mse_loss(self.model(time, state), velocity)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss; no budget/precision fallback')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.optimizer_updates_ += 1
                    total += float(loss.detach()) * len(target)
                record = {'epoch': epoch, 'train_loss': total / len(self.fit_windows_), 'optimizer_updates': self.optimizer_updates_}
                if epoch % self.validation_interval == 0 or epoch == self.epochs:
                    record['validation_loss'] = self._fixed_validation_loss()
                    if not math.isfinite(record['validation_loss']):
                        raise FloatingPointError('Nonfinite validation loss')
                self.history_.append(record)
                if callback:
                    callback(record)
        self.model.eval()
        self.epochs_completed_ = self.epochs
        self.checkpoint_selection_ = 'last_epoch_local_choice_no_invented_early_stop'
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != len(self.adjacency_):
            raise ValueError('Scoring feature shape differs')
        bad = ~np.isfinite(values)
        values[bad] = np.broadcast_to(self.scaler_['median'], values.shape)[bad]
        out = ((values - self.scaler_['mean']) / self.scaler_['scale']).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError('Scoring normalization overflow')
        return out

    def score(self, values, *, source_samples=None, flow_evaluations=None, batch_size=None, split_tag=0,
              return_details=False):
        if not hasattr(self, 'model'):
            raise RuntimeError('Fit before scoring')
        samples = self.source_samples if source_samples is None else source_samples
        evaluations = self.flow_evaluations if flow_evaluations is None else flow_evaluations
        batch = self.batch_size if batch_size is None else batch_size
        if min(samples, evaluations, batch) < 1 or split_tag < 0:
            raise ValueError('Positive scoring budgets required')
        windows = SeriesWindows(self.transform(values), self.window)
        times = torch.arange(1, evaluations + 1, device=self.device, dtype=torch.float32) / (evaluations + 1)
        loader = torch.utils.data.DataLoader(windows, batch_size=batch, shuffle=False, num_workers=0)
        all_scores, offset = [], 0
        raw_energy = np.zeros((evaluations, len(self.adjacency_)), dtype=np.float64)
        weighted_energy = np.zeros_like(raw_energy)
        self.model.eval()
        with torch.no_grad():
            for targets in loader:
                targets = targets.to(self.device)
                # Independent per-window priors, stateless window IDs, same sample
                # prefixes for sensitivity sweeps; independent of score batching.
                sources = np.stack([np.random.default_rng(np.random.SeedSequence([self.seed, 91237, split_tag, offset + i])).standard_normal(
                    (samples, targets.shape[1], self.window)).astype(np.float32) for i in range(len(targets))])
                sources = torch.as_tensor(sources, device=self.device)
                result = torch.zeros(len(targets), device=self.device)
                for sample in range(samples):
                    for time_index, time in enumerate(times):
                        t = time.expand(len(targets))
                        state, velocity = self.path(sources[:, sample], targets, t)
                        residual = self.path.spectral(self.model(t, state) - velocity).square()
                        weight = self.path.weights(t) if self.variant == 'grasp' else 1.
                        result += (residual * weight).sum((1, 2))
                        if return_details:
                            raw_energy[time_index] += residual.double().sum((0, 2)).cpu().numpy() / samples
                            weighted_energy[time_index] += (residual * weight).double().sum((0, 2)).cpu().numpy() / samples
                all_scores.append((result / samples).cpu().numpy())
                offset += len(targets)
        window_scores = np.concatenate(all_scores)
        if not np.isfinite(window_scores).all():
            raise FloatingPointError('Nonfinite full-window score')
        # Every source timestamp is covered. Uniform overlap averaging is a
        # declared local convention: the paper does not identify its mapping.
        totals = np.zeros(len(values), dtype=np.float64)
        counts = np.zeros(len(values), dtype=np.int64)
        for start in range(len(window_scores)):
            totals[start:start + self.window] += window_scores[start]
            counts[start:start + self.window] += 1
        if not counts.all():
            raise ValueError('Incomplete timestamp coverage')
        scores = totals / counts
        if not return_details:
            return scores
        bins = np.array_split(np.arange(len(self.adjacency_)), 8)
        heatmaps = {}
        for name, energy in [('raw', raw_energy), ('weighted', weighted_energy)]:
            grouped = np.stack([energy[:, indices].sum(1) for indices in bins], axis=1)
            heatmaps[name] = np.zeros_like(grouped) if grouped.sum() == 0 else grouped / grouped.sum() * 100
        return scores, {'window_scores': window_scores, 'overlap_counts': counts,
                        'mode_energy_raw': raw_energy, 'mode_energy_weighted': weighted_energy,
                        'quantile_heatmap_raw_percent': heatmaps['raw'],
                        'quantile_heatmap_weighted_percent': heatmaps['weighted'],
                        'mode_bins': [indices.tolist() for indices in bins],
                        'times': times.cpu().numpy(),
                        'attribution_population': 'all_scored_windows_local_choice',
                        'eigenvalue_ties_may_split_quantiles': True}
