"""Epsilon-continuation numerical repair for the unchanged entropic FM coupling.

The final squared-distance objective uses exactly 2*sigma**2 regularization,
uniform marginals and the original 1e-6 absolute marginal tolerance. Row/column
cost shifts do not change the coupling. No exact-OT/uniform fallback, relaxed
tolerance or rescaled final regularization is used. The original implementation
and failed artifacts remain unchanged; corrected runs have separate configs.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from .fm_objective_detectors import CFMObjectiveDetector, probability_path


def stable_plan(x0, x1, regularization, tolerance=1e-6, max_iterations=20000):
    if len(x0) != len(x1) or not len(x0) or regularization <= 0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid uniform entropic transport problem')
    cost = torch.cdist(x0.flatten(1).double(), x1.flatten(1).double()).square().detach().cpu().numpy()
    if not np.isfinite(cost).all():
        raise ValueError('Nonfinite OT cost')
    # Adding row/column potentials leaves all feasible coupling objectives
    # shifted by a constant; the optimal coupling is unchanged.
    cost = cost - cost.min(1, keepdims=True)
    cost = cost - cost.min(0, keepdims=True)
    n = len(cost)
    log_mass = -math.log(n)
    initial = max(float(regularization), float(cost.std()))
    temperatures = np.geomspace(initial, regularization, 12) if initial > regularization else [regularization]
    log_v, previous = np.zeros(n), initial
    for index, temperature in enumerate(temperatures):
        log_v *= previous / temperature
        previous = temperature
        kernel = -cost / temperature
        iterations = max_iterations if index == len(temperatures) - 1 else 300
        for iteration in range(iterations):
            log_u = log_mass - logsumexp(kernel + log_v[None, :], axis=1)
            log_v = log_mass - logsumexp(kernel + log_u[:, None], axis=0)
            if iteration % 10 == 0 or iteration == iterations - 1:
                plan = np.exp(kernel + log_u[:, None] + log_v[None, :])
                error = max(abs(plan.sum(0) - 1 / n).max(), abs(plan.sum(1) - 1 / n).max())
                if error <= tolerance:
                    break
        # Intermediate temperatures only warm start the final, unchanged target.
    if not np.isfinite(plan).all() or error > tolerance:
        raise RuntimeError(f'Epsilon-continuation Sinkhorn did not converge: marginal error={error:.3g}')
    return plan


class StableCFMObjectiveDetector(CFMObjectiveDetector):
    """Identical SB/SF2M path/backbone/RNG/training; repaired OT numerics only."""

    def fit(self, windows):
        if self.objective not in ('sb', 'sf2m'):
            raise ValueError('Numerical repair is registered only for entropic objectives')
        values = self._windows(windows)
        if not len(values):
            raise ValueError('No training windows')
        self._shape = tuple(values.shape[1:])
        dimensions = int(np.prod(self._shape))
        self._device = torch.device('cuda' if self.device == 'auto' and torch.cuda.is_available() else 'cpu' if self.device == 'auto' else self.device)
        self._score_head = self.objective == 'sf2m' and self.score_weight > 0
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self._model = nn.Sequential(nn.Linear(dimensions + 1, self.hidden_size), nn.SiLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(),
                                        nn.Linear(self.hidden_size, dimensions * (2 if self._score_head else 1)))
        self._model.to(self._device)
        generator = torch.Generator(device=self._device).manual_seed(self.seed)
        tensor = torch.from_numpy(np.ascontiguousarray(values.reshape(len(values), -1)))
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        self.training_losses_ = []
        for _ in range(self.epochs):
            order = torch.randperm(len(values), generator=generator, device=self._device).cpu()
            total = 0.
            for start in range(0, len(values), self.batch_size):
                x1 = tensor[order[start:start + self.batch_size]].to(self._device)
                x0 = torch.randn(x1.shape, generator=generator, device=self._device)
                plan = stable_plan(x0, x1, 2 * self.sigma ** 2)
                weights = torch.as_tensor(plan.ravel(), dtype=torch.float64, device=self._device)
                pairs = torch.multinomial(weights, len(x0), replacement=True, generator=generator)
                x0, x1 = x0[pairs // len(x1)], x1[pairs % len(x1)]
                time = torch.rand(len(x1), generator=generator, device=self._device).clamp(1e-4, 1 - 1e-4)
                noise = torch.randn(x1.shape, generator=generator, device=self._device)
                state, velocity, std = probability_path(x0, x1, time, noise, self.objective, self.sigma)
                output = self._model(torch.cat([state, time[:, None]], dim=1))
                loss = (output[:, :dimensions] - velocity).square().mean()
                if self._score_head:
                    weight = 2 * std / (self.sigma ** 2 + 1e-8)
                    loss = loss + self.score_weight * (weight * output[:, dimensions:] + noise).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite FM training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                optimizer.step()
                total += float(loss.detach()) * len(x1)
            self.training_losses_.append(total / len(values))
        self._model.eval()
        return self
