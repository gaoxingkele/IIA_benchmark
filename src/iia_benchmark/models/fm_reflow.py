"""Iterative Rectified Flow with persisted learned endpoint couplings.

Transcribes gnobitab/RectifiedFlow 5a1fd4d ImageGeneration/losses.py and
run_lib_reflow.py (uniform-time L2 reflow, warm start, reset Adam). The shared
MLP, epoch budget, fixed Heun integration, absence of EMA and independent
velocity-residual TSAD score are local adaptations, not image reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from .fm_objective_detectors import CFMObjectiveDetector


def transport_endpoints(model, origins, steps, solver):
    """Integrate paired origins without resampling or changing pair order."""
    import torch
    if steps < 1 or solver not in ('euler', 'heun'):
        raise ValueError('Invalid reflow ODE settings')
    if origins.ndim != 2 or not torch.isfinite(origins).all():
        raise ValueError('Finite flattened origins required')
    x = origins.detach().clone()
    action = torch.zeros(len(x), device=x.device, dtype=torch.float64)
    with torch.no_grad():
        for step in range(steps):
            t = torch.full((len(x), 1), step / steps, dtype=x.dtype, device=x.device)
            velocity = model(torch.cat([x, t], 1))
            if velocity.shape != x.shape:
                raise ValueError('Velocity shape differs from state')
            if solver == 'heun':
                next_velocity = model(torch.cat([x + velocity / steps, t + 1 / steps], 1))
                velocity = (velocity + next_velocity) / 2
            if not torch.isfinite(velocity).all():
                raise FloatingPointError('Nonfinite reflow trajectory')
            x = x + velocity / steps
            action += velocity.double().square().mean(1) / steps
    displacement = (x - origins).double().square().mean(1)
    return x.detach(), {'discrete_excess_action_mean': float((action - displacement).mean()),
                        'squared_displacement_per_coordinate': float(displacement.mean()),
                        'definition': 'Mean per coordinate integral of squared effective step velocity minus squared endpoint displacement; finite-step numerical diagnostic.'}


def pair_digest(z0, z1):
    digest = hashlib.sha256()
    for value in (z0, z1):
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        digest.update(json.dumps({'shape': list(array.shape), 'dtype': 'float32'}, sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class ReflowCFMDetector(CFMObjectiveDetector):
    reflow_stages: int = 2
    reflow_epochs: int = 20
    reflow_ode_steps: int = 50
    reflow_solver: str = 'heun'
    reflow_time_epsilon: float = 1e-3
    stage_records_: list = field(default_factory=list, init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.objective != 'rectified' or self.sigma != 0:
            raise ValueError('Reflow requires zero-noise straight-path objective')
        if self.reflow_stages < 2 or self.reflow_epochs < 1 or self.reflow_ode_steps < 1:
            raise ValueError('At least two flows and positive budgets required')
        if self.reflow_solver not in ('euler', 'heun') or not 0 < self.reflow_time_epsilon < 1:
            raise ValueError('Invalid integration or time schedule')

    def fit(self, windows):
        import torch
        super().fit(windows)
        count = len(windows)
        dimensions = int(np.prod(self._shape))
        self.stage_records_ = [{'stage': 1, 'kind': 'independent_rectification', 'epochs': self.epochs,
                                'training_windows': count, 'losses': list(self.training_losses_)}]
        # Modules with buffers expose every pair and teacher checkpoint to the
        # existing runner without adding trainable parameters or GPU residency.
        self._reflow_evidence = torch.nn.Module()
        for stage in range(2, self.reflow_stages + 1):
            teacher_keys = []
            for index, (name, parameter) in enumerate(self._model.named_parameters()):
                key = f'stage{stage}_teacher{index}'
                self._reflow_evidence.register_buffer(key, parameter.detach().cpu().clone())
                teacher_keys.append({'name': name, 'buffer': key})
            pair_seed = self.seed + 1000 * stage
            generator = torch.Generator(device='cpu').manual_seed(pair_seed)
            z0 = torch.randn((count, dimensions), generator=generator)
            z1 = torch.empty_like(z0)
            geometry = []
            self._model.eval()
            for start in range(0, count, self.batch_size):
                stop = min(start + self.batch_size, count)
                terminal, record = transport_endpoints(self._model, z0[start:stop].to(self._device),
                                                       self.reflow_ode_steps, self.reflow_solver)
                z1[start:stop] = terminal.cpu()
                geometry.append((stop - start, record))
            self._reflow_evidence.register_buffer(f'stage{stage}_z0', z0)
            self._reflow_evidence.register_buffer(f'stage{stage}_z1', z1)
            # Warm start preserves preceding flow parameters; a fresh Adam
            # matches the pinned author reflow routine's optimizer reset.
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
            train_generator = torch.Generator(device='cpu').manual_seed(pair_seed + 1)
            losses = []
            self._model.train()
            for _ in range(self.reflow_epochs):
                order = torch.randperm(count, generator=train_generator)
                total = 0.0
                for start in range(0, count, self.batch_size):
                    indexes = order[start:start + self.batch_size]
                    # Index both endpoints with the SAME permutation.
                    x0, x1 = z0[indexes].to(self._device), z1[indexes].to(self._device)
                    t = (torch.rand((len(indexes), 1), generator=train_generator)
                         * (1 - self.reflow_time_epsilon) + self.reflow_time_epsilon).to(self._device)
                    xt = (1 - t) * x0 + t * x1
                    prediction = self._model(torch.cat([xt, t], 1))
                    loss = (prediction - (x1 - x0)).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite reflow training loss')
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                    optimizer.step()
                    total += float(loss.detach()) * len(indexes)
                losses.append(total / count)
            self.training_losses_.extend(losses)
            self._model.eval()
            record = {'stage': stage, 'kind': 'learned_coupling_uniform_time_l2', 'epochs': self.reflow_epochs,
                      'training_pairs': count, 'pair_seed': pair_seed, 'pair_sha256': pair_digest(z0, z1),
                      'teacher_parameters': teacher_keys, 'solver': self.reflow_solver, 'ode_steps': self.reflow_ode_steps,
                      'losses': losses, 'warm_start': True, 'optimizer_reset': True,
                      'teacher_discrete_excess_action_mean': sum(n * r['discrete_excess_action_mean'] for n, r in geometry) / count,
                      'teacher_squared_displacement_per_coordinate': sum(n * r['squared_displacement_per_coordinate'] for n, r in geometry) / count}
            self.stage_records_.append(record)
        metadata = json.dumps(self.stage_records_, sort_keys=True, allow_nan=False).encode('utf-8')
        self._reflow_evidence.register_buffer('metadata_json_utf8', torch.tensor(list(metadata), dtype=torch.uint8))
        return self
