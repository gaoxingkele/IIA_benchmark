"""CFM-TS original-task reconstruction with declared architecture and dopri5.

Primary source: https://openreview.net/forum?id=Hqn4Aj7xrQ, pp.3,4,8.
Paper conflicts and local interpretations are frozen in the experiment config.
This module does not certify equivalence to unavailable author code.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fm_native_trajectory import brownian_bridge_targets, fit_rbf_hyperparameters, gp_targets


class PaperTrajectoryField(nn.Module):
    def __init__(self, dimensions, input_projection=64, hidden=258, hidden_layers=3, negative_slope=.01):
        super().__init__()
        layers = [nn.Linear(dimensions+1, input_projection), nn.LeakyReLU(negative_slope)]
        width = input_projection
        for _ in range(hidden_layers):
            layers += [nn.Linear(width, hidden), nn.LeakyReLU(negative_slope)]
            width = hidden
        layers.append(nn.Linear(width, dimensions))
        self.network = nn.Sequential(*layers)
        self.nfe = 0

    def forward(self, time, state):
        self.nfe += 1
        t = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        time_column = t.expand(state.shape[:-1]).unsqueeze(-1)
        return self.network(torch.cat([state, time_column], dim=-1))


class CFMTSPaperTrajectory:
    def __init__(self, method='bb', target_mode='paper_literal', epochs=400, batch_size=256,
                 learning_rate=.01, samples=1, grid_points=50, seed=1103, device='cpu',
                 input_projection=64, hidden=258, hidden_layers=3, negative_slope=.01,
                 process_noise=.005, observation_noise=.001, gp_max_iterations=400,
                 solver='dopri5', rtol=1e-7, atol=1e-9, max_num_steps=10000):
        if method not in {'bb', 'gp', 'node'} or device != 'cpu' or solver != 'dopri5':
            raise ValueError('Choose registered bb/gp/node, CPU and dopri5')
        if min(epochs, batch_size, samples, grid_points, input_projection, hidden, hidden_layers, gp_max_iterations) < 1:
            raise ValueError('Positive frozen budgets required')
        self.method, self.target_mode, self.epochs, self.batch_size = method, target_mode, epochs, batch_size
        self.learning_rate, self.samples, self.grid_points, self.seed = learning_rate, samples, grid_points, seed
        self.network_parameters = dict(input_projection=input_projection, hidden=hidden,
                                       hidden_layers=hidden_layers, negative_slope=negative_slope)
        self.process_noise, self.observation_noise, self.gp_max_iterations = process_noise, observation_noise, gp_max_iterations
        self.solver, self.rtol, self.atol, self.max_num_steps = solver, rtol, atol, max_num_steps

    def prepare_targets(self, trajectories, progress_callback=None):
        rng = np.random.default_rng(self.seed)
        states, velocities, times = [], [], []
        self.preprocessing_ = []
        for index, (observed_times, y) in enumerate(trajectories):
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y[:, None]
            query = np.linspace(observed_times[0], observed_times[-1], self.grid_points)
            parameters = fit_rbf_hyperparameters(observed_times, y, self.observation_noise, self.gp_max_iterations) if self.method == 'gp' else {}
            self.preprocessing_.append(parameters)
            for _ in range(self.samples):
                noise = rng.normal(size=(len(query), y.shape[1]))
                if self.method == 'bb':
                    x, u = brownian_bridge_targets(observed_times, y, query, noise, self.target_mode,
                                                   self.process_noise, self.observation_noise)
                else:
                    x, u = gp_targets(observed_times, y, query, noise, self.target_mode,
                                      lengthscale=parameters['lengthscale'], variance=parameters['variance'],
                                      noise_variance=self.observation_noise)
                states.append(x); velocities.append(u); times.append(query)
            if progress_callback:
                progress_callback({'phase': 'preprocessing', 'trajectories_completed': index+1,
                                   'required_trajectories': len(trajectories)})
        arrays = [np.concatenate(v) for v in [states, velocities, times]]
        if not all(np.isfinite(v).all() for v in arrays):
            raise FloatingPointError('Nonfinite CFM targets')
        return tuple(torch.tensor(a, dtype=torch.float32) for a in arrays)

    def _solve(self, initial, times, adjoint=False):
        from torchdiffeq import odeint, odeint_adjoint
        integrate = odeint_adjoint if adjoint else odeint
        return integrate(self.model, initial, times, method=self.solver, rtol=self.rtol, atol=self.atol,
                         options={'max_num_steps': self.max_num_steps})

    def fit(self, trajectories, progress_callback=None):
        if not trajectories:
            raise ValueError('No training trajectories')
        dimensions = np.asarray(trajectories[0][1]).reshape(len(trajectories[0][0]), -1).shape[1]
        self.optimizer_updates_ = 0
        self.loss_history_ = []
        self.preprocessing_ = []
        prepared = self.prepare_targets(trajectories, progress_callback) if self.method != 'node' else None
        node_data = [(torch.tensor(t, dtype=torch.float64), torch.tensor(y, dtype=torch.float32).reshape(len(t), -1))
                     for t, y in trajectories] if self.method == 'node' else None
        count = len(node_data) if node_data is not None else len(prepared[0])
        self.training_items_ = count
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = PaperTrajectoryField(dimensions, **self.network_parameters)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            for epoch in range(self.epochs):
                total = 0.
                for ids in torch.randperm(count).split(self.batch_size):
                    optimizer.zero_grad()
                    if node_data is not None:
                        # Separate time grids; stream adjoint backward to retain the exact
                        # averaged minibatch gradient without holding hundreds of graphs.
                        for index in ids.tolist():
                            t, y = node_data[index]
                            prediction = self._solve(y[0], t, adjoint=True)
                            loss = F.mse_loss(prediction, y)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite NODE loss')
                            (loss / len(ids)).backward()
                            total += float(loss.detach())
                    else:
                        x, u, t = prepared
                        loss = F.mse_loss(self.model(t[ids], x[ids]), u[ids])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite CFM loss')
                        loss.backward()
                        total += float(loss.detach()) * len(ids)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise FloatingPointError('Nonfinite gradient')
                    optimizer.step()
                    self.optimizer_updates_ += 1
                self.loss_history_.append(total / count)
                if progress_callback:
                    progress_callback({'phase': 'training', 'epoch': epoch+1, 'required_epochs': self.epochs,
                                       'optimizer_updates': self.optimizer_updates_, 'loss': self.loss_history_[-1]})
        if self.optimizer_updates_ != self.epochs * math.ceil(count / self.batch_size):
            raise AssertionError('Full optimizer budget differs')
        self.model.eval()
        return self

    def simulate(self, initial_state, times):
        t = np.asarray(times, dtype=float)
        if not hasattr(self, 'model') or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Fit model and supply strictly increasing finite physical times')
        initial = torch.tensor(initial_state, dtype=torch.float32)
        with torch.no_grad():
            result = self._solve(initial, torch.tensor(t, dtype=torch.float64)).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite full-trajectory prediction')
        return result
