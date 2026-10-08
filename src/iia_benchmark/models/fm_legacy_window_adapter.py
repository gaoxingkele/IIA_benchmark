"""Expose existing window baselines to the frozen full-TSAD score runner.

This changes neither model architecture, optimizer, losses nor scoring. It
exposes the delegate's modules for checkpoint storage. Existing departures
(including non-adversarial USAD) remain explicitly named in each config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .mtsad_detectors import build_detector, WINDOW_DETECTORS


@dataclass
class LegacyWindowAdapter:
    method: str
    parameters: dict
    epochs: int
    seed: int = 1103
    device: str = 'cpu'
    _delegate: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    def fit(self, windows):
        if self.method not in WINDOW_DETECTORS:
            raise ValueError('Only complete-score window baselines accepted')
        params = {**self.parameters, 'epochs': self.epochs, 'seed': self.seed, 'device': self.device}
        self._delegate = build_detector(self.method, params)
        if self._delegate.kind != 'window':
            raise ValueError('Only complete-score window baselines accepted')
        self._delegate.fit(windows)
        modules = self._delegate._model
        if isinstance(modules, (tuple, list)):
            if not all(isinstance(module, torch.nn.Module) for module in modules):
                raise ValueError('Unrecognized delegate checkpoint contract')
            modules = torch.nn.ModuleList(modules)
        if not isinstance(modules, torch.nn.Module):
            raise ValueError('Delegate exposes no fitted modules')
        self._model = modules
        return self

    def score(self, windows):
        if self._delegate is None:
            raise RuntimeError('fit before score')
        scores = np.asarray(self._delegate.score(windows))
        if scores.shape not in ((len(windows),), windows.shape[:2]) or not np.isfinite(scores).all():
            raise ValueError('Delegate does not cover every configured window')
        return scores
