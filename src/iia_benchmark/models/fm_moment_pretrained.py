"""Strictly load actual MOMENT release weights; full-window zero-shot MSE scores."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from .fm_supplemental_author import _resolve_asset


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained(source_root, checkpoint_directory, checkpoint_sha256, config_sha256):
    """No Hub requests, missing keys, random heads or partially loaded tensors."""
    root = _resolve_asset(source_root, None)
    directory = _resolve_asset(checkpoint_directory, None)
    weights_path, config_path = directory / 'model.safetensors', directory / 'config.json'
    if file_sha(weights_path) != checkpoint_sha256 or file_sha(config_path) != config_sha256:
        raise ValueError('Registered MOMENT checkpoint/config checksum mismatch')
    loaded = sys.modules.get('momentfm')
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / 'momentfm':
        raise ValueError('Another MOMENT source version is already imported')
    sys.path.insert(0, str(root))
    from momentfm import MOMENTPipeline
    from safetensors.torch import load_file
    config = json.loads(config_path.read_text(encoding='utf-8'))
    model = MOMENTPipeline(config, model_kwargs={'task_name': 'reconstruction'})
    state = load_file(str(weights_path), device='cpu')
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError({'missing': sorted(set(expected)-set(state)), 'unexpected': sorted(set(state)-set(expected))})
    if any(expected[k].shape != value.shape or not torch.isfinite(value).all() for k, value in state.items()):
        raise ValueError('MOMENT checkpoint tensor shape or finiteness differs')
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError('Partially loaded MOMENT weights')
    model.init()
    model.requires_grad_(False)
    model.eval()
    audit = {'checkpoint_sha256': checkpoint_sha256, 'config_sha256': config_sha256,
             'state_tensors': len(state), 'all_keys_strictly_loaded': True,
             'parameter_count': sum(p.numel() for p in model.parameters()),
             'pretrained_size': config['transformer_backbone'], 'configured_sequence_length': config['seq_len'],
             'task': 'reconstruction', 'random_reinitialized_head': False, 'hub_downloads': False}
    return model, audit


def reconstruct_windows(model, windows, batch_size, device, padding_policy):
    if windows.ndim != 3 or not np.isfinite(windows).all():
        raise ValueError('MOMENT needs finite [windows,time,channels]')
    result = []
    length = windows.shape[1]
    patch = model.patch_len
    if padding_policy == 'minimal_patch':
        padding = (-length) % patch
    elif padding_policy == 'TAB_always_extra_patch':
        padding = patch - length % patch
    else:
        raise ValueError('Unknown frozen MOMENT padding policy')
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            original = torch.from_numpy(windows[start:start + batch_size].astype(np.float32, copy=False)).to(device).transpose(1, 2)
            inputs = torch.nn.functional.pad(original, (0, padding), mode='replicate') if padding else original
            # Explicit all-observed mask prevents any random reconstruction masking.
            observed = torch.ones((len(inputs), inputs.shape[-1]), device=device)
            reconstructed = model(x_enc=inputs, input_mask=observed, mask=observed).reconstruction[..., :length]
            if reconstructed.shape != original.shape or not torch.isfinite(reconstructed).all():
                raise ValueError('MOMENT reconstruction lacks complete source timestamps')
            result.append((original-reconstructed).square().mean(dim=1).cpu().numpy())
    return np.concatenate(result, axis=0) if result else np.empty((0, length), dtype=np.float32)


class MomentPretrainedWindow:
    kind = 'window'

    def __init__(self, source_root, checkpoint_directory, checkpoint_sha256, config_sha256,
                 window=512, batch_size=1, padding_policy='minimal_patch', epochs=0, seed=42, device='cpu'):
        if epochs != 0 or batch_size < 1:
            raise ValueError('Zero-shot MOMENT must not silently fine-tune or shorten training')
        self.source_root = source_root
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_sha256 = config_sha256
        self.window, self.batch_size = window, batch_size
        self.padding_policy, self.seed, self.device = padding_policy, seed, device
        self.training_losses_ = []

    def fit(self, windows):
        if windows.ndim != 3 or windows.shape[1] != self.window or not np.isfinite(windows).all():
            raise ValueError('Incomplete/nonfinite MOMENT fit contract')
        self.model, self.checkpoint_load_audit_ = load_pretrained(self.source_root, self.checkpoint_directory, self.checkpoint_sha256, self.config_sha256)
        self.model.to(self.device)
        self.checkpoint_load_audit_['fine_tuning_epochs'] = 0
        self.checkpoint_load_audit_['fit_points_used_for_parameter_estimation'] = 0
        return self

    def score(self, windows):
        if not hasattr(self, 'model'):
            raise RuntimeError('MOMENT pretrained weights not loaded')
        if windows.shape[1] != self.window:
            raise ValueError('Frozen MOMENT context changed')
        return reconstruct_windows(self.model, windows, self.batch_size, self.device, self.padding_policy)
