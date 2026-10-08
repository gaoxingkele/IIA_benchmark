"""Auditable, isolated historical Pi runtime and equivalent streamed DFA input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Pinned Pi source differs: ' + before[:100])
    return text.replace(before, after, 1)


def dfa_sample(loader, device, cache_path):
    """Same loader/RNG order and randperm as torch.cat author sampling; disk backing."""
    import numpy as np
    import torch
    count = len(loader.dataset)
    first_shape = loader.dataset[0][0].shape
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Preserved DFA cache already exists: ' + str(path))
    required = count * first_shape[0] * first_shape[1] * 4
    if shutil.disk_usage(path.parent).free < required + 1024**3:
        raise OSError('Insufficient disk for exact full author DFA loader capture')
    mapped = np.memmap(path, mode='w+', dtype=np.float32, shape=(count, *first_shape))
    position = 0
    before = torch.get_rng_state().clone()
    for inputs, _ in loader:
        batch = inputs.cpu().numpy().astype(np.float32, copy=False)
        mapped[position:position + len(batch)] = batch
        position += len(batch)
    if position != count:
        raise ValueError('Incomplete author DFA windows')
    # The author draws this permutation AFTER consuming the shuffled loader.
    indices = torch.randperm(count)[:max(100, count // 10)]
    selected = np.array(mapped[indices.numpy()], copy=True)
    audit = {'all_training_windows': count, 'sampled_windows': len(indices),
             'window_shape': list(first_shape), 'backing_bytes': required,
             'selection_index_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
             'selected_values_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
             'rng_before_sha256': hashlib.sha256(before.numpy().tobytes()).hexdigest(),
             'rng_after_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
             'policy': 'Original shuffled-loader order then original randperm; no window sampling reduction'}
    del mapped
    path.unlink()
    path.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return torch.from_numpy(selected).to(device)


def prepare(source, destination, branch):
    source, destination = Path(source), Path(destination)
    if branch not in ['source_runtime', 'causal_axis_repaired']:
        raise ValueError(branch)
    files = [p for p in source.rglob('*') if p.is_file() and p.suffix in ['.py', '.yaml'] and '__pycache__' not in p.parts]
    generated = {p.relative_to(source).as_posix(): p.read_text(encoding='utf-8') for p in files}
    generated['model/attn.py'] = replace_once(generated['model/attn.py'], 'device="cuda",', 'device="cpu",')
    repairs = ['CPU-safe constructor device; forward keeps tensor device', 'Configured registered raw directory and PSM filename aliases', 'Exact shuffled DFA sample captured on disk without all-window RAM copies', 'Artifact-only instrumentation; mirror objectives and optimizer steps unchanged']
    if branch == 'causal_axis_repaired':
        text = generated['model/attn.py']
        text = replace_once(text, 'scores.masked_fill(attn_mask.mask, -np.inf)', 'scores = scores.masked_fill(attn_mask.mask, -np.inf)')
        text = replace_once(text, 'x_flat = q.permute(0, 2, 1, 3).reshape(B, L, -1).detach()', 'x_flat = q.reshape(B, L, -1).detach()')
        text = replace_once(text, 'prior = prior * gate', 'prior = prior * gate\n        if self.mask_flag:\n            prior = prior.masked_fill(attn_mask.mask, 0.0)')
        generated['model/attn.py'] = text
        repairs += ['Assign series causal mask', 'Preserve query time axis', 'Apply identical causal support to prior before final normalization']
    text = generated['main.py']
    text = replace_once(text, 'import argparse', 'import argparse\nimport json\nfrom iia_benchmark.models.fm_pi_transformer_runtime import dfa_sample')
    text = replace_once(text, 'self.data_path = os.path.join(self.data_dir, self.dataset)', 'self.data_path = self.config["datasets"][dataset]["raw_directory"]')
    text = replace_once(text, 'self.global_hurst = None', 'self.global_hurst = None\n        self.epoch_records = []')
    before = '''            all_data = []
            for input_data, _ in self.train_loader:
                all_data.append(input_data)
            all_data = torch.cat(all_data, dim=0).float().to(self.device)
            subsample_size = max(100, len(all_data) // 10)
            indices = torch.randperm(len(all_data))[:subsample_size]
            all_data = all_data[indices]'''
    text = replace_once(text, before, '            all_data = dfa_sample(self.train_loader, self.device, self.config["general"]["dfa_cache"])')
    text = replace_once(text, 'early_stopping(vali_loss1, vali_loss2, self.model, path)', '''early_stopping(vali_loss1, vali_loss2, self.model, path)
            self.epoch_records.append(dict(epoch=epoch+1, steps=train_steps, train_loss=float(train_loss), validation_loss1=float(vali_loss1), validation_loss2=float(vali_loss2), early_stop=bool(early_stopping.early_stop), seconds=time.time()-epoch_time))
            with open(os.path.join(path, "epochs.json"), "w") as receipt:
                json.dump(self.epoch_records, receipt, allow_nan=False)''')
    text = replace_once(text, 'attens_energy = []\n        attens_phase = []', 'attens_energy = []\n        attens_phase = []\n        reconstruction_train = []')
    text = replace_once(text, 'attens_phase.append(phase_stream.detach().cpu().numpy())', 'attens_phase.append(phase_stream.detach().cpu().numpy())\n            reconstruction_train.append(loss.detach().cpu().numpy())')
    text = replace_once(text, 'test_labels = []\n        attens_energy = []', 'test_labels = []\n        attens_energy = []\n        reconstruction_test = []\n        mismatch_test = []')
    text = replace_once(text, 'test_labels.append(labels)', 'test_labels.append(labels)\n            reconstruction_test.append(loss.detach().cpu().numpy())\n            mismatch_test.append(phase_stream.detach().cpu().numpy())')
    text = replace_once(text, 'gt = test_labels.astype(int)', 'gt = test_labels.astype(int)\n        raw_pred = pred.copy()')
    text = replace_once(text, 'accuracy = accuracy_score(gt, pred)', '''np.savez_compressed(os.path.join(self.model_save_path, "author_scores.npz"),
            train_energy=train_energy, train_phase=train_phase, fused_train=fused_train,
            test_energy=test_energy, labels=gt, raw_predictions=raw_pred, pa_predictions=pred,
            reconstruction_train=np.concatenate(reconstruction_train).reshape(-1),
            reconstruction_test=np.concatenate(reconstruction_test).reshape(-1),
            mismatch_test=np.concatenate(mismatch_test).reshape(-1),
            threshold=thresh, med_E=med_E, iqr_E=iqr_E, med_P=med_P, iqr_P=iqr_P)
        accuracy = accuracy_score(gt, pred)''')
    generated['main.py'] = text
    text = generated['data_loader.py']
    for before, after in [('PSM_train.csv', 'train.csv'), ('PSM_test.csv', 'test.csv'), ('PSM_test_label.csv', 'test_label.csv')]:
        text = replace_once(text, before, after)
    generated['data_loader.py'] = text
    manifest = {'branch': branch, 'source_files': [{'path': p.relative_to(source).as_posix(), 'sha256': sha(p)} for p in files], 'repairs': repairs}
    manifest['files'] = [{'path': name, 'sha256': hashlib.sha256(text.encode()).hexdigest()} for name, text in sorted(generated.items())]
    for name, text in generated.items():
        path = destination / name
        if path.exists() and path.read_bytes() != text.encode():
            raise ValueError('Preserved Pi runtime differs: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(text.encode())
    receipt = destination / 'runtime_patch_manifest.json'
    content = json.dumps(manifest, indent=2) + '\n'
    if receipt.exists() and receipt.read_text(encoding='utf-8') != content:
        raise ValueError('Preserved Pi runtime manifest differs')
    if not receipt.exists():
        receipt.write_text(content, encoding='utf-8')
    return manifest
