"""Preserve official CrossAD source; patch CPU masks and cache exact CSV parses."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_runtime(source, target):
    source, target = Path(source), Path(target)
    records = []
    for original in sorted(source.rglob('*.py')):
        relative = original.relative_to(source)
        if '__pycache__' in relative.parts:
            continue
        text = original.read_text(encoding='utf-8')
        patches = []
        if relative.as_posix() == 'models/CrossAD/Basic_CrossAD.py':
            for name in ['scale_ind_mask', 'next_scale_mask']:
                old = f'self.ms_utils.{name}(self.ms_p_lens).cuda()'
                if text.count(old) != 1:
                    raise ValueError('Pinned CrossAD CPU placement anchor changed')
                text = text.replace(old, f'self.ms_utils.{name}(self.ms_p_lens)')
                patches.append('CPU mask placement: ' + name)
        destination = target / relative
        payload = text.encode('utf-8')
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError('CrossAD runtime is frozen; use a new version')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
        records.append({'path': relative.as_posix(), 'source_sha256': digest(original),
                        'runtime_sha256': digest(destination), 'patches': patches})
    manifest = {'files': records, 'algorithm_changed': False,
                'boundary': 'Only two unconditional .cuda() mask placements removed. CPU execution only.'}
    payload = json.dumps(manifest, indent=2) + '\n'
    receipt = target / 'runtime_manifest.json'
    if receipt.exists() and receipt.read_text(encoding='utf-8') != payload:
        raise ValueError('CrossAD runtime manifest changed')
    receipt.write_text(payload, encoding='utf-8', newline='\n')
    return manifest


def import_author(runtime):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    return importlib.import_module('exp.exp_anomaly_detection')


def build_model(runtime, model_parameters):
    import_author(runtime)
    from types import SimpleNamespace
    from models.CrossAD.Basic_CrossAD import Basic_CrossAD
    return Basic_CrossAD(SimpleNamespace(**model_parameters))


def install_wide_cache(runtime, cache_root):
    """Original provider/scaler/splits/windows unchanged; reuse exact wide frames."""
    import_author(runtime)
    import pandas as pd
    provider = importlib.import_module('data_provider.data_provider')
    audits = []
    def cached(path, nrows=None):
        name = Path(path).stem
        receipt_path = Path(cache_root) / (name + '.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if digest(path) != receipt['raw_sha256']:
            raise ValueError('CrossAD raw data changed')
        frame_path = Path(cache_root) / (name + '.pkl')
        if digest(frame_path) != receipt['cache_sha256']:
            raise ValueError('CrossAD parsed cache changed')
        frame = pd.read_pickle(frame_path)
        if list(frame.shape) != receipt['shape'] or list(frame.columns) != receipt['columns']:
            raise ValueError('CrossAD complete source frame changed')
        audits.append(receipt)
        return frame.iloc[:nrows] if nrows is not None and isinstance(nrows, int) and len(frame) >= nrows else frame
    provider.read_data = cached
    return audits


def load_release(model, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    expected = model.state_dict()
    extras = set(state) - set(expected)
    if set(expected) - set(state) or any(not k.endswith(('total_ops', 'total_params')) for k in extras):
        raise ValueError('CrossAD release has missing/unexpected algorithm weights')
    algorithm_state = {k: state[k] for k in expected}
    if any(v.shape != expected[k].shape or not torch.isfinite(v).all() for k, v in algorithm_state.items()):
        raise ValueError('CrossAD release shape/nonfinite state')
    model.load_state_dict(algorithm_state, strict=True)
    return {'checkpoint_sha256': digest(path), 'algorithm_tensors_strictly_loaded': len(expected),
            'ignored_thop_counter_keys': sorted(extras), 'missing_algorithm_weights': [],
            'all_algorithm_weights_strictly_loaded': True}
