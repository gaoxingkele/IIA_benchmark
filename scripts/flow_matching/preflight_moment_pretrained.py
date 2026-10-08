"""Load all real released weights strictly and validate native reconstruction shapes."""
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.model_registry import build_configured_method


def main():
    import numpy as np
    import torch
    queue_path = ROOT / 'configs/experiments/fm_moment_pretrained_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    torch.set_num_threads(queue['evaluation']['cpu_threads'])
    manifest = json.loads((ROOT / queue['jobs'][0]['dataset_manifest_path']).read_text(encoding='utf-8'))
    dataset = next(d for d in manifest['datasets'] if d['id'] == 'psm')
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as data:
        values = data[dataset['entities'][0]['array_prefix'] + '_train'][:512].copy()
    records = []
    for size in ['small', 'base', 'large']:
        for track, context in [('release512', 512), ('TAB100', 100)]:
            config_path = ROOT / f'configs/models/fm_moment_pretrained/fm_moment_pretrained_{size}__{track}.json'
            started = time.monotonic()
            model = build_configured_method(config_path)
            window = values[:context][None]
            model.fit(window)
            first = model.score(window)
            second = model.score(window)
            if first.shape != (1, context) or not np.isfinite(first).all() or not np.array_equal(first, second):
                raise ValueError('Actual MOMENT complete/deterministic reconstruction failed')
            records.append({'size': size, 'context': context, 'load_audit': model.checkpoint_load_audit_,
                            'native_input_shape': list(window.shape), 'score_shape': list(first.shape),
                            'repeat_identical': True, 'seconds': time.monotonic()-started, 'benchmark_performance': False})
            print(json.dumps({'preflight_size': size, 'context': context, 'strict_checkpoint_loaded': True}), flush=True)
            del model
    report = {'queue_sha256': sha(queue_path), 'records': records,
              'environment': {name: version(name) for name in ['torch','numpy','transformers','huggingface-hub','safetensors','einops','scipy']},
              'inherited_dependency_path': '.venv/fm-author-py310/Lib/site-packages; parent packages preserved',
              'benchmark_performance': False, 'boundary': queue['boundary']}
    write_json(ROOT / 'docs/reports/fm_moment_pretrained_preflight_2026-10-09.json', report)


if __name__ == '__main__':
    main()
