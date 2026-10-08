"""Check every real author checkpoint and full native training batch, no benchmark claims."""
import importlib
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from iia_benchmark.models.fm_crossad_author_runtime import build_model, install_wide_cache, load_release
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    import torch
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); parser.add_argument('--row', type=int)
    parser.add_argument('--queue', default='configs/experiments/fm_crossad_author_queue.v1.json')
    parser.add_argument('--report-prefix', default='fm_crossad_author_preflight')
    options = parser.parse_args()
    queue_path = ROOT/options.queue
    queue = json.loads(queue_path.read_text()); settings = queue['settings']
    torch.set_num_threads(settings['cpu_threads'])
    install_wide_cache(ROOT/settings['runtime_root'], ROOT/settings['cache_root'])
    provider = importlib.import_module('data_provider.data_provider')
    records = []
    for job in [j for j in queue['jobs'] if j['mode'] == 'release_checkpoint' and (not options.dataset or j['dataset'] == options.dataset)]:
        started = time.monotonic(); torch.manual_seed(2025)
        model = build_model(ROOT/settings['runtime_root'], job['model_parameters'])
        if options.row:
            from iia_benchmark.models.fm_crossad_ablation import build_ablation
            model = build_ablation(ROOT/settings['runtime_root'], job['model_parameters'], options.row)
        if job.get('activation_checkpointing'):
            if job['activation_checkpointing']=='pure_layer_v2':
                from iia_benchmark.models.fm_crossad_memory_v2 import install_activation_checkpointing
            else:
                from iia_benchmark.models.fm_crossad_memory import install_activation_checkpointing
            model = install_activation_checkpointing(model)
        data, loader = provider.data_provider(str(ROOT/settings['raw_root']), job['dataset'],
                       job['train_parameters']['batch_size'], job['model_parameters']['seq_len'], 1, 'train')
        # Native contiguous batches without fetching all overlap windows or consuming benchmark loader RNG.
        window, batch = job['model_parameters']['seq_len'], job['train_parameters']['batch_size']
        values = torch.from_numpy(np.stack([data.train[i:i+window] for i in range(batch)]).astype(np.float32))
        model.train(); loss, _ = model(values, None, None, None)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        # Fresh initialization is the author training contract; release weights tested separately.
        loaded = {'reconstructed_ablation_row': options.row} if options.row else load_release(model, ROOT/job['release_checkpoint'])
        model.eval()
        with torch.no_grad():
            one, _ = model.infer(values[:1], None, None, None)
            two, _ = model.infer(values[:1], None, None, None)
        assert one.shape == values[:1].shape and torch.isfinite(one).all() and torch.equal(one, two)
        record = {'dataset': job['dataset'], 'checkpoint': loaded, 'native_batch_shape': list(values.shape),
                  'training_windows_stride1': len(data), 'test_native_points': len(data.test),
                  'native_test_tail_points': len(data.test)%window, 'finite_backward': True,
                  'deterministic_release_inference': True, 'parameter_count': sum(p.numel() for p in model.parameters()),
                  'seconds': time.monotonic()-started, 'benchmark_performance': False}
        records.append(record); print(json.dumps({'preflight': job['dataset'], 'shape': record['native_batch_shape']}), flush=True)
        del model, values, data, loader, one, two, loss
    report = {'queue_sha256': sha(queue_path), 'records': records, 'environment': {'torch': torch.__version__, 'numpy': np.__version__},
              'source_required_torch': '1.10.0', 'framework_equivalence': False,
              'benchmark_performance': False, 'boundary': settings['boundary']}
    suffix = ('_'+options.dataset if options.dataset else '')+('_row'+str(options.row) if options.row else '')
    destination = ROOT/'docs/reports'/(options.report_prefix+suffix+'_2026-10-09.json')
    write_json(destination, report)


if __name__ == '__main__':
    main()
