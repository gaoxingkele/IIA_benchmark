"""Native full-capacity Pi backward preflight; never report as benchmark performance."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', choices=['source_runtime', 'causal_axis_repaired'])
    args = parser.parse_args()
    settings = json.loads((ROOT / 'configs/experiments/fm_pi_transformer_author_execution.v1.json').read_text(encoding='utf-8'))
    queue = json.loads((ROOT / settings['queue_path']).read_text(encoding='utf-8'))
    if not args.branch:
        records = []
        for branch in settings['branches']:
            proc = subprocess.run([sys.executable, __file__, '--branch', branch], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            if proc.returncode:
                raise RuntimeError(proc.stdout + proc.stderr)
            records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        report = {'queue_sha256': sha(ROOT / settings['queue_path']), 'records': records,
                  'native_data_and_full_capacity_backward_verified': True,
                  'benchmark_performance': False, 'boundary': settings['boundary']}
        write_json(ROOT / settings['preflight_report'], report)
        print(json.dumps(report, indent=2))
        return
    import torch
    import yaml
    torch.set_num_threads(settings['cpu_threads'])
    torch.set_num_interop_threads(1)
    job = next(j for j in queue['jobs'] if j['branch'] == args.branch and j['dataset'] == 'PSM' and j['recipe'] == 'full' and j['seed'] == 42)
    for source in job['frozen_files']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen Pi preflight source differs')
    base = ROOT / settings['output_root'] / 'preflight' / args.branch
    base.mkdir(parents=True, exist_ok=True)
    config = {'general': {'device': 'cpu', 'data_dir': str(ROOT / settings['raw_root']), 'model_dir': str(base)},
              'datasets': {'PSM': dict(job['author_config'], raw_directory=str(ROOT / job['raw_directory']))}}
    config_path = base / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    sys.path.insert(0, str(ROOT / job['runtime']))
    from main import AnomalyDetection
    from utils.utils import kl_loss, set_seed
    set_seed(42)
    detector = AnomalyDetection(str(config_path), 'PSM')
    inputs, _ = next(iter(detector.train_loader))
    outputs, series, prior, _, hurst, tau, smooth, beta, tau_smooth = detector.model(inputs.float())
    series_loss, prior_loss = 0, 0
    for s, p in zip(series, prior):
        normalized = p / p.sum(-1, keepdim=True)
        series_loss += (kl_loss(s, normalized.detach()).mean() + kl_loss(normalized.detach(), s).mean())
        prior_loss += (kl_loss(normalized, s.detach()).mean() + kl_loss(s.detach(), normalized).mean())
    series_loss /= len(prior)
    prior_loss /= len(prior)
    rec = detector.criterion(outputs, inputs)
    distill = detector.compute_distillation_loss(hurst)
    loss1 = rec - detector.k * series_loss + smooth + beta + tau_smooth + distill
    loss2 = rec + detector.k * prior_loss + smooth + beta + tau_smooth + distill
    detector.optimiser.zero_grad()
    torch.nn.utils.clip_grad_norm_(detector.model.parameters(), max_norm=1.0)
    loss1.backward(retain_graph=True)
    loss2.backward()
    if not all(torch.isfinite(p.grad).all() for p in detector.model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite full native Pi backward')
    detector.optimiser.step()
    if not all(torch.isfinite(p).all() for p in detector.model.parameters()):
        raise ValueError('Nonfinite full native Pi step')
    record = {'branch': args.branch, 'dataset': 'PSM', 'batch_shape': list(inputs.shape),
              'configured_d_model': detector.d_model, 'configured_heads': detector.n_heads,
              'configured_layers': detector.e_layers, 'actual_layers': len(prior),
              'native_train_shape': list(detector.train_loader.dataset.train.shape),
              'native_test_shape': list(detector.test_loader.dataset.test.shape),
              'full_stride_one_training_windows': len(detector.train_loader.dataset),
              'parameter_count': sum(p.numel() for p in detector.model.parameters()),
              'loss1': float(loss1.detach()), 'loss2': float(loss2.detach()),
              'torch_version': torch.__version__, 'python': sys.version, 'benchmark_performance': False}
    print(json.dumps(record))


if __name__ == '__main__':
    main()
