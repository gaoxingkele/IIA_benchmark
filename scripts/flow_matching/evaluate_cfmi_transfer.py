"""Use author CFMI sampling, then compute the frozen common transfer metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.flow_matching.metrics import ensemble_totals, finalize, PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--campaign-config', type=Path, required=True)
    command.add_argument('--checkpoint', type=Path, required=True)
    args = command.parse_args()
    settings = json.loads(args.campaign_config.read_text(encoding='utf-8'))
    (ROOT / settings['output_root']).mkdir(parents=True, exist_ok=True)
    if settings['metric_protocol'] != PROTOCOL:
        raise ValueError('Unknown transfer metric protocol')
    sys.path.insert(0, str(ROOT / settings['author_source']))
    import eval_imputation_timeseries as author
    from eval_imputation_timeseries_argparser import build_argparser
    import torch
    author.parser = build_argparser()
    hparams = author.parser.parse_args([
        '--config', str(ROOT / settings['eval_config']),
        '--data.dataset.init_args.root', str(ROOT / settings['data_root']),
        '--cfm_model_path', str(args.checkpoint.resolve()),
        '--default_root_dir', str(ROOT / settings['output_root']),
        '--experiment_subdir_base', 'evaluation',
        '--num_imputations', str(settings['num_samples']),
    ])
    sampler = author.impute
    def impute(hparams, target, conditioning, **models):
        samples = sampler(hparams, target, conditioning, **models)
        with np.load(ROOT / settings['data_root'], allow_pickle=False) as data:
            indices = data['test']
            expected = data['values'][indices]
            mask = data['eval_mask'][indices]
            if not np.array_equal(expected, target) or not np.array_equal(data['conditioning'][indices], conditioning):
                raise ValueError('Author evaluation reordered or changed registered transfer data')
        totals = {}
        # Avoid allocating/sorting the entire industrial ensemble in float64 at once.
        for start in range(0, len(target), 16):
            addition = ensemble_totals(target[start:start + 16], samples[start:start + 16], mask[start:start + 16])
            for key, value in addition.items():
                totals[key] = totals.get(key, 0) + value
        output = ROOT / settings['output_root'] / 'shared_metrics.json'
        output.write_text(json.dumps({'metric_protocol': PROTOCOL, 'metrics': finalize(totals),
                          'eval_count': totals['eval_count'], 'test_cases': len(target)}, indent=2), encoding='utf-8')
        return samples
    author.impute = impute
    torch.set_num_threads(settings.get('cpu_threads', 4))
    author.run(hparams)


if __name__ == '__main__':
    main()
