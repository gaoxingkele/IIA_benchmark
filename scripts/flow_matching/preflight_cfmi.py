"""CPU-only author training/sampling checks; never benchmark performance."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected'))
from train_argparser import build_argparser
from eval_imputation_timeseries import impute
from scripts.flow_matching.metrics import ensemble_totals, finalize


def main():
    torch.set_num_threads(4)
    records = []
    for dataset in ['physio', 'tep', 'skab', 'pronto']:
        parser = build_argparser()
        config = 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected/configs/physionet/icfm_fold0.yaml' if dataset == 'physio' else f'configs/experiments/fm_transfer_jobs/cfmi_{dataset}_point_missing0.1_seed2026.train.yaml'
        arguments = ['--config', str(ROOT / config)]
        if dataset == 'physio':
            arguments += ['--data.dataset.init_args.root', str(ROOT / 'data/public_datasets/flow_matching/processed_campaign_v1/author_cfmi')]
        args = parser.instantiate_classes(parser.parse_args(arguments))
        trainer = pl.Trainer(accelerator='cpu', devices=1, fast_dev_run=True, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False,
                             default_root_dir=str(ROOT / 'experiments/runs/flow_matching_campaign/cfmi_cpu_preflight'))
        trainer.fit(model=args.model, datamodule=args.data)
        args.data.setup('test')
        data = args.data.test_data[:2]
        values, observed = np.asarray(data[0]), np.asarray(data[1], dtype=bool)
        if dataset == 'physio':
            conditioning = observed.copy()
            conditioning[:, ::2] = False
        else:
            conditioning = np.asarray(data[2], dtype=bool)
        with patch('torch.cuda.is_available', return_value=False):
            samples = impute(SimpleNamespace(imputer='cfm_euler', num_imputations=2), values, conditioning, cfm=args.model.eval())
        metrics = finalize(ensemble_totals(values, samples, observed & ~conditioning))
        records.append({'dataset': dataset, 'shape': list(samples.shape), 'finite': bool(np.isfinite(samples).all()),
                        'status': 'diagnostic_only', 'metrics': metrics})
        print(json.dumps(records[-1]), flush=True)
    output = ROOT / 'experiments/runs/flow_matching_campaign/cfmi_cpu_preflight/report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'boundary': 'One CPU training batch and two test cases with two samples. Plumbing only.', 'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
