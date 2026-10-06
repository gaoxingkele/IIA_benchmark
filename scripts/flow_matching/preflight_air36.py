"""CPU loss/sampling checks for both author Air-36 interfaces, never scores."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.run_csdi import AuditedDataset
from scripts.flow_matching.metrics import ensemble_totals, finalize


def main():
    torch.set_num_threads(4)
    records = []
    cfg = json.loads((ROOT / 'configs/experiments/fm_air36_jobs/csdi_air36_original_fold0.json').read_text())
    source = ROOT / cfg['author_source']
    sys.path.insert(0, str(source))
    from main_model import CSDI_PM25
    config = yaml.safe_load((source / cfg['author_config']).read_text())
    config['model']['is_unconditional'] = False
    config['model']['target_strategy'] = 'mix'
    with np.load(ROOT / cfg['processed_data'], allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    model = CSDI_PM25(config, 'cpu')
    batch = next(iter(DataLoader(AuditedDataset(data, 'train'), batch_size=2)))
    loss = model(batch)
    if not torch.isfinite(loss):
        raise ValueError('Invalid Air-36 loss')
    loss.backward()
    batch = next(iter(DataLoader(AuditedDataset(data, 'test'), batch_size=2)))
    model.eval()
    with torch.inference_mode():
        samples, target, mask, _, _ = model.evaluate(batch, 2)
    metrics = finalize(ensemble_totals(target.permute(0, 2, 1).numpy(), samples.permute(0, 1, 3, 2).numpy(), mask.permute(0, 2, 1).numpy()))
    records.append({'method': 'csdi', 'status': 'diagnostic_only', 'finite': True})
    sys.path.insert(0, str(ROOT / 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected'))
    from train_argparser import build_argparser
    from eval_imputation_timeseries import impute
    parser = build_argparser()
    hparams = parser.instantiate_classes(parser.parse_args([
        '--config', str(ROOT / 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected/configs/pm25/icfm_fold0.yaml'),
        '--data.dataset.init_args.root', str(ROOT / 'data/public_datasets/flow_matching/processed_campaign_v1/author_cfmi'),
    ]))
    import pytorch_lightning as pl
    trainer = pl.Trainer(accelerator='cpu', devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(hparams.model, datamodule=hparams.data)
    hparams.data.setup('test')
    test = hparams.data.test_data[:2]
    with patch('torch.cuda.is_available', return_value=False):
        samples = impute(SimpleNamespace(imputer='cfm_euler', num_imputations=2), test[0], test[2], cfm=hparams.model.eval())
    if not np.isfinite(samples).all():
        raise ValueError('Invalid Air-36 samples')
    records.append({'method': 'cfmi', 'status': 'diagnostic_only', 'finite': True})
    output = ROOT / 'experiments/runs/flow_matching_campaign/air36_cpu_preflight/report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'boundary': 'CPU plumbing check only', 'records': records}, indent=2), encoding='utf-8')
    print(json.dumps({'passed_methods': len(records)}))


if __name__ == '__main__':
    main()
