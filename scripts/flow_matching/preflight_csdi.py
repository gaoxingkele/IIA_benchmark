"""Real-data CSDI transfer shape/loss/sampling checks on CPU, diagnostic only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.run_csdi import AuditedDataset
from scripts.flow_matching.metrics import ensemble_totals, finalize
source = ROOT / 'experiments/runs/flow_matching_campaign/sources/csdi/corrected'
sys.path.insert(0, str(source))
from main_model import CSDI_Physio


def main():
    torch.set_num_threads(4)
    records = []
    for dataset in ['tep', 'skab', 'pronto']:
        settings = json.loads((ROOT / f'configs/experiments/fm_transfer_jobs/csdi_{dataset}_point_missing0.1_seed2026.json').read_text())
        with np.load(ROOT / settings['processed_data'], allow_pickle=False) as payload:
            data = {key: payload[key] for key in payload.files}
        config = yaml.safe_load((source / settings['author_config']).read_text())
        config['model']['is_unconditional'] = False
        config['model']['test_missing_ratio'] = .1
        model = CSDI_Physio(config, 'cpu', target_dim=data['values'].shape[-1])
        batch = next(iter(DataLoader(AuditedDataset(data, 'train'), batch_size=2)))
        loss = model(batch)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite diagnostic training loss')
        loss.backward()
        model.eval()
        batch = next(iter(DataLoader(AuditedDataset(data, 'test'), batch_size=2)))
        with torch.inference_mode():
            samples, target, mask, _, _ = model.evaluate(batch, 2)
        metrics = finalize(ensemble_totals(target.permute(0, 2, 1).numpy(), samples.permute(0, 1, 3, 2).numpy(), mask.permute(0, 2, 1).numpy()))
        records.append({'dataset': dataset, 'status': 'diagnostic_only', 'finite': True, 'loss': loss.item(), 'metrics': metrics})
        print(json.dumps(records[-1]), flush=True)
    output = ROOT / 'experiments/runs/flow_matching_campaign/csdi_cpu_preflight/report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'boundary': 'One CPU loss/backprop batch and two real test cases with two samples; plumbing only.', 'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
