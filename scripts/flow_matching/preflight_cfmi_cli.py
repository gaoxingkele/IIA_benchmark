"""Check author checkpoint/CLI evaluation end to end using real tiny CPU subsets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.run_cfmi import latest_checkpoint, summarize_author_metric


def main():
    base = ROOT / 'experiments/runs/flow_matching_campaign/cfmi_cli_preflight'
    base.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    def execute(command, name):
        with (base / f'{name}.stdout.log').open('w', encoding='utf-8') as out, (base / f'{name}.stderr.log').open('w', encoding='utf-8') as err:
            subprocess.run([sys.executable] + command, cwd=ROOT, env=environment, stdout=out, stderr=err, check=True)
    records = []
    for dataset in ['physio', 'tep']:
        training_root = base / f'{dataset}_train'
        if dataset == 'physio':
            train_config = source / 'configs/physionet/icfm_fold0.yaml'
            data_root = ROOT / 'data/public_datasets/flow_matching/processed_campaign_v1/author_cfmi'
        else:
            train_config = ROOT / 'configs/experiments/fm_transfer_jobs/cfmi_tep_point_missing0.1_seed2026.train.yaml'
            original_pack = ROOT / 'data/public_datasets/flow_matching/processed_campaign_v1/tep_transfer/point_missing0.1_seed2026.npz'
            data_root = base / 'tep_tiny.npz'
            if not data_root.exists():
                with np.load(original_pack, allow_pickle=False) as payload:
                    arrays = {k: payload[k] for k in payload.files}
                    for split in ['train', 'val', 'test']:
                        arrays[split] = arrays[split][:16 if split != 'test' else 2]
                    np.savez_compressed(data_root, **arrays)
        if latest_checkpoint(training_root) is None:
            execute([str(source / 'train.py'), '--config', str(train_config), '--data.dataset.init_args.root', str(data_root),
                     '--trainer.default_root_dir', str(training_root), '--experiment_subdir_base', 'train',
                     '--trainer.accelerator', 'cpu', '--trainer.devices', '1', '--trainer.max_steps', '1',
                     '--trainer.limit_train_batches', '1', '--trainer.limit_val_batches', '1',
                     '--trainer.num_sanity_val_steps', '0', '--trainer.enable_progress_bar', 'false'], f'{dataset}_train')
        checkpoint = latest_checkpoint(training_root)
        if checkpoint is None:
            raise ValueError('Preflight did not save a checkpoint')
        output = base / f'{dataset}_eval'
        if dataset == 'physio':
            execute([str(source / 'eval_imputation_timeseries.py'), '--config', str(source / 'configs/physionet/imputation/umis10/icfm_fold0.yaml'),
                     '--data.dataset.init_args.root', str(data_root), '--data.num_first_datapoints_test', '2',
                     '--cfm_model_path', str(checkpoint), '--default_root_dir', str(output),
                     '--experiment_subdir_base', 'evaluation', '--num_imputations', '2'], f'{dataset}_eval')
        else:
            cfg = json.loads((ROOT / 'configs/experiments/fm_transfer_jobs/cfmi_tep_point_missing0.1_seed2026.json').read_text())
            cfg.update(data_root=data_root.relative_to(ROOT).as_posix(), output_root=output.relative_to(ROOT).as_posix(), num_samples=2, diagnostic=True)
            path = base / 'tep_diagnostic.json'
            path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            execute([str(ROOT / 'scripts/flow_matching/evaluate_cfmi_transfer.py'), '--campaign-config', str(path), '--checkpoint', str(checkpoint)], f'{dataset}_eval')
        metrics = {}
        for name in ['crps', 'normalised_crps', 'rmse', 'mae']:
            paths = list(output.glob(f'evaluation/seed_m*_d*/{name}.npz'))
            if len(paths) != 1:
                raise ValueError('Missing preflight metric')
            with np.load(paths[0], allow_pickle=False) as payload:
                metrics[name] = summarize_author_metric(payload['loss'], name)[1]
        records.append({'dataset': dataset, 'status': 'passed', 'metric_audit': metrics})
        print(json.dumps(records[-1]), flush=True)
    (base / 'report.json').write_text(json.dumps({'status': 'diagnostic_only', 'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
