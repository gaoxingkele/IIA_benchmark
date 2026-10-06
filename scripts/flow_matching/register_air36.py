"""Freeze the second original-data queue, using author Air-36 protocols."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.register_transfer import write_frozen


def main():
    cfg = json.loads((ROOT / 'configs/datasets/fm_air36_original_v1.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / cfg['output_root'] / 'audit.json').read_text(encoding='utf-8'))
    jobs = []
    for fold in cfg['validation_indices']:
        for method in ['csdi', 'cfmi']:
            identifier = f'{method}_air36_original_fold{fold}'
            settings = {'schema_version': 1, 'id': identifier, 'author_source': f'experiments/runs/flow_matching_campaign/sources/{method}/corrected',
                        'output_root': f'experiments/runs/flow_matching_campaign/{identifier}',
                        'fold': fold, 'num_samples': 100, 'epochs': 200, 'diagnostic': False,
                        'reference_id': f'{method}_air36', 'boundary': cfg['boundary']}
            if method == 'csdi':
                pack = next(item for item in audit['files'] if item['fold'] == fold)
                settings.update(author_config='config/base.yaml', processed_data=pack['path'], processed_sha256=pack['sha256'],
                                author_kind='pm25', target_strategy='mix', metric_scale='physical', mode='train_evaluate',
                                device='cuda:0', seed=1, missing_ratio=0.0, evaluation_batch_size=16)
            else:
                settings.update(author_train_config=f'configs/pm25/icfm_fold{fold}.yaml',
                                author_eval_config=f'configs/pm25/imputation/icfm_fold{fold}.yaml',
                                data_root=cfg['cfmi_data_root'], dual_air36_cutoff=True,
                                training_root=f'experiments/runs/flow_matching_campaign/cfmi_air36_training_fold{fold}')
            path = ROOT / f'configs/experiments/fm_air36_jobs/{identifier}.json'
            checksum = write_frozen(path, settings)
            jobs.append({'config': path.relative_to(ROOT).as_posix(), 'runner': f'scripts/flow_matching/run_{method}.py', 'config_sha256': checksum})
    write_frozen(ROOT / 'configs/experiments/fm_air36_original_queue.v1.json',
                 {'schema_version': 1, 'id': 'fm_air36_original_queue_v1', 'python': '.venv/fm-author-py310/Scripts/python.exe',
                  'requires_complete_queue': 'configs/experiments/fm_original_queue.v1.json', 'jobs': jobs})
    print(json.dumps({'registered_air36_jobs': len(jobs)}))


if __name__ == '__main__':
    main()
