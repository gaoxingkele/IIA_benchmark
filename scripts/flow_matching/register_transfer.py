"""Freeze paired industrial transfer jobs after original-protocol experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def write_frozen(path, payload):
    text = json.dumps(payload, indent=2) + '\n' if path.suffix == '.json' else yaml.safe_dump(payload, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_text(encoding='utf-8')
        actual = json.loads(previous) if path.suffix == '.json' else yaml.safe_load(previous)
        if actual != payload:
            raise ValueError(f'Registered file changed; existing file preserved: {path}')
    if not path.exists():
        path.write_text(text, encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/cfmi/corrected'
    train_base = yaml.safe_load((source / 'configs/physionet/icfm_fold0.yaml').read_text())
    eval_base = yaml.safe_load((source / 'configs/physionet/imputation/umis10/icfm_fold0.yaml').read_text())
    jobs = []
    for dataset in ['tep', 'skab', 'pronto']:
        config = json.loads((ROOT / f'configs/datasets/fm_{dataset}_transfer_v1.json').read_text(encoding='utf-8'))
        audit = json.loads((ROOT / config['output_root'] / 'audit.json').read_text(encoding='utf-8'))
        features = audit['shape'][-1]
        for seed in config['mask_seeds']:
            shared_train_path = None
            for pattern in config['missing_patterns']:
                for ratio in config['missing_ratios']:
                    pack = f"{config['output_root']}/{pattern}_missing{ratio:g}_seed{seed}.npz"
                    for method in ['csdi', 'cfmi']:
                        identifier = f'{method}_{dataset}_{pattern}_missing{ratio:g}_seed{seed}'
                        out = f'experiments/runs/flow_matching_campaign/{identifier}'
                        base = {'schema_version': 1, 'id': identifier, 'dataset': dataset,
                                'author_source': f'experiments/runs/flow_matching_campaign/sources/{method}/corrected',
                                'output_root': out, 'seed': seed, 'mask_seed': seed,
                                'missing_pattern': pattern, 'missing_ratio': ratio, 'num_samples': 100,
                                'epochs': 200, 'diagnostic': False, 'metric_protocol': 'transfer_ensemble_v1',
                                'dataset_config': f'configs/datasets/fm_{dataset}_transfer_v1.json',
                                'processed_sha256': hashlib.sha256((ROOT / pack).read_bytes()).hexdigest(),
                                'checkpoint_policy': 'final_epoch',
                                'boundary': 'New grouped-split benchmark. Fixed original hyperparameters and uniform random training masks; point/block/channel test masks share targets across methods. No paper reference score on this dataset.'}
                        first = pattern == config['missing_patterns'][0] and ratio == config['missing_ratios'][0]
                        if method == 'csdi':
                            if first:
                                shared_train_path = out + '/model.pth'
                            base.update(processed_data=pack, author_config='config/base.yaml',
                                        mode='train_evaluate' if first else 'evaluate', device='cuda:0', evaluation_batch_size=16)
                            if not first:
                                base['checkpoint'] = shared_train_path
                        else:
                            train_yaml = copy.deepcopy(train_base)
                            train_yaml['seed_everything'] = seed
                            train_yaml['data']['setup_seed'] = seed
                            train_yaml['data']['num_workers'] = 0
                            train_yaml['data']['dataset'] = {'class_path': 'scripts.flow_matching.transfer_dataset.CampaignWindowDataset', 'init_args': {'root': str(ROOT / pack)}}
                            train_yaml['model']['init_args']['vector_field_net']['init_args']['target_dim'] = features
                            train_path = ROOT / f'configs/experiments/fm_transfer_jobs/{identifier}.train.yaml'
                            write_frozen(train_path, train_yaml)
                            eval_yaml = copy.deepcopy(eval_base)
                            eval_yaml['seed_everything'] = seed
                            eval_yaml['data']['setup_seed'] = seed
                            eval_yaml['data']['num_workers'] = 0
                            eval_yaml['data']['dataset'] = train_yaml['data']['dataset']
                            eval_yaml['data']['missingness'] = {'class_path': 'imp_cfm.data.missing_data_module.DummyMissingnessDataset', 'init_args': {'miss_mask_idx': 2}}
                            eval_path = ROOT / f'configs/experiments/fm_transfer_jobs/{identifier}.eval.yaml'
                            write_frozen(eval_path, eval_yaml)
                            base.update(data_root=pack, train_config=train_path.relative_to(ROOT).as_posix(),
                                        eval_config=eval_path.relative_to(ROOT).as_posix(),
                                        training_root=f'experiments/runs/flow_matching_campaign/cfmi_training_{dataset}_seed{seed}')
                            base['train_config_sha256'] = hashlib.sha256(train_path.read_bytes()).hexdigest()
                            base['eval_config_sha256'] = hashlib.sha256(eval_path.read_bytes()).hexdigest()
                        path = ROOT / f'configs/experiments/fm_transfer_jobs/{identifier}.json'
                        checksum = write_frozen(path, base)
                        jobs.append({'config': path.relative_to(ROOT).as_posix(), 'runner': f'scripts/flow_matching/run_{method}.py', 'config_sha256': checksum})
    queue = {'schema_version': 1, 'id': 'fm_industrial_transfer_queue_v1', 'python': '.venv/fm-author-py310/Scripts/python.exe',
             'requires_complete_queue': 'configs/experiments/fm_original_queue.v1.json',
             'requires_original_scope_review': 'configs/reproducibility/flow_matching_campaign.v1.json',
             'boundary': 'First-wave CFMI/CSDI transfer only; remaining papers and Air Quality original experiments remain separately tracked.', 'jobs': jobs}
    write_frozen(ROOT / 'configs/experiments/fm_transfer_queue.v1.json', queue)
    print(json.dumps({'registered_transfer_jobs': len(jobs), 'full_training_jobs': 30, 'datasets': 3}))


if __name__ == '__main__':
    main()
