"""Run pinned SAITS model/training functions with audited epoch-boundary resume."""
from __future__ import annotations

import argparse
from configparser import ConfigParser, ExtendedInterpolation
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import working_copy
from scripts.flow_matching.prepare_saits_physio import sha


def patient_bootstrap(statistics, seed, repetitions=2000):
    """Resample entire patients, weighting metrics by held-out coordinates."""
    import numpy as np
    records = np.asarray(statistics, dtype=float)
    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        total = records[generator.integers(0, len(records), size=len(records))].sum(axis=0)
        if total[2] > 0 and total[3] > 0:
            draws.append([total[0] / total[2], np.sqrt(total[1] / total[2]), total[0] / total[3]])
    if not draws:
        raise ValueError('No evaluable bootstrap samples')
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return {key: intervals[:, i].tolist() for i, key in enumerate(('mae', 'rmse', 'mre'))}


def configure(author, config, output):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(ROOT / config['author_config'], encoding='utf-8')
    args = author.read_arguments(argparse.Namespace(param_searching_mode=False, test_mode=False), cfg)
    args.dataset_path = str(ROOT / config['dataset_root'])
    args.device = config['device']
    args.num_workers = 0
    args.model_saving = str(output / 'models')
    if config.get('diagnostic'):
        args.epochs = config['diagnostic_epochs']
        args.batch_size = 2
        args.eval_every_n_steps = 1
    args.final_epoch = False
    model_args = {'device': args.device, 'MIT': args.MIT}
    if args.model_type in ('SAITS', 'Transformer'):
        for key in ('input_with_mask', 'diagonal_attention_mask'):
            setattr(args, key, cfg.getboolean('model', key))
        for key in ('n_groups', 'n_group_inner_layers', 'd_model', 'd_inner', 'n_head', 'd_k', 'd_v'):
            setattr(args, key, cfg.getint('model', key))
        args.dropout = cfg.getfloat('model', 'dropout')
        args.param_sharing_strategy = cfg.get('model', 'param_sharing_strategy')
        model_args.update({key: getattr(args, key) for key in ('n_groups', 'n_group_inner_layers', 'd_model', 'd_inner',
            'n_head', 'd_k', 'd_v', 'dropout', 'input_with_mask', 'diagonal_attention_mask', 'param_sharing_strategy')})
        model_args.update(d_time=args.seq_len, d_feature=args.feature_num)
    elif args.model_type == 'BRITS':
        args.consistency_loss_weight = cfg.getfloat('training', 'consistency_loss_weight')
        args.rnn_hidden_size = cfg.getint('model', 'rnn_hidden_size')
        model_args.update(seq_len=args.seq_len, feature_num=args.feature_num, rnn_hidden_size=args.rnn_hidden_size)
    else:
        raise ValueError('Unreviewed author model type')
    author.args = args
    return args, model_args


def run(config_path, stop_after_epoch=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if sha(ROOT / config['author_config']) != config['author_config_sha256']:
        raise ValueError('Author hyperparameters changed after registration')
    config_sha = sha(config_path)
    output = ROOT / config['output_root']
    output.mkdir(parents=True, exist_ok=True)
    report_name = 'diagnostic_report.json' if config.get('diagnostic') else 'result.json'
    if (output / report_name).exists():
        report = json.loads((output / report_name).read_text(encoding='utf-8'))
        if report['config_sha256'] != config_sha:
            raise ValueError('Output configuration mismatch')
        return report
    if stop_after_epoch and not config.get('diagnostic'):
        raise ValueError('Interruption testing is permitted only for diagnostic jobs')
    working_copy('saits')
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/corrected'
    sys.path.insert(0, str(source))
    import run_models as author
    torch.set_num_threads(4)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    args, model_args = configure(author, config, output)
    model = author.MODEL_DICT[args.model_type](**model_args).to(args.device)
    author.model = model
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger = author.setup_logger(str(output / 'training.log'), config['id'])
    author.logger = logger
    loaders = author.UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num, args.model_type,
                                      args.batch_size, args.num_workers, args.MIT)
    train, validation = loaders.get_train_val_dataloader()
    test = loaders.get_test_dataloader()
    if config.get('diagnostic'):
        train = DataLoader(Subset(train.dataset, range(4)), batch_size=2, shuffle=True)
        validation = DataLoader(Subset(validation.dataset, range(2)), batch_size=2, shuffle=True)
        test = DataLoader(Subset(test.dataset, range(2)), batch_size=2, shuffle=True)
    optimizer = author.OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    author.optimizer = optimizer
    controller = author.Controller(args.early_stop_patience)
    dataset_sha = sha(ROOT / config['dataset_root'] / 'datasets.h5')
    data_audit = json.loads((ROOT / config['dataset_root'] / 'data_audit.json').read_text(encoding='utf-8'))
    if dataset_sha != data_audit['dataset_sha256']:
        raise ValueError('Dataset differs from frozen audit')
    if sha(ROOT / config['dataset_root'] / 'audit_arrays.npz') != data_audit['audit_arrays_sha256']:
        raise ValueError('Patient identities differ from frozen audit')
    best_path = output / 'best.pt'

    def save_validation_best(model, optimizer, state, args, saving_path):
        # Author validate invokes this only on a strict validation-MAE improvement.
        temporary = best_path.with_suffix('.tmp')
        torch.save({'model_state_dict': model.state_dict(), 'validation_mae': float(state['best_imputation_MAE']),
                    'training_step': state['train_step'], 'epoch': state['epoch']}, temporary)
        os.replace(temporary, best_path)

    author.save_model = save_validation_best
    resume_path = output / 'resume.pt'
    start_epoch, finished, training_seconds = 0, False, 0.0
    if resume_path.exists():
        state = torch.load(resume_path, map_location=args.device, weights_only=False)
        if state['config_sha256'] != config_sha or state['dataset_sha256'] != dataset_sha:
            raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        controller.state_dict = state['controller']
        controller.early_stop_patience = state['patience']
        start_epoch, finished, training_seconds = state['next_epoch'], state['finished'], state['training_seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    with SummaryWriter(str(output / 'tensorboard')) as writer:
        for epoch in range(start_epoch, args.epochs):
            if finished:
                break
            start = time.monotonic()
            args.final_epoch = epoch == args.epochs - 1
            early_stop = False
            for data in train:
                model.train()
                early_stop = author.model_processing(data, model, 'train', optimizer, validation, writer, controller, logger)
                if early_stop:
                    break
            if not early_stop:
                controller.epoch_num_plus_1()
            finished = early_stop or args.final_epoch
            training_seconds += time.monotonic() - start
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'controller': controller.state_dict,
                     'patience': controller.early_stop_patience, 'next_epoch': epoch + 1, 'finished': finished,
                     'training_seconds': training_seconds, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     'config_sha256': config_sha, 'dataset_sha256': dataset_sha}
            temporary = resume_path.with_suffix('.tmp')
            torch.save(state, temporary)
            os.replace(temporary, resume_path)
            logger.info(f'epoch={epoch + 1} validation_best={controller.state_dict["best_imputation_MAE"]} finished={finished}')
            if stop_after_epoch and epoch + 1 >= stop_after_epoch and not finished:
                return {'status': 'diagnostic_interrupted', 'next_epoch': epoch + 1}
    if not best_path.exists():
        raise ValueError('No validation-selected checkpoint; cannot evaluate test')
    best = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    model.eval()
    predictions, targets, masks, indices = [], [], [], []
    evaluation_start = time.monotonic()
    with torch.no_grad():
        for data in test:
            inputs, result = author.model_processing(data, model, 'test')
            predictions.append(result['imputed_data'].cpu())
            targets.append(inputs['X_holdout'].cpu())
            masks.append(inputs['indicating_mask'].cpu())
            indices.append(inputs['indices'].cpu())
    prediction, target, mask = map(torch.cat, (predictions, targets, masks))
    if not torch.isfinite(prediction).all():
        raise ValueError('Non-finite model predictions')
    metrics = {name: float(function(prediction, target, mask)) for name, function in
               (('mae', author.masked_mae_cal), ('rmse', author.masked_rmse_cal), ('mre', author.masked_mre_cal))}
    difference = (prediction.double() - target.double()) * mask
    stats = torch.stack((difference.abs().sum((1, 2)), difference.square().sum((1, 2)),
                         mask.sum((1, 2)).double(), (target.double() * mask).abs().sum((1, 2))), dim=1).numpy()
    with np.load(ROOT / config['dataset_root'] / 'audit_arrays.npz') as arrays:
        patient_ids = arrays['test_ids'][torch.cat(indices).numpy()]
    np.savez_compressed(output / 'patient_statistics.npz', patient_ids=patient_ids, statistics=stats)
    snapshot = json.loads((source.parent / 'snapshot.json').read_text(encoding='utf-8'))
    report = {'status': 'completed', 'config': config, 'config_sha256': config_sha, 'metrics': metrics,
              'test_patients': len(prediction), 'target_count': int(mask.sum()), 'training_seconds': training_seconds,
              'evaluation_seconds': time.monotonic() - evaluation_start, 'validation_best': {k: v for k, v in best.items() if k != 'model_state_dict'},
              'author_snapshot': snapshot, 'dataset_sha256': dataset_sha, 'parameters': args.total_params,
              'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
              'patient_bootstrap_95_ci': patient_bootstrap(stats, config['seed'] + 10000),
              'protocol_boundaries': data_audit['protocol_boundaries'] + [
                  'Windows loader workers set to zero versus author four; MIT random draw stream can differ.',
                  'Author model, losses, Adam, validation frequency and MAE controller retained; resume saves full state at epoch boundaries.',
                  'Final checkpoint selected solely on validation MAE; test metrics use author functions aggregated on CPU.',
                  'Five preregistered model seeds share the frozen patient split; paper Table 2 does not report uncertainty.']}
    (output / report_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--diagnostic_stop_after_epoch', type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, args.diagnostic_stop_after_epoch)))
    except Exception as error:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        output = ROOT / config['output_root']
        output.mkdir(parents=True, exist_ok=True)
        (output / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}, indent=2), encoding='utf-8')
        raise
