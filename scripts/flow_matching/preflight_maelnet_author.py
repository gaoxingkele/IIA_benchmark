"""Real author interface/data preflight, never a benchmark performance result."""
import argparse
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.register_maelnet_author import script_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-env-only', action='store_true')
    options = parser.parse_args()
    settings = json.loads((ROOT / 'configs/experiments/fm_maelnet_author_execution.v1.json').read_text(encoding='utf-8'))
    source = ROOT / settings['source' if options.original_env_only else 'corrected_source']
    sys.path.insert(0, str(source))
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), 'CPU preflight must not acquire the shared GPU'
    from utils.agentreward import TrainEnvOffline_dist_conf, eval_model
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env.patch_gym import _patch_env
    scores = [np.linspace(-1, 2, 64), np.sin(np.arange(64)), np.cos(np.arange(64))]
    labels = np.zeros(64, dtype=int)
    labels[8:20] = 1
    labels[40:55] = 1
    env = TrainEnvOffline_dist_conf(scores, [0.5] * 3, labels.copy())
    if options.original_env_only:
        try:
            check_env(_patch_env(env), warn=False)
        except AssertionError as error:
            receipt = {'original_env_rejected': True, 'exception': str(error), 'source_sha256': sha(source / 'utils/agentreward.py')}
            write_json(ROOT / 'docs/reports/fm_maelnet_original_env_failure_2026-10-09.json', receipt)
            print(json.dumps(receipt))
            return
        raise AssertionError('Original expected dtype failure not reproduced')
    check_env(_patch_env(env), warn=False)
    random.seed(1103)
    model = DQN('MlpPolicy', env, seed=1103, verbose=0, device='cpu', learning_starts=0, buffer_size=128)
    model.learn(total_timesteps=64)
    result = eval_model(model, env)
    assert len(result[5]) == len(labels) and np.isfinite(result[:4]).all()
    without_labels = TrainEnvOffline_dist_conf(scores, [0.5] * 3, np.zeros_like(labels))
    changed = int(sum(np.count_nonzero(a != b) for a, b in zip(env.list_pred, without_labels.list_pred)))
    assert changed > 0
    sys.argv = ['run_anomaly.py', '--chunk_size', '0', '--num_workers', '0']
    module = runpy.run_path(str(source / 'run_anomaly.py'), run_name='preflight_import')
    author_parser, base_args = module['parser'], module['args']
    from data_provider.data_factory import data_provider
    report = {'purpose': 'full_native_data_and_runtime_interface_preflight_not_benchmark_performance',
              'original_gym_failure': json.loads((ROOT / 'docs/reports/fm_maelnet_original_env_failure_2026-10-09.json').read_text(encoding='utf-8')),
              'source_commit': settings['source_commit'], 'original_source_preserved': True,
              'gym_observation_checker_passed': True, 'diagnostic_DQN_steps': model.num_timesteps,
              'test_label_dependent_PA_prediction_changes': changed, 'native_datasets': {}, 'model_interfaces': [],
              'environment': {'python': sys.version, 'executable': sys.executable,
                              'versions': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'pandas', 'torch', 'gym', 'gymnasium', 'shimmy', 'stable-baselines3', 'sktime', 'scikit-learn', 'einops', 'tensorboard']},
                              'deviations': ['Python 3.12, inherited Torch 2.13.0+cu126 vs listed Torch2.6.0; CPU execution',
                                             'Author DQN call leaves seed=None; preflight DQN seed is explicit and is not a paper result']}}
    psm_batch = None
    for name in settings['datasets']:
        config = json.loads((ROOT / settings['dataset_config_root'] / ('mtsad_' + name.lower() + '.json')).read_text(encoding='utf-8'))
        base_args.data = name
        base_args.root_path = str(ROOT / settings['raw_root'] / config['directory'])
        data, loader = data_provider(base_args, 'train')
        assert list(data.train.shape) == config['expected_shapes']['train']
        assert list(data.test.shape) == config['expected_shapes']['test']
        assert len(data.test_labels) == len(data.test) and np.isin(data.test_labels, [0, 1]).all()
        assert np.isfinite(data.train).all() and np.isfinite(data.test).all()
        counts = {}
        for flag in ['train', 'val', 'test', 'threshold']:
            data.flag = flag
            counts[flag] = len(data)
        data.flag = 'train'
        if name == 'PSM':
            psm_batch = next(iter(loader))[0]
        report['native_datasets'][name] = {'shapes': {'train': list(data.train.shape), 'test': list(data.test.shape)},
                                           'window': data.win_size, 'stride': data.step, 'window_counts': counts,
                                           'label_shape': list(data.test_labels.shape),
                                           'train_validation_overlap_retained': np.array_equal(data.val, data.train[-len(data.val):]),
                                           'flattened_test_points': counts['test'] * data.win_size,
                                           'config': settings['dataset_config_root'] + '/mtsad_' + name.lower() + '.json'}
        del data, loader
        gc.collect()
    for recipe in settings['script_recipes']:
        for stage in settings['stages'][:3]:
            script = source / settings['script_recipe_root'] / recipe / 'PSM_scripts' / stage
            args = author_parser.parse_args(script_arguments(script.read_text(encoding='utf-8')))
            args.use_gpu = False
            args.patch_size = [int(p) for p in args.patch_size]
            network = importlib.import_module('models.' + args.model).Model(args).float()
            # Exact capacity and real PSM input; one batch interface test only.
            output = network(psm_batch[:1])
            loss = sum(t.square().mean() for part in output for t in (part if isinstance(part, list) else [part]))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
            report['model_interfaces'].append({'recipe': recipe, 'model': args.model, 'e_layers': args.e_layers,
                                                'd_layers_argument': args.d_layers, 'd_model': args.d_model,
                                                'parameter_count': sum(p.numel() for p in network.parameters()),
                                                'real_input_shape': list(psm_batch[:1].shape), 'backward_finite': True})
            del network, output, loss
            gc.collect()
    from utils.tools import EarlyStopping_Asso_Discrep
    EarlyStopping_Asso_Discrep()
    report['numpy2_early_stopping_constructed'] = True
    write_json(ROOT / settings['preflight_report'], report)
    print(json.dumps({'datasets': len(report['native_datasets']), 'full_capacity_interfaces': len(report['model_interfaces']),
                      'gym_check_passed': True, 'diagnostic_only': True}))


if __name__ == '__main__':
    main()
