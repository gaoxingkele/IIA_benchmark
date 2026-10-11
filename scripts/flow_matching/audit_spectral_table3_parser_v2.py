"""Check original config parsing and module imports for every complete Table3 case."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from scripts.flow_matching.run_spectral_table3_original_v2 import ROOT, read, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    verify(queue)
    lock = read(ROOT / queue['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(name) != version
            for name, version in lock['packages'].items()):
        raise ValueError('Every registered native environment package must match')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(ROOT / queue['source_root']))
    from utils.utils_args import parse_args_uncond, parse_args_uncond_spectral
    records = []
    for job in queue['jobs']:
        config = read(ROOT / job['model_config'])
        old = sys.argv
        sys.argv = ['original-author-parser', '--config', str(ROOT / config['author_config']), '--seed', str(config['seed'])]
        try:
            original = (parse_args_uncond_spectral if config['method'] == 'spectral_flow' else parse_args_uncond)()
        finally:
            sys.argv = old
        for key, value in config['author_hyperparameters'].items():
            if getattr(original, key) != value:
                raise ValueError('The complete original YAML setting changed: ' + key)
        if original.epochs != 1000 or original.batch_size != 32 or original.num_workers != 4:
            raise ValueError('Full original training and loader budgets required')
        if original.seed != 42 or original.test_seed != 0:
            raise ValueError('Both original training and evaluation seeds required')
        records.append(dict(id=job['id'], original_canonical_id=job['original_canonical_id'],
            author_yaml_sha256=sha(ROOT / config['author_config']), all_yaml_settings_identical=True,
            epochs=original.epochs, batch_size=original.batch_size, num_workers=original.num_workers,
            seed=original.seed, test_seed=original.test_seed))
    modules = ['models.spectral_flow.model', 'models.spectral_flow.muon', 'models.imagentime.model',
        'models.imagentime.sampler', 'metrics', 'metrics.metrics_long_range', 'utils.utils_data']
    imported = {}
    for name in modules:
        module = importlib.import_module(name)
        imported[name] = dict(path=str(Path(module.__file__).resolve()), sha256=sha(module.__file__))
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Readonly import audit must not initialize a CUDA context')
    report = dict(queue_sha256=sha(ROOT / args.queue), environment_lock_sha256=sha(ROOT / queue['environment_lock']),
        all_four_original_configurations_parsed=True, full_cases=4, records=records,
        full_original_model_optimizer_sampler_and_evaluator_modules_imported=imported,
        cuda_context_initialized=False, diagnostic_only=True, benchmark_performance=False,
        actual_full_models_constructed=False, full_gpu_capacity_preflight_completed=False,
        original_training_started=False, author_equivalence_certified=False,
        boundary='All original YAML settings and full native module imports verified. No model, optimizer step, generation or evaluator training executed by this import audit. Actual full GPU capacity and all1000 training epochs remain required.')
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError('Preserve every native parser/import observation')
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(full_original_configurations_parsed=len(records), original_modules_imported=len(imported),
        full_gpu_preflight_still_required=True, benchmark_performance=False)))


if __name__ == '__main__':
    main()
