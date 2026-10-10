"""Construct the full original GT-GAN models and strictly load released weights."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_gtgan_branch(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    matches = [n for n in ast.walk(main) if isinstance(n, ast.If) and
               ast.dump(n.test) == ast.dump(ast.parse("args.model1 == 'gtgan'", mode='eval').body)]
    if len(matches) != 1:
        raise ValueError('Exactly one full native GT-GAN constructor is required')
    return ast.Module(body=matches[0].body, type_ignores=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen native model audit source changed: ' + item['path'])
    lock = json.loads((ROOT / config['environment_lock']).read_text(encoding='utf-8'))
    if sys.version != lock['python']:
        raise ValueError('Frozen Python runtime differs')
    for name, version in lock['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Frozen native dependency differs: ' + name)
    import torch
    import numpy as np
    import tensorflow as tf
    # Match the original Linux CPU TensorFlow evaluator device; do not occupy GPU.
    tf.config.set_visible_devices([], 'GPU')
    torch.set_num_threads(1)
    models = json.loads((ROOT / config['model_config']).read_text(encoding='utf-8'))
    runtime = Path(config['runtime_root'])
    preparation = json.loads((runtime / 'runtime_preparation_receipt.json').read_text(encoding='utf-8'))
    if len(preparation['complete_copied_assets']) != 245 or not preparation['original_source_unchanged']:
        raise ValueError('Complete copied original source is required')
    repaired = {x['path']: x['repaired_sha256'] for x in preparation['runtime_repairs']}
    for asset in preparation['complete_copied_assets']:
        if sha(runtime / asset['path']) != repaired.get(asset['path'], asset['original_sha256']):
            raise ValueError('Pinned isolated runtime changed: ' + asset['path'])
    sys.path.insert(0, str(runtime))
    rows = []
    for case in models['cases']:
        filename = Path(case['source_entry']).name
        path = runtime / filename
        spec = importlib.util.spec_from_file_location('native_gtgan_' + case['id'], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = ROOT / case['source_entry']
        if sha(original) != case['source_entry_sha256']:
            raise ValueError('Original model source changed')
        branch = original_gtgan_branch(original.read_text(encoding='utf-8'))
        torch.manual_seed(case['original_seed'])
        np.random.seed(case['original_seed'])
        namespace = dict(module.__dict__)
        parsed = dict(case['parsed_original_arguments'])
        parsed['effective_shape'] = parsed['input_size']
        namespace.update(args=SimpleNamespace(**parsed), input_size=6, hidden_size=24,
                         num_layers=3, x_hidden=48, device=torch.device('cpu'))
        exec(compile(branch, str(original) + ':original_constructor', 'exec'), namespace)
        counts = {}
        for name in ('embedder', 'generator', 'supervisor', 'recovery', 'discriminator'):
            model = namespace[name]
            counts[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError('Native constructor produced nonfinite parameters')
        released = []
        # Only the two published matching-mask checkpoints have original weights.
        if case['missing_rate'] in (0, 0.7):
            sub = 'stock' if case['missing_rate'] == 0 else 'stock_0.7'
            pre = 'stock_pretrained' if case['missing_rate'] == 0 else 'stock_pretrained_0.7'
            for name, directory in [('embedder', pre), ('recovery', pre), ('generator', sub), ('recovery', sub)]:
                weight = runtime / 'stock_model' / directory / (name + '.pt')
                state = torch.load(str(weight), map_location='cpu')
                namespace[name].load_state_dict(state, strict=True)
                if any(not torch.isfinite(v).all().item() for v in state.values()):
                    raise ValueError('Released weights contain nonfinite values')
                released.append(dict(model=name, path=str(weight), sha256=sha(weight),
                                     state_keys=len(state), strict_load_passed=True))
        regular = case['missing_rate'] == 0
        rows.append(dict(case=case['id'], missing_rate=case['missing_rate'],
                         exact_original_constructor=True, counts=counts, released_weight_checks=released,
                         published_matching_mask_weights_available=bool(released),
                         native_embedding_optimizer_steps=0 if regular else 10000,
                         native_joint_steps=case['joint_training_steps'],
                         native_embedding_boundary='Release embedder skips initial10000 loop' if regular else
                         'Original10000 updates followed by released0.7 embedder/recovery reload; updates are overwritten',
                         all_original_network_classes_unchanged=True,
                         native_training_complete=False, full_model_inference_complete=False))
    output = ROOT / config['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every native constructor proof')
    output.mkdir(parents=True)
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(ROOT / args.config),
                  environment_sha256=sha(ROOT / config['environment_lock']),
                  torch_version=torch.__version__, tensorflow_version=tf.__version__,
                  full_original_constructor_cases=len(rows), cases=rows,
                  author_equivalence_certified=False, performance_result=False,
                  boundary='Full actual native model constructors and strict released weight compatibility only; '
                  'no full optimizer/evaluator execution or benchmark performance is claimed.')
    (output / 'native_model_construction_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(dict(full_native_constructor_cases=len(rows), strict_released_weight_loads=sum(len(x['released_weight_checks']) for x in rows)))


if __name__ == '__main__':
    main()
