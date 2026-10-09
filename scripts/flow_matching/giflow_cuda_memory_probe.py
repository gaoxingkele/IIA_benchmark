"""Measure one real author CUDA update at the registered batch size, never a benchmark."""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

from scripts.flow_matching import run_giflow_native_job as original
from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json


def bound_main():
    source = Path(original.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changes = {'budget_device': 0, 'cpu_override': 0, 'batch_receipt': 0, 'boundary': 0}

    class CUDAProfile(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [ast.unparse(n) for n in node.targets[0].elts]
                    == ['args.batch_size', 'args.training_epoch', 'args.device', 'args.cuda']):
                if ast.literal_eval(node.value) != (2, 1, 'cpu', False):
                    raise ValueError('Original CPU diagnostic budget changed')
                changes['budget_device'] += 1
                node.value = ast.parse("(args.batch_size, 1, 'cuda:0', True)", mode='eval').body
            elif (len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'torch.cuda.is_available'):
                if ast.unparse(node.value) != 'lambda: False':
                    raise ValueError('Original CUDA override changed')
                changes['cpu_override'] += 1
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'cpu_batch':
                    if not isinstance(node.values[i], ast.Constant) or node.values[i].value != 2:
                        raise ValueError('Original diagnostic batch receipt changed')
                    changes['batch_receipt'] += 1
                    node.keys[i] = ast.Constant('cuda_batch')
                    node.values[i] = ast.parse('args.batch_size', mode='eval').body
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == 'Real Air36 plumbing only; reduced diagnostic budget never satisfies a registered full experiment.':
                changes['boundary'] += 1
                return ast.copy_location(ast.Constant(
                    'Real Air36 CUDA memory preflight at the original full batch size. One train, validation and test batch only; not a formal experiment or benchmark performance.'), node)
            return node

    function = CUDAProfile().visit(function)
    if changes != dict.fromkeys(changes, 1):
        raise ValueError(f'Original diagnostic source requires review: {changes}')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['main'], changes


def verify_config(config, root=ROOT):
    if sha(root / config['queue']) != config['queue_sha256']:
        raise ValueError('Frozen formal queue differs')
    for source in config['source_receipts']:
        if sha(root / source['path']) != source['sha256']:
            raise ValueError('Frozen memory probe source differs: ' + source['path'])
    queue = json.loads((root / config['queue']).read_text(encoding='utf-8'))
    verify(queue, root)
    for profile in config['profiles']:
        job = next(j for j in queue['jobs'] if j['id'] == profile['job_id'])
        if job['dataset'] != 'air36' or job['arguments']['batch_size'] != 128:
            raise ValueError('Reviewed memory probe is restricted to full-batch Air36')
        if profile['output_directory'] == job['output_directory']:
            raise ValueError('Memory diagnostics cannot occupy a formal experiment slot')
    for key in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes',
                'emergency_commit_headroom_bytes', 'minimum_gpu_free_bytes', 'emergency_gpu_free_bytes']:
        if config['settings'][key] < queue['settings'][key]:
            raise ValueError('Memory preflight must not weaken formal resource floors')
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--profile', required=True)
    cli = parser.parse_args()
    config_path = ROOT / cli.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    queue = verify_config(config)
    profile = next(p for p in config['profiles'] if p['id'] == cli.profile)
    job = next(j for j in queue['jobs'] if j['id'] == profile['job_id'])
    output = ROOT / profile['output_directory']
    if output.exists():
        raise FileExistsError('Preserve previous memory diagnostic')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA memory probe cannot fall back to CPU')
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    bound, changes = bound_main()
    started = time.perf_counter()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], '--queue', config['queue'], '--job-id', job['id'],
                    '--integration-check', '--audit-output', profile['output_directory']]
        bound()
    finally:
        sys.argv = old_argv
    torch.cuda.synchronize(0)
    value = json.loads((output / 'integration_result.json').read_text(encoding='utf-8'))
    import numpy as np
    receipt = value['prediction_receipts'][0]
    if sha(Path(receipt['path'])) != receipt['sha256']:
        raise ValueError('Captured author CUDA prediction differs')
    with np.load(receipt['path']) as batch:
        shape = batch['batch_shape'].tolist()
    if shape[0] != job['arguments']['batch_size'] or value.get('cuda_batch') != shape[0]:
        raise ValueError('CUDA diagnostic did not execute the full registered batch')
    if not value['captured_predictions_match_native_errors'] or value['optimizer_updates'] != 1:
        raise ValueError('CUDA author execution proof incomplete')
    verify_config(config)
    write_json(output / 'cuda_memory_result.json', {
        'status': 'cuda_full_batch_memory_preflight_passed_not_benchmark', 'diagnostic': True,
        'profile': cli.profile, 'job_id': job['id'], 'formal_queue_sha256': sha(ROOT / config['queue']),
        'probe_config_sha256': sha(config_path), 'original_job_unchanged': True,
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda_version': torch.version.cuda, 'actual_test_batch_shape': shape,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'elapsed_seconds': time.perf_counter() - started, 'reviewed_ast_bindings': changes,
        'source_receipts': config['source_receipts'],
        'integration_result_sha256': sha(output / 'integration_result.json'),
        'full_registered_batch_size': shape[0], 'optimizer_updates': 1,
        'train_validation_test_batches_each': 1, 'formal_metrics_published': False,
        'boundary': 'Only real-data full-batch CUDA execution and memory evidence. Dataset scaling, full-epoch memory peaks, original-paper fidelity and benchmark performance are not established by this diagnostic.'})


if __name__ == '__main__':
    main()
