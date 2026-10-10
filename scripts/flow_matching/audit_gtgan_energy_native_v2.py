"""Run complete original Energy data processing and construct all native networks."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.audit_gtgan_native_models_v1 import original_gtgan_branch
from scripts.flow_matching.gtgan_energy_native_common_v1 import verify_recipe


def write(path, value):
    # Python3.8 Path.write_text has no newline keyword. Preserve atomic LF JSON.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def original_loader_module(source):
    tree = ast.parse(source)
    names = {'normalize', 'to_tensor', 'load_data', 'save_data', 'TimeDataset'}
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Complete original Energy loader is required')

    class CPU(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                if node.args or node.keywords:
                    raise ValueError('Unrecognized native device binding')
                self.count += 1
                node.func.attr = 'to'
                node.args = [ast.Constant(value='cpu')]
            return node

    mapper = CPU()
    module = mapper.visit(ast.Module(body=nodes, type_ignores=[]))
    if mapper.count != 2:
        raise ValueError('Only the original two data preprocessing CUDA bindings may move to CPU')
    return ast.fix_missing_locations(module)


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen Energy source changed: ' + item['path'])


def worker(config):
    lock = read(ROOT / config['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k, v in lock['packages'].items()):
        raise ValueError('Frozen original-version environment differs')
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    torch.set_num_threads(2)
    verify(config)
    source = ROOT / config['source_root']
    runtime = Path(config['runtime_root'])
    prep = read(runtime / 'runtime_preparation_receipt.json')
    repaired = {x['path']: x['repaired_sha256'] for x in prep['runtime_repairs']}
    if len(prep['complete_copied_assets']) != 245 or not prep['original_source_unchanged']:
        raise ValueError('Complete original source copy is required')
    for item in prep['complete_copied_assets']:
        if sha(runtime / item['path']) != repaired.get(item['path'], item['original_sha256']):
            raise ValueError('Frozen isolated runtime changed: ' + item['path'])
    sys.path.insert(0, str(runtime))
    cde = importlib.import_module('controldiffeq')
    output = ROOT / config['output_directory']
    artifacts = Path(config['artifact_root'])
    if output.exists() or artifacts.exists():
        raise FileExistsError('Preserve every complete-data/model audit attempt')
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    namespace = dict(np=np, torch=torch, os=os, here=artifacts, controldiffeq=cde)
    loader_ast = original_loader_module((source / 'time_dataset.py').read_text(encoding='utf-8'))
    exec(compile(loader_ast, str(source / 'time_dataset.py'), 'exec'), namespace)
    model_config = read(ROOT / config['model_config'])
    data_config = read(ROOT / model_config['dataset_config'])
    raw = ROOT / data_config['raw_file']
    raw_sha = sha(raw)
    data = np.loadtxt(raw, delimiter=',', skiprows=1)[::-1]
    if data.shape != (data_config['raw_rows'], data_config['variables']):
        raise ValueError('Full raw Energy rows and all variables required')
    normalized = (data - data.min(0)) / (np.ptp(data, axis=0) + 1e-7)
    spec = importlib.util.spec_from_file_location('native_gtgan_energy_audit', runtime / 'GTGAN_energy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for case in model_config['cases']:
        expected = verify_recipe(case, source)
        torch.manual_seed(case['original_seed'])
        np.random.seed(case['original_seed'])
        loader = namespace['TimeDataset'](raw, expected['sequence_length'], 'energy', case['missing_rate'])
        samples = np.asarray(loader.samples)
        truth = np.asarray(loader.original_sample)
        windows, length, variables = expected['full_generation_windows'], expected['sequence_length'], expected['variables']
        if samples.shape != (windows, length, variables + 1) or truth.shape != (windows, length, variables):
            raise ValueError('Full Energy window and variable coverage differs')
        indices = samples[:, :, -1].astype(np.int64)
        if not np.array_equal(np.sort(indices[:, 0]), np.arange(windows)) or not np.array_equal(indices, indices[:, 0, None] + np.arange(length)[None, :]):
            raise ValueError('Full shuffled original row identities differ')
        if not np.array_equal(truth, normalized[indices]):
            raise ValueError('Original full-data normalization or truth differs')
        removed = torch.randperm(len(data), generator=torch.Generator().manual_seed(56789))[:int(len(data) * case['missing_rate'])].sort().values.numpy()
        raw_mask = np.zeros(len(data), dtype=bool)
        raw_mask[removed] = True
        mask = np.broadcast_to(raw_mask[indices][..., None], truth.shape)
        if not np.array_equal(np.isnan(samples[:, :, :-1]), mask) or not np.array_equal(samples[:, :, :-1][~mask], truth[~mask]):
            raise ValueError('Original whole-row mask or observed values differ')
        coeffs = [c.detach().numpy() for c in loader.train_coeffs]
        if any(c.shape != (windows, length - 1, variables) or not np.isfinite(c).all() for c in coeffs):
            raise ValueError('Complete original cubic coefficients must be finite')
        artifact = artifacts / (case['id'] + '.npz')
        np.savez_compressed(artifact, samples=samples, truth=truth, raw_row_ids=indices,
            removed_raw_row_ids=removed, **{'coefficient_' + str(i): c for i, c in enumerate(coeffs)})
        branch = original_gtgan_branch((source / 'GTGAN_energy.py').read_text(encoding='utf-8'))
        torch.manual_seed(case['original_seed'])
        ns = dict(module.__dict__)
        parsed = dict(case['parsed_original_arguments'], effective_shape=24)
        ns.update(args=SimpleNamespace(**parsed), input_size=28, hidden_size=24, num_layers=3, x_hidden=48, device=torch.device('cpu'))
        exec(compile(branch, str(source / 'GTGAN_energy.py') + ':original_constructor', 'exec'), ns)
        counts = {}
        for name in ('embedder', 'generator', 'supervisor', 'recovery', 'discriminator'):
            native = ns[name]
            if not all(torch.isfinite(p).all().item() for p in native.parameters()):
                raise ValueError('Nonfinite constructor weights')
            counts[name] = sum(p.numel() for p in native.parameters())
        weights = []
        directory = 'energy' if case['missing_rate'] == 0 else 'energy_0.5'
        for name in ('generator', 'recovery'):
            path = runtime / 'energy_model' / directory / (name + '.pt')
            state = torch.load(str(path), map_location='cpu')
            ns[name].load_state_dict(state, strict=True)
            if any(not torch.isfinite(v).all().item() for v in state.values()):
                raise ValueError('Released checkpoint has nonfinite state')
            weights.append(dict(model=name, path=str(path), sha256=sha(path), state_keys=len(state), strict_load_passed=True))
        rows.append(dict(case=case['id'], missing_rate=case['missing_rate'], raw_rows=len(data),
            variables=variables, complete_windows=windows, sequence_length=length,
            removed_raw_rows=len(removed), missing_window_values=int(mask.sum()),
            original_seed56789_whole_row_mask_exact=True, full_original_truth_and_time_row_identity=True,
            all_observed_values_exact=True, spline_coefficient_shapes=[list(c.shape) for c in coeffs],
            full_original_cubic_coefficients_finite=True, artifact_path=str(artifact), artifact_sha256=sha(artifact),
            exact_original_constructor=True, counts=counts, released_weight_checks=weights,
            expected_budget=expected, native_training_complete=False, full_model_inference_complete=False))
        write(output / 'progress.json', dict(completed_profiles=len(rows), required_profiles=len(model_config['cases'])))
        del loader, samples, truth, coeffs, mask, ns
        gc.collect()
    if sha(raw) != raw_sha or torch.cuda.is_initialized():
        raise ValueError('Raw preservation and CPU-only audit required')
    verify(config)
    write(output / 'native_energy_data_and_model_audit.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(ROOT / config['config_path']),
        full_original_constructor_cases=len(rows), cases=rows, raw_csv_unchanged=True,
        cuda_context_initialized=False, full_raw_data_processing_verified=True,
        original_generator_heldout_split_present=False, preprocessing_device_only_cpu=True,
        source_receipts=config['source_receipts'], author_equivalence_certified=False,
        performance_result=False, all_original_experiments_complete=False, boundary=config['boundary']))
    print(dict(complete_energy_profiles=len(rows), full_windows_per_profile=19712, passed=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    c = read(ROOT / a.config)
    verify(c)
    if c['config_path'] != a.config:
        raise ValueError('Frozen Energy audit binding differs')
    if a.worker:
        worker(c)
        return
    api = resource_api()
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Native Energy audit controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)

    def update(values):
        state.update(values)
        write(base / 'status.json', state)

    try:
        with api['resource_slot'](ROOT / c['resource_lock'], c['settings'], update):
            command = [str(ROOT / c['python']), '-X', 'utf8', '-u', '-m',
                'scripts.flow_matching.audit_gtgan_energy_native_v2', '--config', a.config, '--worker']
            env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='2', MPLBACKEND='Agg')
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                    stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(active_pid=child.pid, active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, c, api, update)
        proof = ROOT / c['output_directory'] / 'native_energy_data_and_model_audit.json'
        passed = code == 0 and proof.exists() and read(proof)['passed']
        write(base / 'receipt.json', dict(exit_code=code, full_native_energy_audit_passed=passed, **usage))
        update(dict(active_pid=None, state='completed' if passed else 'failed_preserved'))
        if not passed:
            raise RuntimeError('Full Energy audit failed; preserve the attempt')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
