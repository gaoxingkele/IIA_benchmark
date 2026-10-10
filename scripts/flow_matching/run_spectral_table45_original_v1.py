"""Observe full released Table4/5 flows without replacing author model or loops."""
import argparse
import ast
from datetime import datetime, timezone
import functools
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write, execute_original_entry

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def original_args(entry, arguments):
    tree = ast.parse(Path(entry).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'define_args')
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(entry), 'exec'), namespace)
    return vars(namespace['define_args']().parse_args(arguments))


def verify(queue):
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen Table4/5 source differs: ' + item['path'])
    ids = []
    for job in queue['jobs']:
        cfg = read(ROOT / job['model_config'])
        if sha(ROOT / job['model_config']) != job['model_config_sha256']:
            raise ValueError('Frozen full model config differs')
        if original_args(ROOT / cfg['entry'], cfg['arguments']) != cfg['author_arguments']:
            raise ValueError('Original full argument contract differs')
        if cfg['author_arguments']['epochs'] != 600 or cfg['author_arguments']['batch_size'] != 64:
            raise ValueError('Full original 600 epochs and batch64 required')
        if cfg['diagnostic']:
            raise ValueError('No smoke model in full queue')
        ids.append(job['id'])
    if len(ids) != 18 or len(set(ids)) != 18:
        raise ValueError('All18 released Table4/5 commands required')


def prepare_workspace(queue, artifact):
    artifact = Path(artifact)
    if artifact.exists():
        raise FileExistsError('Preserve previous raw/workspace/model attempt')
    workspace = artifact / 'workspace'
    shutil.copytree(ROOT / queue['source_root'], workspace,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for item in queue['source_receipts']:
        relative = Path(item['path']).relative_to(queue['source_root']) if Path(item['path']).is_relative_to(queue['source_root']) else None
        if relative is not None and sha(workspace / relative) != item['sha256']:
            raise ValueError('Private original source/raw copy differs')
    return workspace


def row_multiset(array):
    import numpy as np
    rows = np.ascontiguousarray(array).reshape(len(array), -1)
    return sorted(sha_bytes(row.tobytes()) for row in rows)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def data_audit(queue, output):
    """Run original full CPU loaders, checking every cache/window and physics sample."""
    if Path(output).exists():
        raise FileExistsError('Preserve previous full data audit')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    workspace = prepare_workspace(queue, queue['data_audit_artifact'])
    sys.path.insert(0, str(workspace))
    os.chdir(workspace)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    original_load = torch.load
    def cpu_load(*args, **kwargs):
        kwargs.update(map_location='cpu', weights_only=False)
        return original_load(*args, **kwargs)
    torch.load = cpu_load
    try:
        from utils.utils_data import TimeDataset_irregular, TimeDataset_joint_irregular, real_data_loading, pendulum_nonlinear, MinMaxScaler
        raw = np.loadtxt(workspace / 'datasets/stock_data.csv', delimiter=',', skiprows=1)
        full = MinMaxScaler(raw[::-1])
        np.random.seed(10); torch.manual_seed(10)
        regular = np.asarray(real_data_loading('stock', 24))
        expected_regular = np.asarray([full[i:i+24] for i in range(len(full)-24)])
        if row_multiset(regular) != row_multiset(expected_regular):
            raise ValueError('Original regular stride-one window multiset differs')
        records = {'regular': dict(shape=list(regular.shape), full_windows_verified=True, samples=len(regular))}
        for rate in (.3, .5, .7):
            np.random.seed(10); torch.manual_seed(10)
            source = TimeDataset_irregular(24, 'stock', rate)
            expected = np.asarray([full[i:i+24] for i in range(len(full)-24+1)])
            if row_multiset(source.original_sample) != row_multiset(expected):
                raise ValueError('All original interpolation reference windows differ')
            mask = torch.randperm(len(full), generator=torch.Generator().manual_seed(56789))[:int(len(full)*rate)].numpy()
            dropped = full.copy(); dropped[mask] = np.nan
            dropped = np.concatenate((dropped, np.arange(len(full))[:, None]), axis=1)
            expected_masked = np.asarray([dropped[i:i+24] for i in range(len(full)-24+1)])
            # Cache permutation may differ; exact time columns bind each complete row.
            actual = np.asarray(source.samples)
            order = np.argsort(actual[:, 0, -1]); wanted = np.argsort(expected_masked[:, 0, -1])
            if not np.array_equal(actual[order], expected_masked[wanted], equal_nan=True):
                raise ValueError('Original full dropped-time mask differs')
            from controldiffeq import NaturalCubicSpline
            times = torch.arange(24, dtype=torch.float32)
            interpolated = torch.stack([NaturalCubicSpline(times, source.train_coeffs).evaluate(t) for t in times.unbind()], dim=1)
            if not torch.isfinite(interpolated).all():
                raise ValueError('Full original spline interpolation is nonfinite')
            records['interpolate_' + str(rate)] = dict(samples=len(source), shape=list(source.original_sample.shape),
                missing_entries=int(np.isnan(actual[:, :, :-1]).sum()), full_mask_and_reference_verified=True,
                all_four_spline_coefficients_finite=all(bool(torch.isfinite(c).all()) for c in source.train_coeffs),
                full_interpolation_shape=list(interpolated.shape))
            np.random.seed(10); torch.manual_seed(10)
            joint = TimeDataset_joint_irregular(24, 'stock', rate)
            retained = np.ones(len(full), dtype=bool); retained[mask] = False
            values = full[retained][1:]
            intervals = MinMaxScaler(np.diff(np.arange(len(full))[retained])[:, None])
            combined = np.concatenate((values, intervals), axis=1)
            expected_joint = np.asarray([combined[i:i+24] for i in range(len(combined)-24+1)])
            if row_multiset(np.asarray(joint.samples)) != row_multiset(expected_joint):
                raise ValueError('All joint value/time-interval windows differ')
            records['joint_' + str(rate)] = dict(samples=len(joint), shape=list(joint.samples.shape),
                full_value_and_interval_windows_verified=True,
                unused_released_joint_cache_directory='datasets/joint_stock' + str(rate),
                actual_original_directory='datasets_joint/stock' + str(rate))
        raw_pendulum = pendulum_nonlinear(2000, .08, t_max=10, dt=.25)
        original_tensor = torch.tensor(raw_pendulum, dtype=torch.float32)
        normalized = (original_tensor-original_tensor.mean([0, 1], keepdim=True))/original_tensor.std([0, 1], keepdim=True)
        if list(normalized.shape) != [2000, 40, 2] or not torch.isfinite(normalized).all():
            raise ValueError('Full2000 original pendulum trajectories required')
        # A second complete original simulation must be identical, including noise.
        if not np.array_equal(raw_pendulum, pendulum_nonlinear(2000, .08, t_max=10, dt=.25)):
            raise ValueError('Original fixed seed1 physics source differs')
        arrays = workspace.parent / 'full_pendulum_source.npz'
        np.savez_compressed(arrays, minmax=raw_pendulum, training=normalized.numpy())
        records['pendulum'] = dict(samples=2000, shape=[2000, 40, 2], arrays=str(arrays), arrays_sha256=sha(arrays),
            original_data_seed=1, paper_gravity=9.8, source_gravity=9.81,
            original_normalization='MinMax over sample axis, then global torch mean/std',
            paper_normalization='each trajectory in [0,1]', no_heldout_generator_split=True)
        proof = dict(passed=True, diagnostic_only=True, formal_benchmark_result=False,
            queue_sha256=sha(ROOT / queue['queue_path']), records=records, original_full_sources_verified=True,
            workspace=str(workspace), stock_raw_shape=list(raw.shape), paired_generator_seed=10,
            boundary='Original generation tasks fit all data; this audit does not establish generalization or GPU model capacity.')
        write(output, proof)
        return proof
    finally:
        torch.load = original_load


def seed_worker(worker_id):
    """Picklable version of the exact author local worker callback for Windows."""
    import torch
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_worker_seed_source(entry):
    tree = ast.parse(Path(entry).read_text(encoding='utf-8'))
    actual = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'seed_worker')
    expected = ast.parse(inspect.getsource(seed_worker)).body[0].body[4:]
    if [ast.dump(n) for n in actual.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Original worker callback changed; cannot substitute')


def validate_full_receipt(receipt, config):
    epochs = config['author_arguments']['epochs']
    loaders = receipt['generator_loaders']
    if len(loaders) != 1:
        raise ValueError('Exactly one full original generator loader required')
    loader = loaders[0]
    expected = math.ceil(loader['samples'] / 64)
    if loader['batch_size'] != 64 or loader['drop_last'] or loader['num_workers'] != config['num_workers']:
        raise ValueError('Original full batch/worker/data coverage changed')
    if receipt['epoch_updates'] != {str(i): expected for i in range(epochs)} or receipt['optimizer_updates'] != epochs*expected:
        raise ValueError('Full600-epoch optimizer coverage not completed')
    required = 5 if config['table'] == 5 else 10
    if len(receipt['metric_repeats']) != required or not all(math.isfinite(v) for v in receipt['metric_repeats']):
        raise ValueError('All original full evaluator repeats required')
    if receipt['generated_shape'] != receipt['reference_shape'] or receipt['generated_shape'][0] != loader['samples']:
        raise ValueError('Full original generated/reference arrays differ')
    if not receipt.get('checkpoint_sha256') or not receipt.get('arrays_sha256'):
        raise ValueError('Actual full checkpoint and arrays required')


def run_job(queue, job):
    cfg = read(ROOT / job['model_config'])
    proof = read(ROOT / queue['data_audit_output'])
    if not proof['passed'] or proof['queue_sha256'] != sha(ROOT / queue['queue_path']):
        raise ValueError('Complete frozen original data audit required')
    artifact = Path(job['artifact_directory'])
    workspace = prepare_workspace(queue, artifact)
    entry = workspace / Path(cfg['entry']).name
    check_worker_seed_source(entry)
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(workspace)); os.chdir(workspace)
    import torch
    import numpy as np
    torch.set_num_threads(queue['settings']['cpu_threads'])
    if not torch.cuda.is_available():
        raise ValueError('Original full CUDA model execution required')
    out = ROOT / job['output_directory']; out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', id=job['id'], table=cfg['table'], full_original_budget=True,
        diagnostic_only=False, queue_sha256=sha(ROOT / queue['queue_path']), model_config_sha256=sha(ROOT / job['model_config']),
        source_entry_sha256=sha(entry), optimizer_updates=0, epoch_updates={}, generator_loaders=[], metric_repeats=[],
        compatibility=['Windows local worker callback mapped to identical picklable module callback, worker count retained',
            'Trusted frozen author tensor caches loaded with weights_only=False for torch2.8 compatibility', 'Noninteractive plot rendering with Agg'])
    originals = []; holder = {}; arrays_saved = False
    def persist():
        write(out / 'progress.json', state)
    old_load = torch.load
    def compatible_load(*a, **kw):
        kw.setdefault('weights_only', False)
        return old_load(*a, **kw)
    torch.load = compatible_load; originals.append((torch, 'load', old_load))
    loader_type = torch.utils.data.DataLoader
    old_init = loader_type.__init__
    @functools.wraps(old_init)
    def observed_loader(loader, *a, **kw):
        caller = inspect.currentframe().f_back
        direct = Path(caller.f_code.co_filename).resolve() == entry
        if direct and kw.get('num_workers', 0) and kw.get('worker_init_fn') is not None:
            if kw['worker_init_fn'].__name__ != 'seed_worker':
                raise ValueError('Unexpected original callback')
            kw['worker_init_fn'] = seed_worker
        old_init(loader, *a, **kw)
        if direct:
            state['generator_loaders'].append(dict(samples=len(loader.dataset), batch_size=loader.batch_size,
                num_workers=loader.num_workers, drop_last=loader.drop_last))
            key = cfg['data_audit_key']
            if len(loader.dataset) != proof['records'][key]['samples']:
                raise ValueError('Frozen original full data coverage differs')
            persist()
    loader_type.__init__ = observed_loader; originals.append((loader_type, '__init__', old_init))
    old_step = torch.optim.Adam.step
    @functools.wraps(old_step)
    def observed_step(optimizer, *a, **kw):
        caller = inspect.currentframe().f_back
        direct = Path(caller.f_code.co_filename).resolve() == entry
        value = old_step(optimizer, *a, **kw)
        if direct:
            epoch = str(int(caller.f_locals['epoch']))
            state['optimizer_updates'] += 1
            state['epoch_updates'][epoch] = state['epoch_updates'].get(epoch, 0) + 1
            holder.update(frame=caller, optimizer=optimizer)
            losses = caller.f_locals.get('losses', [caller.f_locals.get('loss')])
            if not bool(torch.isfinite(losses[0]).all()):
                raise ValueError('Nonfinite original generator objective')
            if state['optimizer_updates'] % 100 == 0:
                persist()
        return value
    torch.optim.Adam.step = observed_step; originals.append((torch.optim.Adam, 'step', old_step))
    def capture_arrays(real, generated):
        nonlocal arrays_saved
        if arrays_saved:
            return
        real = np.asarray(real); generated = np.asarray(generated)
        if not np.isfinite(real).all() or not np.isfinite(generated).all():
            raise ValueError('Nonfinite full generated/reference data')
        arrays = artifact / 'full_evaluation_arrays.npz'
        np.savez_compressed(arrays, reference=real, generated=generated)
        state.update(reference_shape=list(real.shape), generated_shape=list(generated.shape),
                     arrays=str(arrays), arrays_sha256=sha(arrays))
        arrays_saved = True
    if cfg['table'] == 4:
        module = importlib.import_module('metrics.discriminative_torch')
        old_metric = module.discriminative_score_metrics
        @functools.wraps(old_metric)
        def observed_metric(real, generated, args):
            capture_arrays(real, generated)
            before = state['optimizer_updates']; value = float(old_metric(real, generated, args))
            if before != state['optimizer_updates']:
                raise ValueError('Evaluator updates incorrectly counted as generator training')
            state['metric_repeats'].append(value); persist(); return value
        module.discriminative_score_metrics = observed_metric
        originals.append((module, 'discriminative_score_metrics', old_metric))
    else:
        module = importlib.import_module('metrics.cross_correlation')
        cls = module.CrossCorrelLoss; old_metric = cls.compute
        @functools.wraps(old_metric)
        def observed_metric(loss, generated):
            caller = inspect.currentframe().f_back
            if Path(caller.f_code.co_filename).resolve() != entry:
                raise ValueError('Original physics evaluator caller differs')
            capture_arrays(caller.f_locals['x_real'].numpy(), caller.f_locals['x_fake'].numpy())
            value = old_metric(loss, generated)
            state['metric_repeats'].append(float(value.item())); persist(); return value
        cls.compute = observed_metric; originals.append((cls, 'compute', old_metric))
    begin = time.perf_counter(); persist()
    try:
        execute_original_entry(entry, cfg['arguments'])
        local = holder['frame'].f_locals
        checkpoint = artifact / 'final_full_checkpoint.pt'
        torch.save(dict(model=local['model'].state_dict(), optimizer=holder['optimizer'].state_dict(),
            ema=local['ema'].state_dict() if 'ema' in local else None, arguments=vars(local['args']),
            full_budget_completed=True, optimizer_updates=state['optimizer_updates'], epoch_updates=state['epoch_updates'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=__import__('random').getstate()), checkpoint)
        state.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        validate_full_receipt(state, cfg)
        values = state['metric_repeats']; n = len(values)
        from scipy.stats import t
        mean = float(np.mean(values)); std = float(np.std(values, ddof=1)); se = std / math.sqrt(n)
        state.update(status='completed', seconds=time.perf_counter()-begin, completed_utc=datetime.now(timezone.utc).isoformat(),
            metric=cfg['metric'], mean=mean, std=std, population_std=float(np.std(values)), se=se,
            ci95=[mean-float(t.ppf(.975, n-1))*se, mean+float(t.ppf(.975, n-1))*se],
            spread_scope='Evaluator repetitions within one generator seed10; no between-training-seed interval',
            paper_claim=cfg['paper_claim'], author_equivalence_certified=False, formal_strict_tsad_benchmark=False)
        write(out / 'result.json', state)
    except BaseException as error:
        state.update(status='failed_or_partial_preserved', error=repr(error), seconds=time.perf_counter()-begin)
        write(out / 'failure.json', state); raise
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def verify_result(queue, job):
    result = read(ROOT / job['output_directory'] / 'result.json')
    cfg = read(ROOT / job['model_config'])
    validate_full_receipt(result, cfg)
    if result['status'] != 'completed' or result['queue_sha256'] != sha(ROOT / queue['queue_path']) or result['model_config_sha256'] != sha(ROOT / job['model_config']):
        raise ValueError('Full result source binding differs')
    for path, binding in [('arrays', 'arrays_sha256'), ('checkpoint', 'checkpoint_sha256')]:
        if sha(result[path]) != result[binding]:
            raise ValueError('Full model/array artifact differs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--job-id')
    parser.add_argument('--audit-data', action='store_true'); parser.add_argument('--verify-result', action='store_true')
    args = parser.parse_args(); queue = read(ROOT / args.queue); verify(queue)
    if args.audit_data:
        data_audit(queue, ROOT / queue['data_audit_output'])
    else:
        job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
        if args.verify_result:
            result = verify_result(queue, job); print(json.dumps(dict(id=job['id'], completed=result['status']=='completed')))
        else:
            run_job(queue, job)


if __name__ == '__main__':
    main()
