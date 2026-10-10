"""Execute downloaded LS4 USHCN climate training/evaluation with full-budget observers."""
import argparse
import functools
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.ls4_cauchy_fallback_repair_v1 import install


def budget(config):
    if config['epochs'] != 500 or config['batch_size'] != 64 or config['variables'] != 5:
        raise ValueError('Both full original climate YAML budgets required')
    if config['train_series'] <= 0 or config['test_series'] <= 0:
        raise ValueError('Full audited original split counts required')
    return dict(train_batches=math.ceil(config['train_series']/64),
                test_batches=math.ceil(config['test_series']/64),
                optimizer_updates=500*math.ceil(config['train_series']/64), evaluation_calls=50)


def periodic_epochs(epochs, interval):
    return [e for e in range(epochs) if e > 0 and e % interval == 0]


def verify_job_scope(queue):
    expected = {(b,f,p,s) for b in queue['split_branches'] for f in queue['folds']
                for p in queue['profiles'] for s in queue['seeds']}
    if len(queue['jobs']) != 100 or len({j['id'] for j in queue['jobs']}) != 100:
        raise ValueError('All100 original/ID-corrected fold/profile/seed jobs required')
    if {(j['split_branch'],j['fold'],j['profile'],j['seed']) for j in queue['jobs']} != expected:
        raise ValueError('A complete frozen Cartesian experiment scope is required')


def verify(queue):
    for item in queue['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Pinned full climate pipeline input changed: '+item['path'])
    proof = read(ROOT/queue['loader_proof'])
    if not proof['full_original_loader_passed'] or proof['full_profiles'] != 20 or proof['full_batches'] != 360:
        raise ValueError('Both split branches/all5folds/both original loaders required')
    lock = read(ROOT/queue['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(k)!=v for k,v in lock['packages'].items()):
        raise ValueError('Original locked LS4 environment required')
    verify_job_scope(queue)
    for job in queue['jobs']:
        config = read(ROOT/job['model_config'])
        budget(config)
        if config['author_equivalence_certified']:
            raise ValueError('Reconstructed splits cannot claim historical author equivalence')


class ObservedLoader:
    """Observe complete original batches without editing, filtering or resampling."""
    def __init__(self, loader, expected_patients, callback):
        self.loader, self.expected, self.callback = loader, expected_patients, callback
        self.calls = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        covered = 0
        for index, data in enumerate(self.loader):
            count = len(data['observed_data'])
            if count != min(self.loader.batch_size, self.expected - covered):
                raise ValueError('Original full batch coverage differs')
            covered += count
            self.callback(index, covered, data)
            yield data
        if covered != self.expected:
            raise ValueError('Every original patient required')
        self.calls += 1


def create_runtime(queue, config, artifact):
    artifact = Path(artifact)
    if artifact.exists():
        raise FileExistsError('Preserve previous model/diagnostic attempt')
    target = artifact / 'workspace'
    shutil.copytree(ROOT / queue['source_root'], target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for item in queue['source_receipts']:
        path = Path(item['path'])
        if path.is_relative_to(queue['source_root']):
            if sha(target / path.relative_to(queue['source_root'])) != item['sha256']:
                raise ValueError('Private source copy differs')
    sys.path.insert(0, str(target))
    os.chdir(target)
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    import datasets
    import models.s4 as s4
    native = importlib.import_module('train_interp_extrap')
    correction = install(s4, read(ROOT / queue['cauchy_correction_config']), ROOT)
    yaml = OmegaConf.load(ROOT / config['source_config'])
    # Only the machine-specific data path and local output document are bound.
    yaml.data.base = str((ROOT / queue['raw_csv_path']).parent.resolve())
    yaml.data.dir = str((ROOT / config['split_directory']).resolve())
    yaml.doc = config['profile']
    if (OmegaConf.to_container(yaml.optim, resolve=True) != config['original_optim']
            or OmegaConf.to_container(yaml.model, resolve=True) != config['original_model']
            or yaml.data.channel != 5 or bool(yaml.data.extrap) != config['extrapolation']
            or yaml.data.T_val != 150 or yaml.data.max_val_samples != 3):
        raise ValueError('Original YAML scientific scope changed')
    resolved = artifact / 'resolved_original.yaml'
    OmegaConf.save(yaml, resolved)
    torch.set_num_threads(queue['cpu_threads'])
    torch.random.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    logs = []
    # Local observational logging only; no external messages or service writes.
    native.wandb.init = lambda *a, **kw: None
    native.wandb.save = lambda *a, **kw: None
    def log(values, *a, **kw):
        row = {k: float(v) for k, v in values.items()}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite original loss/evaluation')
        logs.append(row)
    native.wandb.log = log
    original_parse = native.parse_datasets
    data_receipts = []
    def checked_parse(arguments, batch_size, device):
        objects = original_parse(arguments, batch_size=batch_size, device=device)
        expected = budget(config)
        if objects['n_train_batches'] != expected['train_batches'] or objects['n_test_batches'] != expected['test_batches']:
            raise ValueError('Original batch counts differ')
        proof = next(p for p in read(ROOT/queue['loader_proof'])['profiles']
                     if p['branch']==config['split_branch'] and p['fold']==config['fold']
                     and p['extrapolation']==config['extrapolation'])
        train_sets = objects['train_dataloader'].dataset.datasets
        expected_counts = proof['grouped_series_counts']
        if (len(train_sets[0]) != expected_counts['train'] or len(train_sets[1]) != expected_counts['val']
                or len(objects['test_dataloader'].dataset) != expected_counts['test']):
            raise ValueError('Full original grouped climate split counts changed')
        closure = dict(zip(objects['train_dataloader'].collate_fn.__code__.co_freevars,
                           [c.cell_contents for c in objects['train_dataloader'].collate_fn.__closure__]))
        for field,name in [('data_min','normalization_min'),('data_max','normalization_max')]:
            if not np.array_equal(closure[field].numpy(), np.array(proof[name],dtype=np.float32)):
                raise ValueError('Original train-plus-validation-prefix normalizer changed')
        data_receipts.append(dict(full_train_series=config['train_series'],full_test_series=config['test_series'],
            source_original_series=1114,full_variables=objects['input_dim'],original_batch_size=batch_size,
            split_branch=config['split_branch'],fold=config['fold'],split_hashes=config['split_hashes'],
            original_raw_csv_sha256=queue['raw_csv_sha256'],source_profile=proof,**expected))
        return objects
    native.parse_datasets = checked_parse
    return torch, np, native, yaml, resolved, correction, logs, data_receipts


def diagnostic(queue, config, artifact, cpu):
    torch, np, native, yaml, resolved, correction, logs, receipts = create_runtime(queue, config, artifact)
    device = 'cpu' if cpu else 'cuda'
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('Full GPU capacity diagnostic requires actual CUDA')
    objects = native.parse_datasets(yaml.data, config['batch_size'], torch.device('cpu'))
    yaml.model.n_labels = objects.get('n_labels', 1)
    model = native.VAE(yaml.model).to(device)
    optimizer, scheduler = native.setup_optimizer(model, **{k: config['original_optim'][k] for k in ('lr', 'weight_decay', 'epochs')})
    steps = []
    original_step = optimizer.step
    @functools.wraps(original_step)
    def step(*a, **kw):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Full original model gradient nonfinite')
        value = original_step(*a, **kw)
        steps.append(1)
        return value
    optimizer.step = step
    train_loader, test_loader = objects['train_dataloader'], objects['test_dataloader']
    first_train = next(iter(train_loader))
    original_first_shape = list(first_train['observed_data'].shape)
    original_future_shape = list(first_train['data_to_predict'].shape)
    # CPU interface proof retains the whole original first batch and time grid.
    # Full GPU capacity proof trains every batch and evaluates allconfig['test_series'] patients.
    if cpu:
        train_loader, test_loader = [first_train], [next(iter(test_loader))]
    else:
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    actual = native.train(model, device, train_loader, optimizer, extrap=config['extrapolation'], step=0)
    trained = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(dict(model=trained.state_dict(), optimizer=optimizer.state_dict(), optimizer_updates=actual), Path(artifact) / 'diagnostic_trained_model.pt')
    native.eval(model, device, -1, test_loader, float('inf'), str(artifact), classify=False, extrap=config['extrapolation'])
    expected = 1 if cpu else budget(config)['train_batches']
    if actual != expected or len(steps) != expected or not any('eval_full_nll' in r for r in logs):
        raise ValueError('Original full gradient/optimizer/native NLL interface incomplete')
    return dict(profile=config['profile'], seed=config['seed'], original_batch_size=config['batch_size'],
        actual_optimizer_updates=actual, full_test_batches=len(test_loader), full_model_parameters=sum(p.numel() for p in model.parameters()),
        native_evaluation_interface_passed=True, full_patient_capacity_passed=not cpu,
        original_first_observed_shape=original_first_shape, original_first_future_shape=original_future_shape,
        full_original_first_batch_retained=True, diagnostic_only=True, performance_result=False,
        backend_correction=correction, data_receipts=receipts, evaluation_logs=[r for r in logs if 'eval_mse' in r])


def run_job(queue, job):
    config = read(ROOT / job['model_config'])
    artifact, output = Path(job['artifact_directory']), ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('Preserve original full training attempt')
    output.mkdir(parents=True)
    torch, np, native, yaml, resolved, correction, logs, data_receipts = create_runtime(queue, config, artifact)
    if not torch.cuda.is_available():
        raise RuntimeError('Full native USHCN queue requires GPU, no CPU fallback')
    started = time.perf_counter()
    updates, epochs, evaluations = 0, [], []
    optimizer_ref, scheduler_ref, model_ref = [], [], []
    original_setup, original_train, original_eval = native.setup_optimizer, native.train, native.eval
    def save_bundle(epoch, phase, name='training_state.pt'):
        model = model_ref[0]
        path = artifact / name
        temporary = artifact / (name+'.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer_ref[0].state_dict(),
            scheduler=scheduler_ref[0].state_dict(), epoch=epoch, phase=phase, optimizer_updates=updates,
            torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), job=job,
            queue_sha256=sha(ROOT / queue['queue_path']), config_sha256=sha(ROOT / job['model_config'])), temporary)
        temporary.replace(path)
        return path
    def setup(model, *a, **kw):
        optimizer, scheduler = original_setup(model, *a, **kw)
        optimizer_ref.append(optimizer); scheduler_ref.append(scheduler)
        original_step = optimizer.step
        @functools.wraps(original_step)
        def step(*a, **kw):
            nonlocal updates
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Original full training gradient nonfinite')
            value = original_step(*a, **kw); updates += 1
            return value
        optimizer.step = step
        return optimizer, scheduler
    def train(model, device, loader, optimizer, extrap=False, step=0):
        if not model_ref:
            model_ref.append(model)
        epoch, before = len(epochs), updates
        def batch(index, covered, data):
            write(output / 'progress.json', dict(phase='native_full_training', epoch=epoch,
                optimizer_updates=updates, batch=index, covered_train_series=covered, full_train_series=config['train_series']))
        observed = ObservedLoader(loader, config['train_series'], batch)
        returned = original_train(model, device, observed, optimizer, extrap=extrap, step=step)
        if returned != updates or updates - before != budget(config)['train_batches'] or observed.calls != 1:
            raise ValueError('Original complete epoch/update coverage differs')
        epochs.append(dict(epoch=epoch, optimizer_updates=updates, full_train_batches=len(loader)))
        # Persist full weights/optimizer/RNG before potentially expensive evaluation.
        save_bundle(epoch, 'after_train_before_native_eval_and_scheduler')
        write(output / 'training_observation.json', dict(epochs=epochs, optimizer_updates=updates))
        return returned
    def evaluate(model, device, epoch, loader, best_mse, log_dir, classify=None, extrap=False):
        start = len(logs)
        # Native reconstruction/NLL consume random numbers. Preserve the actual
        # pre-evaluation model and RNG for independent best/final-model replay.
        candidate = save_bundle(epoch, 'before_native_full_evaluation', 'evaluation_candidate_state.pt')
        def batch(index, covered, data):
            write(output / 'progress.json', dict(phase='native_full_evaluation', epoch=epoch,
                batch=index, covered_test_series=covered, full_test_series=config['test_series'], optimizer_updates=updates))
        observed = ObservedLoader(loader, config['test_series'], batch)
        value = original_eval(model, device, epoch, observed, best_mse, log_dir, classify=classify, extrap=extrap)
        metrics = [r for r in logs[start:] if 'eval_mse' in r]
        if len(metrics) != 1 or observed.calls != 1 or len(observed) != budget(config)['test_batches']:
            raise ValueError('Full original MSE/CRPS/NLL test coverage differs')
        if metrics[0]['eval_mse'] < best_mse:
            shutil.copy2(candidate, artifact/'best_test_evaluated_state.pt')
        if epoch == -1:
            shutil.copy2(candidate, artifact/'final_evaluated_state.pt')
        evaluations.append(dict(epoch=epoch, full_test_series=config['test_series'], full_test_batches=len(observed), **metrics[0]))
        write(output / 'evaluation_observation.json', dict(evaluations=evaluations))
        return value
    native.setup_optimizer, native.train, native.eval = setup, train, evaluate
    args = native.get_parse_args().parse_args(['--config', str(resolved), '--seed', str(config['seed']),
        '--log_dir', str(artifact / 'native_outputs')])
    native.main(args)
    expected = budget(config)
    if (len(epochs) != 500 or updates != expected['optimizer_updates']
            or [r['epoch'] for r in evaluations] != periodic_epochs(500, 10) + [-1]):
        raise ValueError('Every500 native epochs and50 complete evaluations required')
    save_bundle(499, 'completed_native_main_final_evaluation_scheduler_complete')
    paths = [artifact / 'training_state.pt', artifact/'best_test_evaluated_state.pt',
             artifact/'final_evaluated_state.pt', resolved,
             output / 'training_observation.json', output / 'evaluation_observation.json']
    best_path = artifact / 'native_outputs' / 'climate' / config['profile'] / 'checkpoints' / 'ckpt.pth'
    best = torch.load(best_path, map_location='cpu')
    paths.append(best_path)
    verify(queue)
    result = dict(job=job, config_sha256=sha(ROOT / job['model_config']), queue_sha256=sha(ROOT / queue['queue_path']),
        epochs_completed=len(epochs), optimizer_updates=updates, evaluations=evaluations, final_metrics=evaluations[-1],
        best_native_test_selected_mse=best['mse'], best_native_epoch=best['epoch'],
        total_native_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model_ref[0].parameters()),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), data_receipts=data_receipts,
        backend_correction=correction, author_equivalence_certified=False, all_original_experiments_complete=False,
        selection_boundary='Native periodic test MSE selection; train plus validation prefix min-max normalizer; source future samples unused; interpolation reconstructs its observed input; literal/ID-corrected split branches remain separate',
        artifacts=[dict(path=str(p), sha256=sha(p)) for p in paths])
    write(output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id')
    parser.add_argument('--integration-check', action='store_true')
    parser.add_argument('--gpu-preflight', action='store_true')
    args = parser.parse_args()
    queue = read(ROOT / args.queue); verify(queue)
    if args.job_id:
        run_job(queue, next(j for j in queue['jobs'] if j['id'] == args.job_id)); return
    if not (args.integration_check or args.gpu_preflight):
        parser.error('Choose a full job or diagnostic')
    name = 'integration' if args.integration_check else 'gpu_preflight'
    # One process per profile avoids native import globals/caches crossing tasks.
    case = next(c for c in queue['capacity_cases'] if c['id'] == os.environ['LS4_USHCN_CASE'])
    config = read(ROOT / case['model_config'])
    artifact = Path(queue[name + '_artifact']) / case['id']
    proof = diagnostic(queue, config, artifact, cpu=args.integration_check)
    write(ROOT / queue[name + '_output'] / (case['id'] + '.json'), dict(passed=True,
        queue_sha256=sha(ROOT / args.queue), **proof))


if __name__ == '__main__':
    main()
