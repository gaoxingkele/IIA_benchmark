"""Guarded full native single-cell runs from the pinned original author source."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import otcfm_singlecell_native_contract_v1 as contract

MODULE = 'scripts.flow_matching.run_otcfm_singlecell_native_v3'
QUEUE = 'configs/experiments/fm_otcfm_singlecell_native_full120.v3.json'
ENVIRONMENT = 'configs/reproducibility/fm_otcfm_author_core_py310.lock.v1.json'
AUTHOR = 'experiments/runs/flow_matching_campaign/sources/ot_cfm/original'


def identity(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def registration():
    env = read(ROOT / ENVIRONMENT)
    if not env['native_source_imports_passed']:
        raise ValueError('Actual complete original-author core imports are required')
    acquisition = read(ROOT / 'configs/acquisition/fm_otcfm_author_singlecell_mendeley.v1.json')
    released = next(s for s in acquisition['sources'] if s['filename'] == 'op_train_multi_targets_0.h5ad')
    public_path = 'configs/data/fm_otcfm_multiome13176_public_independent_author_input.v1.json'
    public = read(ROOT / public_path)
    packaged = read(ROOT / public['output_directory'] / 'input_packaging_report.json')
    data = dict(released_author_donor32606=dict(path=released['path'], sha256=released['checksum'].split(':')[1],
        cells=33701, donor='32606', days={'2': 7264, '3': 8544, '4': 10933, '7': 6960},
        citation=released['citation'], historical_paper_donor=False),
        independent_public_donor13176=dict(path=public['input_npz'], sha256=packaged['input_sha256'],
            cells=35384, donor='13176', days=public['day_counts'], citation=public['data_citation'],
            historical_paper_donor=True, historical_artifact_identity_certified=False))
    jobs = contract.scientific_cases()
    for job in jobs:
        job.update(data=data[job['dataset']], source_yaml=f'{AUTHOR}/runner/configs/model/{job["method"]}.yaml')
        job['model_config'] = f'configs/models/fm_otcfm_singlecell_native__{job["method"]}.v3.json'
        job['artifact_directory'] = 'F:/aicoding/IIA_Data/experiments/otcfm_singlecell_native_full120_v3/' + hashlib.sha256(job['id'].encode()).hexdigest()[:20]
    paths = [MODULE.replace('.', '/') + '.py', 'scripts/flow_matching/otcfm_singlecell_native_contract_v1.py',
        'tests/test_fm_otcfm_singlecell_native_contract_v1.py',
        'scripts/flow_matching/otcfm_empty_ode_args_compat_v1.py',
        'tests/test_fm_otcfm_empty_ode_args_compat_v1.py',
        'scripts/flow_matching/preflight_otcfm_singlecell_native_v3.py',
        'tests/test_fm_otcfm_singlecell_native_preflight_v3.py', ENVIRONMENT,
        'configs/acquisition/fm_otcfm_author_singlecell_mendeley.v1.json', public_path,
        public['output_directory'] + '/input_packaging_report.json',
        'papers/literature/flow_matching/pdfs/ot_cfm.pdf', 'papers/literature/flow_matching/extracted/ot_cfm.txt',
        *[f'{AUTHOR}/runner/configs/model/{name}.yaml' for name in contract.METHODS],
        *[f'configs/models/fm_otcfm_singlecell_native__{name}.v3.json' for name in contract.METHODS],
        f'{AUTHOR}/runner/configs/experiment/cfm.yaml', f'{AUTHOR}/runner/configs/callbacks/default.yaml',
        *[p.relative_to(ROOT).as_posix() for p in (ROOT / AUTHOR / 'runner/src').rglob('*.py')]]
    return dict(schema_version=1, jobs=jobs, required_models=120, methods=list(contract.METHODS),
        datasets=list(contract.DATASETS), protocol_tracks=list(contract.PROTOCOLS), seeds=list(contract.SEEDS),
        leftout_timepoints=list(contract.LEAVEOUTS), environment_lock=ENVIRONMENT, python=env['python'],
        state_root='experiments/runs/fm_otcfm_singlecell_native_full120_v3',
        resource_lock='experiments/runs/fm_otcfm_singlecell_training_cpu.lock', numerical_threads=2,
        settings=dict(minimum_available_memory_bytes=16 * 2**30, minimum_commit_headroom_bytes=32 * 2**30,
            emergency_commit_headroom_bytes=16 * 2**30, maximum_worker_rss_bytes=6 * 2**30,
            maximum_worker_private_bytes=8 * 2**30),
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(paths))],
        citation='https://arxiv.org/abs/2302.00482', paper_sections=['E.2', 'E.5'], paper_target_table=4,
        primary_metric='test/t_out/1-Wasserstein',
        source_compatibility='Fractional random_split is converted to the PyTorch1.13+ allocation before the unchanged Torch1.12 integer splitter; original generator42 retained. Only empty Torchdyn auxiliary args are adapted on the original FlowSolver drift and VelocityNet forward methods; same values/gradients/ODE formula, nonempty args rejected.',
        boundary='All120 full available-data/source-method/recipe/leftout/seed conditions; original1000-epoch maxima and both complete stopping recipes. '
            'Original author constructors, losses, optimizer, solver and all full test timepoint arrays execute unchanged. '
            'Paper RK45(rk4) wording is ambiguous; literal rk4 is a disclosed separate recipe. '
            'Released donor32606 is not paper donor13176; independent donor13176 input is not a certified historical artifact. '
            'Validation loss includes leftout-timepoint validation observations and whitening fits all selected timepoints, as release code does. '
            'Five training seeds remain separate from two heldout times. EB/CITE/DSB/TrajectoryNet/other original branches and full45-paper scope remain required.',
        original_scope=public['original_scope'], all_original_experiments_complete=False)


def verify(queue):
    if queue != registration(): raise ValueError('Unchanged complete120-case native source/data/protocol/environment contract required')


def source_runtime(queue):
    import importlib.metadata
    import platform
    locked = read(ROOT / queue['environment_lock'])
    if (platform.python_version() != locked['python_version']
            or {name: importlib.metadata.version(name) for name in locked['packages']} != locked['packages']):
        raise ValueError('The actual isolated author Python and complete package lock must remain exact')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; os.environ['WANDB_MODE'] = 'disabled'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT / AUTHOR / 'runner')); sys.path.insert(1, str(ROOT / AUTHOR))
    import torch
    import pytorch_lightning as pl
    import src.datamodules.distribution_datamodule as distribution
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.models.components.solver import FlowSolver
    from src.models.components.simple_mlp import VelocityNet
    from scripts.flow_matching.otcfm_empty_ode_args_compat_v1 import install
    install(FlowSolver, 'forward_ode_drift'); install(VelocityNet, 'forward')
    torch.set_num_threads(queue['numerical_threads'])
    original_split = getattr(distribution, '_iia_original_random_split', distribution.random_split)
    distribution._iia_original_random_split = original_split
    distribution.random_split = lambda dataset, lengths, generator=None: contract.compatible_random_split(
        dataset, lengths, generator, original_split)
    return torch, pl, distribution, instantiate, OmegaConf


def instantiate_case(queue, job):
    torch, pl, distribution, instantiate, OmegaConf = source_runtime(queue)
    pl.seed_everything(job['seed'], workers=True)
    recipe = job['scientific_protocol']
    dm = distribution.CustomTrajectoryDataModule(system=str(ROOT / job['data']['path']),
        train_val_test_split=tuple(recipe['data_split_fractions']), max_dim=5, whiten=True,
        batch_size=recipe['batch_size'], num_workers=0, pin_memory=False, seed=job['seed'])
    counts = {str(level): len(data) for level, data in zip(dm.ulabels, dm.timepoint_data)}
    if dm.dim != 5 or sum(counts.values()) != job['data']['cells'] or counts != job['data']['days']:
        raise ValueError('Every actual original cell/PC/timepoint must enter the native author data module')
    cfg = OmegaConf.load(ROOT / job['source_yaml'])
    if (cfg.sigma_min != job['sigma_min'] or cfg.ot_sampler != job['ot_sampler']
            or list(cfg.net.hidden_dims) != job['net']['hidden_dims']
            or cfg.net.activation != job['net']['activation'] or cfg.net.batch_norm != job['net']['batch_norm']
            or cfg.optimizer.lr != recipe['learning_rate'] or cfg.optimizer.weight_decay != recipe['weight_decay']):
        raise ValueError('Every frozen native architecture, coupling, noise and optimizer setting must agree')
    cfg.leaveout_timepoint = job['leaveout_timepoint']; cfg.partial_solver.ode_solver = recipe['ode_solver']
    model = instantiate(cfg)(datamodule=dm)
    if sum(p.numel() for p in model.parameters()) != job['expected_model_parameters']:
        raise ValueError('Complete original three64-unit SELU velocity network required')
    return torch, pl, dm, model


def worker(queue, job, probe=False):
    if sha(ROOT / job['data']['path']) != job['data']['sha256']:
        raise ValueError('Complete original input changed')
    torch, pl, dm, model = instantiate_case(queue, job)
    out = Path(job['artifact_directory'])
    if probe:
        batch = next(iter(dm.train_dataloader()))
        reg, mse = model.step(batch, training=True)
        loss = reg + mse; loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Complete native real-data training batch/gradients must be finite')
        validation_batch = next(iter(dm.val_dataloader()))
        with torch.no_grad():
            val_reg, val_mse = model.step(validation_batch, training=False)
            solver = model.partial_solver(model.net, model.dim)
            trajectory = solver.odeint(batch[0], torch.linspace(0, 1, 101))
            solver.augmentations = model.val_augmentations
            augmented_trajectory, regularizers = solver.odeint(batch[0], torch.linspace(0, 1, 101))
            from torchdyn.core import NeuralODE
            archival_trajectory = NeuralODE(model.net).trajectory(batch[0], torch.linspace(0, 3, 60))
        if (not torch.allclose(trajectory, augmented_trajectory, rtol=2e-6, atol=2e-6)
                or not torch.isfinite(regularizers).all() or not torch.isfinite(archival_trajectory).all()):
            raise ValueError('Actual source unaugmented/augmented/archive ODE paths must all pass with the unchanged vector field')
        if (not torch.isfinite(val_reg + val_mse) or trajectory.shape != (101, len(batch[0]), 5)
                or not torch.isfinite(trajectory).all()):
            raise ValueError('Native validation batch and complete101-point source ODE integration must work')
        optimizer = model.configure_optimizers()
        recipe = job['scientific_protocol']
        if (type(optimizer) is not torch.optim.AdamW or
                any(group['lr'] != recipe['learning_rate'] or group['weight_decay'] != recipe['weight_decay']
                    for group in optimizer.param_groups)):
            raise ValueError('Actual original optimizer and all parameter groups must agree')
        observation = dict(preflight_only=True, id=job['id'], full_data_cells=len(dm.data),
            parameters=sum(p.numel() for p in model.parameters()), native_batch=job['scientific_protocol']['batch_size'],
            timepoint_counts=job['data']['days'], actual_train_batches=len(dm.train_dataloader()),
            actual_validation_batches=len(dm.val_dataloader()),
            split_lengths=[[len(part) for part in parts] for parts in dm.split_timepoint_data],
            native_training_gradients_finite=True, native_validation_loss_finite=True,
            native_ode_trajectory_shape=list(trajectory.shape), native_augmented_solver_passed=True,
            native_archival_ode_passed=True, empty_ode_args_ABI_adapter=True,
            augmented_ODE_maximum_absolute_difference=float((trajectory - augmented_trajectory).abs().max()),
            augmented_ODE_check_tolerance=dict(rtol=2e-6, atol=2e-6), native_ode_solver=recipe['ode_solver'],
            full_original_test_loader_shapes=[list(x.shape) for x in next(iter(dm.test_dataloader()))],
            model_optimizer_updates=0, benchmark_performance_claim=False)
        print(json.dumps(observation), flush=True)
        return observation
    out.mkdir(parents=True, exist_ok=False); os.chdir(out)
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    recipe = job['scientific_protocol']; epochs = []; validation_history = []
    class Progress(Callback):
        def on_validation_end(self, trainer, module):
            if not trainer.sanity_checking and 'val/loss' in trainer.callback_metrics:
                value = float(trainer.callback_metrics['val/loss'].detach().cpu())
                finite = __import__('math').isfinite(value)
                validation_history.append(dict(epoch=int(trainer.current_epoch) + 1,
                    finite=finite, validation_loss=value if finite else None))
        def on_train_epoch_end(self, trainer, module):
            epochs.append(dict(epoch=int(trainer.current_epoch) + 1, optimizer_updates=int(trainer.global_step)))
            write(out / 'progress.json', dict(id=job['id'], phase='training', max_epochs=recipe['max_epochs'],
                completed_epoch=epochs[-1]['epoch'], optimizer_updates=int(trainer.global_step),
                maximum_training_wall_seconds=recipe['maximum_training_wall_seconds'], all_original_experiments_complete=False))
    stopping = EarlyStopping(monitor=recipe['early_stopping_monitor'], patience=recipe['early_stopping_patience_validation_checks'],
        mode='min', min_delta=recipe['early_stopping_min_delta'], check_finite=True, check_on_train_epoch_end=False)
    # Retain every validation checkpoint. Best-weight selection stays min val/loss;
    # this changes storage retention only, not optimization or checkpoint choice.
    checkpoint = ModelCheckpoint(dirpath=out / 'checkpoints', filename='epoch_{epoch:04d}',
        monitor='val/loss', mode='min', save_last=True, save_top_k=-1, auto_insert_metric_name=False)
    trainer = pl.Trainer(default_root_dir=out, min_epochs=1, max_epochs=recipe['max_epochs'],
        check_val_every_n_epoch=recipe['validation_every_n_epochs'], accelerator='cpu', devices=1,
        precision=32, deterministic=False, logger=CSVLogger(str(out), name='csv'),
        callbacks=[checkpoint, stopping, Progress()], enable_progress_bar=False,
        max_time=timedelta(seconds=recipe['maximum_training_wall_seconds']) if recipe['maximum_training_wall_seconds'] else None)
    started = time.perf_counter(); trainer.fit(model, datamodule=dm)
    trained_seconds = time.perf_counter() - started
    if not epochs or not checkpoint.best_model_path:
        raise ValueError('Actual complete native training and validation-selected weights required')
    reason = contract.terminal_reason(recipe, len(epochs), int(stopping.stopped_epoch), trained_seconds,
        getattr(trainer, 'interrupted', False),
        checkpoint.best_model_score is not None and bool(torch.isfinite(checkpoint.best_model_score)))
    write(out / 'progress.json', dict(id=job['id'], phase='full_original_test', epochs=len(epochs),
        optimizer_updates=int(trainer.global_step), best_checkpoint=checkpoint.best_model_path))
    test = trainer.test(model=model, datamodule=dm, ckpt_path=checkpoint.best_model_path)
    if len(test) != 1 or queue['primary_metric'] not in test[0] or not all(__import__('math').isfinite(v) for v in test[0].values()):
        raise ValueError('All original full-distribution test metrics must complete and remain finite')
    if torch.cuda.is_initialized(): raise ValueError('This original single-cell CPU lane must not initialize CUDA')
    artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in {'result.json', 'progress.json'}]
    result = dict(status='completed', job=job, job_sha256=identity(job), original_source_native_execution=True,
        maximum_epochs=recipe['max_epochs'], actual_completed_epochs=len(epochs), optimizer_updates=int(trainer.global_step),
        batch_size=recipe['batch_size'], training_seconds=trained_seconds, total_seconds=time.perf_counter() - started,
        full_timepoint_cell_counts=job['data']['days'], original_loader_whiten_true=True,
        best_checkpoint=str(checkpoint.best_model_path), best_checkpoint_sha256=sha(checkpoint.best_model_path),
        early_stopping_epoch=int(stopping.stopped_epoch), early_stopping_patience_checks=int(stopping.patience),
        epoch_history=epochs, validation_history=validation_history,
        nonfinite_validation_observed=any(not row['finite'] for row in validation_history),
        termination_reason=reason,
        test_metrics=test[0], primary_heldout_W1=test[0][queue['primary_metric']],
        parameter_count=sum(p.numel() for p in model.parameters()), artifacts=artifacts,
        environment_lock_sha256=sha(ROOT / ENVIRONMENT), queue_sha256=sha(ROOT / QUEUE),
        historical_paper_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False, boundary=queue['boundary'])
    exclusive(out / 'result.json', result)


def controller(queue):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / queue['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('All120 native single-cell conditions already owned')
    done = []; failed = {}; state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(values):
        state.update(values, completed_ids=done, failed_preserved=failed, required_original_native_cases=120,
            captured_utc=datetime.now(timezone.utc).isoformat(), all_original_experiments_complete=False)
        write(base / 'status.json', state)
    try:
        for job in queue['jobs']:
            key = Path(job['artifact_directory']).name; attempt = base / 'attempts' / key
            try:
                if attempt.exists() or Path(job['artifact_directory']).exists():
                    raise FileExistsError('Preserve every native full or partial experiment; no automatic retry')
                with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
                    attempt.mkdir(parents=True, exist_ok=False)
                    command = [queue['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--job-id', job['id']]
                    with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg', WANDB_MODE='disabled'),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                        update(dict(state='running_full_native_singlecell_case', active_pid=child.pid, exact_child_handle=handle, active_job=job['id']))
                        code, usage = capped_wait(child, queue, api, update)
                    result_path = Path(job['artifact_directory']) / 'result.json'
                    passed = code == 0 and not usage['resource_guard_aborted'] and result_path.exists()
                    if passed:
                        result = read(result_path)
                        passed = result['status'] == 'completed' and result['job_sha256'] == identity(job) and result['job'] == job
                    exclusive(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, exact_child_handle=handle,
                        **usage, result_sha256=sha(result_path) if result_path.exists() else None))
                    if not passed: raise ValueError('Preserve failed full native source execution')
                    done.append(job['id'])
            except Exception as error: failed[job['id']] = str(error)
            update(dict(state='between_complete_original_cases', active_pid=None))
        update(dict(state='all120_native_case_attempts_terminal_other_original_experiments_remain', active_pid=None))
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--job-id'); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.register: exclusive(ROOT / QUEUE, registration()); return
    queue = read(ROOT / QUEUE); verify(queue)
    if args.job_id: worker(queue, next(j for j in queue['jobs'] if j['id'] == args.job_id), probe=args.preflight)
    else: controller(queue)


if __name__ == '__main__': main()
