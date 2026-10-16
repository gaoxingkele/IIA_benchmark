"""Run all90 original-code-referenced EB100 CFM recipe/leaveout/five-seed conditions."""
import argparse
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
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as original
from scripts.flow_matching import otcfm_singlecell_native_contract_v1 as contract

MODULE = 'scripts.flow_matching.run_otcfm_EB100_code_referenced_native90_v1'
QUEUE = 'configs/experiments/fm_otcfm_EB100_code_referenced_native_full90.v1.json'
DATA = 'configs/data/fm_otcfm_EB100_code_referenced_native.v1.json'
ENVIRONMENT = original.ENVIRONMENT
identity = original.identity
def instantiate_case(queue,job):
    from scripts.flow_matching.otcfm_validation_batch_compat_v1 import install
    torch,pl,dm,model=original.instantiate_case(queue,job)
    actual=install(dm,job['validation_compatibility']['effective_validation_batch'])
    if actual!=job['validation_compatibility']:raise ValueError('Exact validation-only repair required')
    return torch,pl,dm,model


def scientific_cases():
    jobs = []
    for dataset, leaveouts in [('trajectorynet_EB100_code_referenced', (1,2,3))]:
        for recipe in contract.PROTOCOLS:
            for method in contract.METHODS:
                for left in leaveouts:
                    for seed in contract.SEEDS:
                        jobs.append(dict(id=f'otcfm_singlecell__{recipe}__{method}__{dataset}__leave{left}__seed{seed}',
                            dataset=dataset, protocol=recipe, method=method, leaveout_timepoint=left, seed=seed,
                            scientific_protocol=contract.protocol(recipe), net=dict(hidden_dims=[64,64,64],batch_norm=False,activation='selu'),
                            sigma_min=1. if method=='sbcfm' else .1,
                            ot_sampler={'cfm':None,'otcfm':'exact','sbcfm':'sinkhorn'}[method],
                            expected_model_parameters=9093, native_dimensions=5,
                            author_equivalence_certified=False, all_original_experiments_complete=False))
    return jobs


def preflight_cases(queue):
    selected = [j for j in queue['jobs'] if j['seed']==42]
    expected = {(j['dataset'],j['protocol'],j['method'],j['leaveout_timepoint']) for j in scientific_cases()}
    actual = [(j['dataset'],j['protocol'],j['method'],j['leaveout_timepoint']) for j in selected]
    if len(selected)!=18 or set(actual)!=expected:
        raise ValueError('Every original EB100 method/recipe/leaveout combination, including leave3, is required')
    return selected


def registration():
    parent=read(ROOT/original.QUEUE);original.verify(parent)
    data=read(ROOT/DATA);report=read(Path(data['output_directory'])/'native_input_audit.json')
    if report['status']!='completed' or not report['full_raw_data_preserved']:
        raise ValueError('Actual complete original-code-referenced EB100 input audit required')
    inputs={r['id']:r for r in report['records']};jobs=scientific_cases()
    for job in jobs:
        r=inputs[job['dataset']]
        if job['leaveout_timepoint'] not in r['required_intermediate_leaveout_indices']:
            raise ValueError('Keep all original intermediate timepoints')
        minimum_val=min(contract.fraction_lengths(n,[.8,.1,.1])[1] for n in r['model_timepoint_counts'].values())
        train_batch=job['scientific_protocol']['batch_size']
        effective=min(train_batch,minimum_val)
        job['validation_compatibility']=dict(original_training_batch=train_batch,effective_validation_batch=effective,
            minimum_available_validation_samples=minimum_val,training_and_test_loaders_changed=False,
            validation_only_empty_loader_repair=effective<train_batch)
        job.update(data=dict(path=r['path'],sha256=r['sha256'],cells=r['cells'],days=r['model_timepoint_counts'],
            citation=data['data_citation'],full_original_PC_dimensions=r['PC_dimensions'],
            original_code_referenced_public_asset_verified=True,historical_paper_identity_certified=False,boundary=r['boundary']),
            source_yaml=f'{original.AUTHOR}/runner/configs/model/{job["method"]}.yaml',
            model_config=f'configs/models/fm_otcfm_EB100_code_referenced_native__{job["method"]}.v1.json',
            artifact_directory='F:/aicoding/IIA_Data/experiments/otcfm_EB100_code_referenced_native_full90_v1/'+hashlib.sha256(job['id'].encode()).hexdigest()[:20])
    paths=[
        'configs/acquisition/fm_TrajectoryNet_EB100_official.v1.json',
        'scripts/flow_matching/acquire_TrajectoryNet_EB100_official_v1.py',
        'scripts/flow_matching/audit_TrajectoryNet_EB100_code_reference_v1.py',
        'tests/test_fm_TrajectoryNet_EB100_official_acquisition_v1.py',
        MODULE.replace('.','/')+'.py','tests/test_fm_otcfm_EB100_code_referenced_native90_v1.py',DATA,
        'scripts/flow_matching/otcfm_validation_batch_compat_v1.py','tests/test_fm_otcfm_validation_batch_compat_v1.py',
        'scripts/flow_matching/package_otcfm_released_EB_CITE_inputs_v1.py','tests/test_fm_otcfm_released_EB_CITE_inputs_v1.py',
        *[f'configs/models/fm_otcfm_EB100_code_referenced_native__{m}.v1.json' for m in contract.METHODS]]
    paths += [r['path'] for r in parent['source_receipts']]
    return dict(jobs=jobs,required_models=90,required_code_referenced_EB_models=90,
        original_multiome_full120_queue=original.QUEUE,original_multiome_full120_queue_sha256=sha(ROOT/original.QUEUE),
        preserved_existing_EB_CITE_full150_queue='configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json',
        methods=list(contract.METHODS),protocol_tracks=list(contract.PROTOCOLS),seeds=list(contract.SEEDS),
        environment_lock=ENVIRONMENT,python=parent['python'],numerical_threads=2,settings=parent['settings'],
        state_root='experiments/runs/fm_otcfm_EB100_code_referenced_native_full90_v1',resource_lock=parent['resource_lock'],
        preflight_lock='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock',
        full_input_packaging_report=str(Path(data['output_directory'])/'native_input_audit.json'),
        full_input_packaging_report_sha256=sha(Path(data['output_directory'])/'native_input_audit.json'),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        primary_metric=parent['primary_metric'],citation=parent['citation'],paper_sections=['E.2','E.5'],paper_target_table=4,
        original_scope=parent['original_scope'],all_original_experiments_complete=False,
        boundary='All90 code-referenced original EB100 full-data conditions, including every leaveout1/2/3, both full stopping recipes and five training seeds. '
            'All16819 cells/100 original PC columns of the verified2020 TrajectoryNet Git blob enter the source loader; native max_dim5 and whitening apply once. No new PCA, package rewrite or raw overwrite. '
            'Original constructors, pair coupling, losses, optimizer, entire test distributions and101-point ODE execute with previously disclosed split/empty-args/SB-pair repairs. '
            'Original code-referenced EB batch512 has238 minimum validation samples, so a readonly validation-only hparams view uses238; original train512/test loaders and all train budgets remain untouched. Paper branch retains128 validation batch. This explicit source repair is not an unchanged released-YAML execution. Full training updates are captured before test restores the selected checkpoint; restored selected updates stay separately recorded. '
            '18 native real-data preflight conditions must pass before training. Existing120 Multiome and150 released EB/CITE cases remain unchanged and required. '
            'Original EB public-file code reference and2020 byte identity are verified; unreported training hyperparameters and historical paper weight identity remain uncertified. CITE donor32606/13176 gap remains in the preserved track. Not strictTAB results.')


def verify(queue):
    if queue!=registration():raise ValueError('Exact full90 code-referenced EB100 source/data/environment/protocol contract required')


def preflight_worker(queue):
    import gc
    results=[]
    for job in preflight_cases(queue):
        observation=worker(queue,job,probe=True)
        observation['validation_compatibility']=job['validation_compatibility']
        results.append(observation);gc.collect()
    exclusive(ROOT/queue['state_root']/'native18_preflight.json',dict(status='passed',records=results,
        model_training_updates=0,benchmark_performance_claim=False,full90_source_queue_sha256=sha(ROOT/QUEUE)))


def run_preflight(queue,api,base,update):
    import psutil
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    stage=base/'preflight';report=base/'native18_preflight.json'
    if stage.exists() or report.exists():raise FileExistsError('Preserve every preflight attempt; no automatic failed retry')
    with api['resource_slot'](ROOT/queue['preflight_lock'],queue['settings'],update):
        stage.mkdir(parents=True,exist_ok=False)
        command=[queue['python'],'-B','-X','utf8','-u','-m',MODULE,'--preflight-all']
        with (stage/'stdout.log').open('xb') as out,(stage/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            update(dict(state='checking_all18_actual_EB100_conditions',phase='preflight',active_pid=child.pid,exact_child_handle=handle))
            code,usage=capped_wait(child,queue,api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and report.exists()
        if passed:
            r=read(report);passed=(r['status']=='passed' and len(r['records'])==18 and r['model_training_updates']==0
                and r['full90_source_queue_sha256']==sha(ROOT/QUEUE))
        exclusive(stage/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
            native18_report_sha256=sha(report) if report.exists() else None))
        if not passed:raise ValueError('Preserve failed original EB100 preflight; full training cannot proceed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--job-id');parser.add_argument('--preflight-all',action='store_true')
    args=parser.parse_args()
    if args.register:exclusive(ROOT/QUEUE,registration());return
    q=read(ROOT/QUEUE);verify(q)
    if args.preflight_all:preflight_worker(q)
    elif args.job_id:worker(q,next(j for j in q['jobs'] if j['id']==args.job_id))
    else:controller(q)

# Full native worker/controller definitions are copied below from the frozen
# Multiome v4 runner, with independent90 scope and corrected scalar accounting.


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
            SB_normalized_pair_assertion_repair=job['method'] == 'sbcfm',
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
    full_training_optimizer_updates = int(trainer.global_step)
    test = trainer.test(model=model, datamodule=dm, ckpt_path=checkpoint.best_model_path)
    if len(test) != 1 or queue['primary_metric'] not in test[0] or not all(__import__('math').isfinite(v) for v in test[0].values()):
        raise ValueError('All original full-distribution test metrics must complete and remain finite')
    if torch.cuda.is_initialized(): raise ValueError('This original single-cell CPU lane must not initialize CUDA')
    artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in {'result.json', 'progress.json'}]
    result = dict(status='completed', job=job, job_sha256=identity(job), original_source_native_execution=True,
        maximum_epochs=recipe['max_epochs'], actual_completed_epochs=len(epochs), optimizer_updates=full_training_optimizer_updates,
        posttest_selected_checkpoint_optimizer_updates=int(trainer.global_step),
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
    if owner is None: raise RuntimeError('All90 native single-cell conditions already owned')
    done = []; failed = {}; state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(values):
        state.update(values, completed_ids=done, failed_preserved=failed, required_original_native_cases=90,
            captured_utc=datetime.now(timezone.utc).isoformat(), all_original_experiments_complete=False)
        write(base / 'status.json', state)
    try:
        update(dict(state='starting_full_EB100_native_queue'))
        run_preflight(queue, api, base, update)
        update(dict(state='all18_preflight_passed_waiting_full_training',phase='training',active_pid=None))
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
        update(dict(state='all90_native_case_attempts_terminal_other_original_experiments_remain', active_pid=None))
    finally: owner.close()


if __name__ == '__main__': main()
