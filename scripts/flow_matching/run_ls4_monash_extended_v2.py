"""Run complete Solar/Temperature LS4 source budgets with multi-batch evaluators."""
import argparse
from datetime import datetime, timezone
import functools
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write, execute_original_entry

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def derive_config(template, config, data_path):
    import copy
    value=copy.deepcopy(template)
    value['data']['path']=str(data_path)
    dataset=config['dataset']
    if config['protocol']=='paper_literal':
        value['optim'].update(epochs=7000 if dataset=='solar_weekly' else 1000,batch_size=64,lamb=.999)
        value['model']['sigma']=.1
        value['model']['z_dim']=5
        value['model']['decoder']['prior'].update(d_input=5,d_output=5)
        value['model']['decoder']['decoder']['d_input']=5
        value['model']['encoder']['posterior']['d_output']=5
    elif config['protocol']!='released':
        raise ValueError('Unknown full Monash protocol')
    return value


def budget(config):
    batches=math.ceil(config['train_samples']/config['optim']['batch_size'])
    epochs=config['optim']['epochs'];test=config['test_samples']
    periodic=[(e+1)*batches for e in range(epochs) if e>0 and e%config['optim']['metric_iter']==0]
    classifier_train=int(2*test*.8);classifier_test=2*test-classifier_train
    return dict(batches=batches,generator_updates=epochs*batches,
        metric_update_positions=periodic+[epochs*batches],metric_calls=len(periodic)+1,
        evaluator_epochs=100,classifier_updates_per_metric=100*math.ceil(classifier_train/128),
        predictor_updates_per_metric=100*math.ceil(test/128),
        classifier_test_batches=math.ceil(classifier_test/128),predictor_test_batches=math.ceil(test/128))


def verify(queue):
    for item in queue['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen full Monash source differs: '+item['path'])
    identities=set()
    for job in queue['jobs']:
        config=read(ROOT/job['model_config'])
        if sha(ROOT/job['model_config'])!=job['model_config_sha256']:raise ValueError('Frozen full model differs')
        key=(config['dataset'],config['protocol'],config['seed'])
        if key in identities:raise ValueError('Duplicate original seed slot')
        identities.add(key)
        dataset=config['dataset'];released=config['protocol']=='released'
        expected_samples,expected_length=(137,52) if dataset=='solar_weekly' else (32072,725)
        expected_epochs=(100000 if released else 7000) if dataset=='solar_weekly' else 1000
        if (config['samples']!=expected_samples or config['length']!=expected_length or
            config['optim']['epochs']!=expected_epochs or config['optim']['batch_size']!=(128 if released else 64)
            or config['sigma']!=.1 or config['diagnostic']):raise ValueError('Unabridged original Monash scope/budget differs')
        if config['model']['z_dim']!=(10 if released and dataset=='temperature_rain' else 5):
            raise ValueError('Original/paper-literal latent dimensions differ')
        if config['optim']['lamb']!=(.99 if released and dataset=='solar_weekly' else .999):
            raise ValueError('Original/paper-literal EMA differs')
    expected={(d,p,s) for d in ('solar_weekly','temperature_rain') for p in ('released','paper_literal') for s in queue['seeds']}
    if identities!=expected or len(identities)!=20:raise ValueError('All20 full original Monash slots required')


def workspace(queue, artifact):
    artifact = Path(artifact)
    if artifact.exists():
        raise FileExistsError('Preserve previous LS4 raw/workspace/model attempt')
    target = artifact / 'workspace'
    shutil.copytree(ROOT / queue['source_root'], target,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for item in queue['source_receipts']:
        source = Path(item['path'])
        if source.is_relative_to(queue['source_root']):
            if sha(target / source.relative_to(queue['source_root'])) != item['sha256']:
                raise ValueError('Private author source copy differs')
    for data in queue['datasets'].values():
        destination = target / 'author_data' / data['dataset'] / data['filename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / data['raw_path'], destination)
        if sha(destination) != data['raw_sha256']:
            raise ValueError('Private complete TSF copy differs')
    return target


def prepare_config(queue, config, target):
    from omegaconf import OmegaConf
    template=OmegaConf.to_container(OmegaConf.load(ROOT/config['source_config']),resolve=True)
    value=derive_config(template,config,target/'author_data'/config['dataset'])
    if value['optim']!=config['optim'] or value['model']!=config['model']:raise ValueError('Full derived original hyperparameters differ')
    resolved=target/'resolved_original.yaml';OmegaConf.save(OmegaConf.create(value),resolved)
    return resolved,OmegaConf.create(value)


def source_imports(target):
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(target)); os.chdir(target)
    import torch
    import numpy as np
    import datasets
    return torch, np, datasets


def checked_data(config, datasets, torch, np, args):
    """Observe original full trajectory preprocessing and row split without advancing RNG."""
    generator = torch.Generator(); generator.set_state(torch.get_rng_state())
    ids = torch.randperm(config['samples'], generator=generator)
    original = datasets.parse_datasets(args, batch_size=config['optim']['batch_size'], device=torch.device('cpu'))
    raw = datasets.Monash(args.path, args.dataset)
    raw_array = np.stack([d.to_numpy()[:, None] for d in raw.full_series_list])
    if list(raw_array.shape) != [config['samples'], config['length'], 1] or not np.isfinite(raw_array).all():
        raise ValueError('Original complete LS4 source data shape differs')
    mean = torch.tensor(raw_array.mean(1, keepdims=True), dtype=torch.float32)
    std = torch.tensor(raw_array.std(1, keepdims=True), dtype=torch.float32); std[std == 0] = 1
    if args.preproc=='normalize_per_seq':
        expected=(torch.tensor(raw_array,dtype=torch.float32)-mean)/std
    elif args.preproc=='squash_per_seq':
        low=torch.tensor(raw_array.min(1,keepdims=True),dtype=torch.float32)
        span=torch.tensor(raw_array.max(1,keepdims=True),dtype=torch.float32)-low;span[span==0]=1
        expected=(torch.tensor(raw_array,dtype=torch.float32)-low)/span
    else:raise ValueError('Unsupported original preprocessing')
    actual = original['dataset_obj']
    if not torch.equal(actual, expected):
        raise ValueError('Full per-sequence normalization differs')
    count = config['train_samples']
    train, test = original['train_dataloader'], original['test_dataloader']
    if not torch.equal(train.dataset.tensors[0], expected[ids[:count]]) or not torch.equal(test.dataset.tensors[0], expected[ids[count:]]):
        raise ValueError('Original whole-trajectory split differs')
    if len(set(ids.tolist())) != config['samples'] or set(ids[:count].tolist()) & set(ids[count:].tolist()):
        raise ValueError('Full trajectory split is not disjoint')
    for loader in (train, test):
        if loader.num_workers != 0 or loader.drop_last or loader.batch_size != config['optim']['batch_size']:
            raise ValueError('Original Monash loader cardinality/worker count differs')
        if not bool(loader.dataset.tensors[1].all()):
            raise ValueError('Full original masks differ')
    record = dict(dataset=config['dataset'], seed=config['seed'], raw_shape=list(raw_array.shape),
        train_ids=ids[:count].tolist(), test_ids=ids[count:].tolist(), train_samples=len(train.dataset),
        test_samples=len(test.dataset), batch_size=train.batch_size, batches=len(train),
        last_train_batch=config['train_samples'] % train.batch_size or train.batch_size,
        num_workers=train.num_workers, full_original_normalization_bitwise=True,preprocessing=str(args.preproc),trajectory_split_disjoint=True)
    return original, record



def audit_data(queue):
    target = workspace(queue, queue['data_audit_artifact'])
    torch, np, datasets = source_imports(target)
    torch.set_num_threads(queue['settings']['cpu_threads'])
    records = []
    for job in queue['jobs']:
        config = read(ROOT / job['model_config'])
        _, resolved = prepare_config(queue, config, target)
        torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
        _, record = checked_data(config, datasets, torch, np, resolved.data)
        records.append(dict(record,id=job['id'],protocol=config['protocol']))
        write(ROOT/queue['data_audit_progress'],dict(queue_sha256=sha(ROOT/queue['queue_path']),cases_completed=len(records),required_cases=20,records=records))
        del _
        import gc
        gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError('CPU LS4 data audit initialized a CUDA context')
    proof = dict(passed=True, diagnostic_only=True, queue_sha256=sha(ROOT / queue['queue_path']),
        cases=len(records), records=records, cuda_context_initialized=False,
        normalization_boundary='Author normalizes each entire trajectory before80/20 row split; no strict TSAD inference',
        captured_utc=datetime.now(timezone.utc).isoformat())
    write(ROOT / queue['data_audit_output'], proof)


def original_module(target):
    spec = importlib.util.spec_from_file_location('ls4_author_diagnostic', target / 'train_monash.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def gpu_preflight(queue):
    target = workspace(queue, queue['gpu_preflight_artifact'])
    torch, np, datasets = source_imports(target)
    torch.set_num_threads(queue['settings']['cpu_threads'])
    if not torch.cuda.is_available():
        raise ValueError('Full LS4 CUDA preflight required')
    import wandb
    wandb.init(mode='disabled')
    module = original_module(target)
    cases = []
    for job in queue['jobs']:
        config = read(ROOT / job['model_config'])
        if config['seed'] != queue['seeds'][0]:
            continue
        _, resolved = prepare_config(queue, config, target)
        torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
        data, record = checked_data(config, datasets, torch, np, resolved.data)
        resolved.model.n_labels = 1
        model = module.VAE(resolved.model).cuda()
        optimizer, scheduler = module.setup_optimizer(model, **{k:resolved.optim[k] for k in ('lr','weight_decay','epochs')})
        model = torch.nn.DataParallel(model)
        # One complete original epoch is a capacity/interface diagnostic only.
        steps = module.train(model, None, 200, 'cuda', data['train_dataloader'], optimizer, scheduler, 0)
        metrics = module.compute_all_metrics(model, data['test_dataloader'], module.setup_optimizer, 'cuda', torch.nn.Sigmoid() if config['dataset']=='temperature_rain' else torch.nn.Identity())
        if steps != budget(config)['batches'] or not all(math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError('Full original LS4 CUDA interface/evaluator failed')
        cases.append(dict(id=job['id'], dataset=config['dataset'], protocol=config['protocol'],
            train_samples=record['train_samples'], test_samples=record['test_samples'], length=config['length'],
            batch_size=record['batch_size'], full_epoch_updates=steps,
            original_classifier_epochs=100, original_predictor_epochs=100, complete_metric_interface_passed=True))
        del model, optimizer, scheduler, data
        import gc
        gc.collect(); torch.cuda.empty_cache()
    if len(cases) != 4:
        raise ValueError('All4 full original LS4 CUDA cases required')
    write(ROOT / queue['gpu_preflight_output'], dict(passed=True, diagnostic_only=True,
        queue_sha256=sha(ROOT / queue['queue_path']), cases=cases,
        source_models_and_evaluators_unchanged=True, generated_from_full_test_trajectories=True,
        captured_utc=datetime.now(timezone.utc).isoformat()))


def validate_result(result, config):
    required=budget(config);epochs=config['optim']['epochs']
    if result['generator_updates']!=required['generator_updates']:raise ValueError('Full generator budget incomplete')
    if result['epoch_updates']!={str(e):required['batches'] for e in range(epochs)}:raise ValueError('Full original epoch coverage incomplete')
    if result['epoch_samples']!={str(e):config['train_samples'] for e in range(epochs)}:raise ValueError('Full original trajectory coverage incomplete')
    if [r['generator_updates'] for r in result['metric_history']]!=required['metric_update_positions']:raise ValueError('Full periodic/final metric calls incomplete')
    for record in result['metric_history']:
        if (record['classifier_updates']!=required['classifier_updates_per_metric'] or
            record['predictor_updates']!=required['predictor_updates_per_metric']):raise ValueError('Full100-epoch multi-batch evaluators incomplete')
        if record['reference_shape']!=[config['test_samples'],config['length'],1] or record['generated_shape']!=record['reference_shape']:
            raise ValueError('Full evaluation trajectory/length coverage incomplete')
        if not all(math.isfinite(record['metrics'][k]) for k in ('clf_score','marginal_score','predictive_score')):
            raise ValueError('Nonfinite full Monash evaluator')
    count=sum(e>0 and e%config['optim']['eval_iter']==0 for e in range(epochs))+1
    if len(result['reconstruction_history'])!=count:raise ValueError('Full original reconstruction calls incomplete')


def run_job(queue, job):
    config = read(ROOT / job['model_config'])
    proof = read(ROOT / queue['gpu_preflight_output'])
    if not proof['passed'] or proof['queue_sha256'] != sha(ROOT / queue['queue_path']):
        raise ValueError('Full original LS4 CUDA proof missing or changed')
    artifact = Path(job['artifact_directory']); target = workspace(queue, artifact)
    resolved_path, _ = prepare_config(queue, config, target)
    torch, np, datasets = source_imports(target)
    torch.set_num_threads(queue['settings']['cpu_threads'])
    if not torch.cuda.is_available():
        raise ValueError('Original full LS4 training requires CUDA')
    import metrics
    entry = (target / 'train_monash.py').resolve()
    out = ROOT / job['output_directory']; out.mkdir(parents=True, exist_ok=False)
    required = budget(config)
    state = dict(id=job['id'], status='running', diagnostic_only=False, source_entry_sha256=sha(entry),
        queue_sha256=sha(ROOT / queue['queue_path']), model_config_sha256=sha(ROOT / job['model_config']),
        generator_updates=0, epoch_updates={}, epoch_samples={}, metric_history=[], reconstruction_history=[],
        full_original_budget=True, protocol=config['protocol'], seed=config['seed'],
        author_equivalence_certified=False, formal_strict_tsad_result=False)
    originals = []; holder = {}; active = {}; namespace = {}
    def persist():
        write(out / 'progress.json', state)
    def patch(module, name, function):
        originals.append((module, name, getattr(module, name))); setattr(module, name, function)
    def source_frame():
        frame = inspect.currentframe().f_back
        while frame:
            filename = Path(frame.f_code.co_filename).resolve()
            if filename == entry or filename == (target / 'metrics.py').resolve():
                return frame
            frame = frame.f_back
        return None
    old_parse = datasets.parse_datasets
    def observed_parse(args, batch_size, device, main_args=None):
        # checked_data calls the saved original parser to avoid recursive wrapping.
        datasets.parse_datasets = old_parse
        try:
            data, record = checked_data(config, datasets, torch, np, args)
        finally:
            datasets.parse_datasets = observed_parse
        state['data_record'] = record; persist(); return data
    patch(datasets, 'parse_datasets', observed_parse)
    old_init = torch.optim.AdamW.__init__
    @functools.wraps(old_init)
    def observed_init(optimizer, *args, **kwargs):
        value = old_init(optimizer, *args, **kwargs)
        frame = source_frame()
        if frame and frame.f_code.co_name == 'setup_optimizer' and 'main_frame' not in holder:
            main = frame.f_back
            if main.f_code.co_name != 'main' or Path(main.f_code.co_filename).resolve() != entry:
                raise ValueError('Original generator optimizer caller differs')
            holder['main_frame'] = main; holder['optimizer'] = optimizer
            original_eval = main.f_globals['eval']
            @functools.wraps(original_eval)
            def observed_eval(*a, **kw):
                begin = time.perf_counter(); value = original_eval(*a, **kw)
                state['reconstruction_history'].append(dict(epoch=int(a[2]), mse=float(value),
                    generator_updates=state['generator_updates'], seconds=time.perf_counter()-begin))
                persist(); return value
            main.f_globals['eval'] = observed_eval
        return value
    patch(torch.optim.AdamW, '__init__', observed_init)
    old_step = torch.optim.AdamW.step
    @functools.wraps(old_step)
    def observed_step(optimizer, *a, **kw):
        frame = source_frame(); value = old_step(optimizer, *a, **kw)
        if frame and Path(frame.f_code.co_filename).resolve() == entry and frame.f_code.co_name == 'train':
            epoch = str(int(frame.f_back.f_locals['epoch']))
            state['generator_updates'] += 1
            state['epoch_updates'][epoch] = state['epoch_updates'].get(epoch,0)+1
            state['epoch_samples'][epoch] = state['epoch_samples'].get(epoch,0)+len(frame.f_locals['data'])
            if not bool(torch.isfinite(frame.f_locals['loss']).all()):
                raise ValueError('Nonfinite original LS4 objective')
            if state['generator_updates'] % 250 == 0:
                persist()
        elif frame and frame.f_code.co_name in ('compute_classification_score', 'compute_predictive_score'):
            name = 'classifier_updates' if frame.f_code.co_name == 'compute_classification_score' else 'predictor_updates'
            active[name] = active.get(name,0)+1
            holder[name+'_frame'] = frame
        return value
    patch(torch.optim.AdamW, 'step', observed_step)
    old_classify = metrics.compute_classification_score
    @functools.wraps(old_classify)
    def observed_classify(fake, real, *a, **kw):
        path = artifact / ('evaluation_%03d.npz' % len(state['metric_history']))
        fake_array = fake.detach().cpu().numpy(); real_array = real.detach().cpu().numpy()
        if not np.isfinite(fake_array).all() or not np.isfinite(real_array).all():
            raise ValueError('Nonfinite complete LS4 evaluation arrays')
        np.savez_compressed(path, generated=fake_array, reference=real_array)
        active.update(arrays=str(path), arrays_sha256=sha(path), generated_shape=list(fake_array.shape),
                      reference_shape=list(real_array.shape))
        return old_classify(fake, real, *a, **kw)
    patch(metrics, 'compute_classification_score', observed_classify)
    old_metrics = metrics.compute_all_metrics
    @functools.wraps(old_metrics)
    def observed_metrics(*a, **kw):
        active.clear(); active.update(generator_updates=state['generator_updates'], classifier_updates=0, predictor_updates=0)
        begin = time.perf_counter(); values = old_metrics(*a, **kw)
        active.update(metrics={name:float(value) for name,value in values.items()},seconds=time.perf_counter()-begin,
            metric_boundary='Original test losses SUM batch means; preserve multi-batch values, not paper-equivalent averages',
            classifier_test_batches=required['classifier_test_batches'],predictor_test_batches=required['predictor_test_batches'])
        for key in ('classifier_updates','predictor_updates'):
            frame = holder[key+'_frame']; model = frame.f_locals['model']; optimizer = frame.f_locals['optimizer']
            path = artifact / ('evaluation_%03d_%s.pt' % (len(state['metric_history']), key))
            evaluation_state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),kind=key)
            if key=='classifier_updates':evaluation_state['randperm']=frame.f_locals['randperm']
            torch.save(evaluation_state,path)
            active[key+'_checkpoint']=str(path); active[key+'_checkpoint_sha256']=sha(path)
        state['metric_history'].append(dict(active)); persist(); return values
    patch(metrics, 'compute_all_metrics', observed_metrics)
    begin = time.perf_counter(); persist()
    try:
        execute_original_entry(entry, ['--config',str(resolved_path),'--seed',str(config['seed']),
                                      '--log_dir',str(artifact/'author_outputs')], namespace)
        validate_result(state, config)
        local = holder['main_frame'].f_locals
        checkpoint = artifact / 'final_full_checkpoint.pt'
        torch.save(dict(model=local['model'].state_dict(), ema_model=local['ema_model'].state_dict(),
            optimizer=local['optimizer'].state_dict(), scheduler=local['scheduler'].state_dict(),
            step=local['step'], config=read(ROOT/job['model_config']), full_original_budget=True,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate()), checkpoint)
        state.update(status='completed', seconds=time.perf_counter()-begin, checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint), metrics=state['metric_history'][-1]['metrics'],
            selected_author_checkpoint=str(artifact/'author_outputs'/config['dataset']/config['doc']/'checkpoints/ckpt.pth'),
            selection_boundary='Author saves best test-marginal checkpoint, but final metrics use last EMA/model without reloading it',
            uncertainty_scope='One frozen training seed; final evaluator call is not a between-seed interval',
            completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'result.json', state)
    except BaseException as error:
        state.update(status='failed_or_partial_preserved', error=repr(error), seconds=time.perf_counter()-begin)
        write(out / 'failure.json', state); raise
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def verify_result(queue, job):
    result = read(ROOT / job['output_directory'] / 'result.json')
    validate_result(result, read(ROOT / job['model_config']))
    if result['status'] != 'completed' or result['queue_sha256'] != sha(ROOT / queue['queue_path']) or result['model_config_sha256'] != sha(ROOT / job['model_config']):
        raise ValueError('Frozen complete LS4 result binding differs')
    if sha(result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Full original LS4 trained model differs')
    for record in result['metric_history']:
        for name in ('arrays','classifier_updates_checkpoint','predictor_updates_checkpoint'):
            if sha(record[name]) != record[name+'_sha256']:
                raise ValueError('Full original evaluator artifact differs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True); parser.add_argument('--job-id')
    parser.add_argument('--audit-data',action='store_true'); parser.add_argument('--gpu-preflight',action='store_true')
    parser.add_argument('--verify-result',action='store_true')
    args=parser.parse_args(); queue=read(ROOT/args.queue); verify(queue)
    if args.audit_data:
        audit_data(queue)
    elif args.gpu_preflight:
        gpu_preflight(queue)
    else:
        job=next(j for j in queue['jobs'] if j['id']==args.job_id)
        if args.verify_result:
            verify_result(queue,job)
        else:
            run_job(queue,job)


if __name__=='__main__':
    main()
