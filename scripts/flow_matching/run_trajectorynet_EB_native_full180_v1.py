"""Execute original EB density/velocity grid; keep unresolved energy cases visible.

Full original 10000-step training, source and corrected checkpoint selections,
all-population native W2 and independently verified matched W1. No smoke scores
are promoted to trained results and no coefficient is selected on test scores.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE = 'scripts.flow_matching.run_trajectorynet_EB_native_full180_v1'
QUEUE = 'configs/experiments/fm_TrajectoryNet_EB_native_full180.v1.json'
PLAN = 'configs/reproducibility/fm_TrajectoryNet_EB_original_grid_full180_plan.v1.json'
AUTHOR = 'configs/reproducibility/fm_TrajectoryNet_author_fig8_full_checkpoint_replay.v1.json'
MODEL = 'configs/models/fm_TrajectoryNet_EB_native_full_grid.v1.json'
CORRECTION = 'src/iia_benchmark/models/trajectorynet_best_checkpoint_v1.py'
HELPER = 'src/iia_benchmark/models/trajectorynet_author_replay_v1.py'


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def jobs_for(plan):
    jobs = []
    # First reproduce the known released density-only recipe, then retain every
    # other reported grid condition; this is fixed before any new test result.
    order = [3, 0, 1, 2, 4, 5, 6, 7, 8]
    for index in order:
        recipe = plan['recipes'][index]
        for leftout in [3, 1, 2]:
            for seed in plan['local_training_seeds']:
                identifier = f'TrajectoryNet_EB100__recipe{index}__leave{leftout}__seed{seed}'
                jobs.append(dict(id=identifier, recipe_index=index, recipe=recipe,
                    leftout_timepoint=leftout, seed=seed,
                    artifact_directory='F:/aicoding/IIA_Data/experiments/TrajectoryNet_EB_native_full180_v1/'
                    + hashlib.sha256(identifier.encode()).hexdigest()[:20]))
    return jobs


def registration():
    plan, author = read(ROOT / PLAN), read(ROOT / AUTHOR)
    jobs = jobs_for(plan)
    assert len(jobs) == 135 and len({j['id'] for j in jobs}) == 135
    paths = [MODULE.replace('.', '/') + '.py', PLAN, AUTHOR, MODEL, CORRECTION, HELPER,
        'configs/models/fm_TrajectoryNet_best_checkpoint_selection.v1.json',
        'tests/test_fm_trajectorynet_full180_native_queue_v1.py',
        'tests/test_fm_trajectorynet_best_checkpoint_v1.py']
    paths += [r['path'] for r in author['source_receipts']]
    return dict(jobs=jobs, available_training_models=135, required_grid_models=180,
        deferred_energy_jobs=[j for j in plan['jobs'] if j['recipe_index'] >= 9],
        deferred_energy_models=45, plan=PLAN, author_replay_configuration=AUTHOR,
        author_replay_report=str(Path(author['artifact_root']) / 'full_author_checkpoint_replay.json'),
        author_replay_guard=author['state_root'] + '/guard_receipt.json',
        original_source_root=author['source_root'], original_data=author['original_data'],
        original_data_sha256=author['original_data_sha256'], full_population_counts=author['full_population_counts'],
        overlay_directory=author['overlay_directory'], material_receipt=author['material_receipt'],
        source_parameters=read(ROOT / author['model'])['source_parameters'],
        python=author['python'], numerical_threads=author['numerical_threads'],
        settings=author['settings'], resource_lock=author['resource_lock'],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(paths))],
        native_probe_recipe_leaveout_cases=27, native_probe_full_population_leftouts=[1, 2, 3],
        actual_optimizer_steps_per_training_model=10000, batch_size=1000, parameters=9621,
        checkpoint_selections=['corrected_validation_minimum', 'source_last_scheduled_validation'],
        primary_metric='past_to_leftout_raw_Euclidean_W1', require_bitwise_full_prediction_replay=True,
        metric_tolerance=dict(rtol=2e-6, atol=2e-6), fixed_test_coefficient_selection=False,
        state_root='experiments/runs/fm_TrajectoryNet_EB_native_full180_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/TrajectoryNet_EB_native_full180_v1',
        original_scope=author['original_scope'], all_original_experiments_complete=False,
        boundary='135 explicit density/velocity conditions have a callable native pipeline;45 energy conditions remain pending Jacobian recipe binding. Five local seeds are not recovered author seeds. Each model executes original10000steps and both checkpoint selections from the same unchanged numerical trajectory. Original training/validation velocity regularization uses every timepoint including leftout; normalization fits alltimepoints. This is source-protocol reproduction,not strictTAB or proof of historical table equivalence. No test-driven coefficient selection or partial-seed/leaveout aggregation.')


def verify(c):
    if c != registration():
        raise ValueError('Every original grid, source, input, seed, budget, and guard binding is required')
    from scripts.flow_matching.replay_TrajectoryNet_author_fig8_full_v1 import verify as verify_author
    verify_author(read(ROOT / AUTHOR))


def author_ready(c):
    guard_path = ROOT / c['author_replay_guard']
    if not guard_path.exists():
        return False
    guard = read(guard_path)
    if not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']:
        raise ValueError('Preserved original checkpoint failure needs isolated repair, not a retry')
    path = Path(c['author_replay_report'])
    report = read(path)
    if (guard['report_sha256'] != sha(path) or not report['passed']
            or report['configuration_sha256'] != sha(ROOT / AUTHOR)
            or report['parameters'] != 9621 or report['full_population_counts'] != c['full_population_counts']
            or not report['all_source_target_float32_coordinates_equal']):
        raise ValueError('Actual full original checkpoint, source metrics and input identity must pass')
    # Historical saved-array coherence is a reported diagnostic, not a fabricated
    # historical environment identity or a condition silently relaxed here.
    return True


def runtime(c, job, output, checkpoint=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, c['overlay_directory'])
    sys.path.insert(0, str(ROOT / c['original_source_root']))
    import torch
    import torchdiffeq
    import numpy as np
    if not Path(torchdiffeq.__file__).resolve().is_relative_to(Path(c['overlay_directory']).resolve()):
        raise ValueError('Require actual original torchdiffeq0.0.1 overlay')
    torch.set_num_threads(c['numerical_threads'])
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    native = importlib.import_module('TrajectoryNet.main')
    helper = import_file('_TrajectoryNet_original_data_helper_grid_v1', HELPER)
    args = native.parser.parse_args([])
    for key, value in c['source_parameters'].items():
        setattr(args, key, value)
    for key, value in job['recipe']['source_flags'].items():
        setattr(args, key, value)
    args.save = str(output)
    args.leaveout_timepoint = job['leftout_timepoint']
    args.data = helper.native_EB_data(ROOT / c['original_source_root'], c['original_data'])
    args.timepoints = args.data.get_unique_times()
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale
    if args.data.get_data().shape != (16819, 5):
        raise ValueError('Full original population and original5PC truncation required')
    counts = {str(t):int(np.sum(args.data.get_times() == t)) for t in args.timepoints}
    if counts != c['full_population_counts']:
        raise ValueError('Original timepoint populations changed')
    if not np.isfinite(args.data.get_data()).all() or not np.isfinite(args.data.get_velocity()).all():
        raise ValueError('Every original coordinate and velocity must be finite')
    regularization_fns, coefficients = native.create_regularization_fns(args)
    if regularization_fns or coefficients:
        raise ValueError('No unresolved energy/Jacobian coefficient may enter density/velocity jobs')
    model = native.build_model_tabular(args, 5, regularization_fns).to('cpu')
    native.set_cnf_options(args, model)
    if sum(p.numel() for p in model.parameters()) != c['parameters']:
        raise ValueError('Original9,621parameter concat-squash/Tanh model required')
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    logger = native.utils.get_logger(str(output / 'author_training.log'), str(ROOT / MODULE.replace('.', '/')).removesuffix('/') + '.py', displaying=False)
    return torch, np, native, args, model, logger


def metrics(c, torch, np, args, model, output):
    from scipy.spatial.distance import cdist
    evaluation = importlib.import_module('TrajectoryNet.eval_utils')
    emd = importlib.import_module('TrajectoryNet.optimal_transport.emd')
    original_metric, original_lp = evaluation.earth_mover_distance, emd.pot.emd2
    clouds, native_proofs = [], []
    def observed_lp(a, b, costs, **kwargs):
        value, proof = original_lp(a, b, costs, log=True, **kwargs)
        if int(proof['result_code']) != 1 or proof['warning'] is not None:
            raise ValueError('Full native squared-cost exactLP must reach OPTIMAL')
        native_proofs.append(dict(shape=list(costs.shape),dtype=str(costs.dtype),result_code=1,
            transport_objective=float(value),reported_W2=math.sqrt(float(value))))
        return value
    def observed_metric(p, q, *a, **kw):
        clouds.append((np.array(p, copy=True), np.array(q, copy=True)))
        return original_metric(p, q, *a, **kw)
    evaluation.earth_mover_distance, emd.pot.emd2 = observed_metric, observed_lp
    model.train(True)
    try:
        W2 = evaluation.evaluate_kantorovich_v2(torch.device('cpu'), args, model)
    finally:
        evaluation.earth_mover_distance, emd.pot.emd2 = original_metric, original_lp
    if len(clouds) != 2 or len(native_proofs) != 2:
        raise ValueError('Both complete original directional evaluations required')
    output.mkdir(parents=True, exist_ok=False)
    values, proofs = {}, []
    target = args.data.get_data()[args.data.get_times() == args.leaveout_timepoint]
    for i, (prediction, truth64) in enumerate(clouds):
        direction = 'future' if i == 0 else 'past'
        adjacent = args.leaveout_timepoint + (1 if i == 0 else -1)
        if prediction.shape != (c['full_population_counts'][str(adjacent)], 5) or not np.array_equal(truth64, target):
            raise ValueError('Full original directional population/target/order required')
        truth = truth64.astype(np.float32)
        costs = torch.cdist(torch.from_numpy(prediction), torch.from_numpy(truth)).numpy()
        a, b = emd.pot.unif(len(prediction)), emd.pot.unif(len(truth))
        value, proof = original_lp(a, b, costs, numItermax=1e7, log=True)
        independent = cdist(prediction, truth, metric='euclidean')
        other, other_proof = original_lp(a, b, independent, numItermax=1e7, log=True)
        if any(int(p['result_code']) != 1 or p['warning'] is not None for p in [proof, other_proof]):
            raise ValueError('Native rawW1 and independent fullfloat64 LP must reach OPTIMAL')
        error = max(float(np.max(np.abs(independent[j:j+256] - costs[j:j+256]))) for j in range(0,len(prediction),256))
        difference = abs(float(value) - float(other))
        bound = error + 1e-8 * max(1., abs(float(value)), abs(float(other)))
        if difference > bound:
            raise ValueError('Full independent W1 violates every-entry cost perturbation bound')
        values['W1_' + direction] = float(value)
        values['W2_' + direction] = float(W2[i])
        np.save(output / (direction + '_prediction.npy'), prediction, allow_pickle=False)
        np.save(output / (direction + '_native_truth64.npy'), truth64, allow_pickle=False)
        np.save(output / (direction + '_matched_truth32.npy'), truth, allow_pickle=False)
        proofs.append(dict(direction=direction, shape=list(costs.shape),native_W2=native_proofs[i],
            raw_W1=float(value),independent_float64_W1=float(other),all_cost_entries_checked=int(costs.size),
            native_W1_LP_result_code=1,independent_W1_LP_result_code=1,
            objective_difference=difference,maximum_cost_entry_difference=error,cost_perturbation_bound=bound))
        del costs, independent
    return dict(metrics=values, proofs=proofs, primary_W1=values['W1_past'],
        primary_direction='past_to_leftout', model_evaluation_training_flag=True,
        selected_better_direction=False, full_population=True)


def probe(c):
    root = Path(c['artifact_root']) / 'native_full27_probe'
    root.mkdir(parents=True, exist_ok=False)
    records, evaluations = [], []
    for index in range(9):
        for leftout in [1,2,3]:
            job = next(j for j in c['jobs'] if j['recipe_index'] == index and j['leftout_timepoint'] == leftout and j['seed'] == 42)
            out = root / f'recipe{index}_leave{leftout}'
            out.mkdir()
            torch, np, native, args, model, logger = runtime(c, job, out)
            optimizer = native.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer.zero_grad(); model.train()
            full = torch.from_numpy(args.data.get_data()[args.data.get_times() != leftout]).float()
            loss = native.compute_loss(torch.device('cpu'), args, model, None, logger, full)
            if not torch.isfinite(loss):
                raise ValueError('Original full batch loss must be finite')
            loss.backward()
            grads = [(name, p.grad) for name,p in model.named_parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for _,g in grads):
                raise ValueError('Original fullbatch exacttrace/adjoint backward must be finite')
            records.append(dict(recipe_index=index,leftout_timepoint=leftout,batch_size=1000,
                full_data_shape=list(full.shape),parameters=9621,loss=float(loss.item()),
                finite_gradient_tensors=len(grads),optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__name__,
                optimizer_updates=0,untrained_probe_not_benchmark=True))
            if index == 0:
                evaluations.append(dict(leftout_timepoint=leftout, **metrics(c,torch,np,args,model,out/'full_population_metrics')))
            write(ROOT/c['state_root']/'probe_progress.json',dict(completed_cases=len(records),required_cases=27,optimizer_updates=0))
            del loss, grads, model, optimizer, full, args
            gc.collect()
    exclusive(root/'report.json',dict(passed=True,configuration_sha256=sha(ROOT/QUEUE),records=records,
        full_population_native_evaluations=evaluations,required_cases=27,optimizer_updates=0,
        untrained_probe_not_benchmark=True,all_original_experiments_complete=False))


def train(c, job):
    out = Path(job['artifact_directory'])
    out.mkdir(parents=True, exist_ok=False)
    torch, np, native, args, model, logger = runtime(c,job,out)
    trainer, _ = import_file('_TrajectoryNet_corrected_native_grid_v1',CORRECTION).corrected_functions(native)
    progress = dict(steps=0,training_loss_calls=0,validation_loss_calls=0,best_loss=float('inf'),best_step=None)
    training_losses, validation_losses, selected_history = [], [], []
    optimizer_ref = []
    original_loss = native.compute_loss
    def loss_observer(*a, **kw):
        result = original_loss(*a, **kw)
        if not torch.isfinite(result):
            raise ValueError('Native fullbatch loss must remain finite throughout10000steps')
        key = 'training_loss_calls' if a[2].training else 'validation_loss_calls'
        progress[key] += 1
        (training_losses if a[2].training else validation_losses).append(dict(steps=progress['steps'],loss=float(result.item())))
        return result
    def optimizer_observer(*a, **kw):
        optimizer = native.optim.Adam(*a, **kw)
        original_step = optimizer.step
        def counted_step(*x, **y):
            value = original_step(*x, **y)
            progress['steps'] += 1
            if progress['steps'] % 25 == 0:
                write(out/'training_progress.json',dict(**progress,all_original_experiments_complete=False))
            return value
        optimizer.step = counted_step
        optimizer_ref.append(optimizer)
        return optimizer
    def full_state(step):
        return dict(state_dict=model.state_dict(),optimizer=optimizer_ref[0].state_dict(),
            job=job,queue_sha256=sha(ROOT/QUEUE),optimizer_updates=step,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    def selected_save(value, path, *a, **kw):
        if Path(path) != out/'checkpt.pth' or not validation_losses:
            raise ValueError('Only original validation checkpoint saves may be observed')
        current = validation_losses[-1]
        if current['loss'] >= progress['best_loss']:
            raise ValueError('Corrected checkpoint must improve source sampled-validation criterion')
        torch.save(value,path,*a,**kw)
        progress.update(best_loss=current['loss'],best_step=progress['steps'])
        selected_history.append(dict(step=progress['steps'],loss=current['loss'],checkpoint_sha256=sha(Path(path))))
        torch.save(full_state(progress['steps']),out/'selected_full_state.pt')
    class SerializationObserver:
        def __getattr__(self,name):
            return selected_save if name == 'save' else getattr(torch,name)
    trainer.__globals__['compute_loss'] = loss_observer
    trainer.__globals__['optim'] = SimpleNamespace(Adam=optimizer_observer)
    trainer.__globals__['torch'] = SerializationObserver()
    start = time.monotonic()
    trainer(torch.device('cpu'),args,model,None,tuple(),tuple(),logger)
    elapsed = time.monotonic()-start
    if (progress['steps'] != 10000 or progress['training_loss_calls'] != 10000
            or progress['validation_loss_calls'] != 100 or not selected_history
            or progress['best_loss'] != min(r['loss'] for r in validation_losses)):
        raise ValueError('All10000originalupdates/100validations and exact global validation minimum required')
    periodic = sorted(out.glob('checkpt-*.pth'))
    if len(periodic) != 400 or not (out/'checkpt-10000.pth').exists():
        raise ValueError('All original400periodic25-step checkpoints must be retained')
    torch.save(full_state(10000),out/'source_final_full_state.pt')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Every final original trained parameter must be finite')
    source_scores = {}
    source_scores['source_last_scheduled_validation'] = metrics(c,torch,np,args,model,out/'source_final_metrics')
    state = torch.load(out/'checkpt.pth',map_location='cpu')
    model.load_state_dict(state['state_dict'],strict=True)
    source_scores['corrected_validation_minimum'] = metrics(c,torch,np,args,model,out/'corrected_best_metrics')
    exclusive(out/'full_loss_history.json',dict(training=training_losses,validation=validation_losses,selected_history=selected_history))
    exclusive(out/'source_result.json',dict(passed=True,job=job,queue_sha256=sha(ROOT/QUEUE),
        parameters=9621,full_population_counts=c['full_population_counts'],actual_optimizer_updates=10000,
        training_loss_calls=10000,validation_loss_calls=100,periodic_checkpoints=400,elapsed_training_seconds=elapsed,
        corrected_best_step=progress['best_step'],corrected_best_validation_loss=progress['best_loss'],
        source_final_step=10000,source_scores=source_scores,checkpoint_selection_does_not_change_training_numerics=True,
        checkpoints=[dict(selection=s,path=str(out/name),sha256=sha(out/name)) for s,name in [
            ('corrected_validation_minimum','checkpt.pth'),('source_last_scheduled_validation','checkpt-10000.pth')]],
        full_state_files=[dict(path=str(out/name),sha256=sha(out/name)) for name in ['selected_full_state.pt','source_final_full_state.pt']],
        raw_input_sha256=c['original_data_sha256'],torch_version=torch.__version__,
        torchdiffeq_version='0.0.1',author_training_seed_unknown=True,local_training_seed=job['seed'],
        source_velocity_regularization_includes_leftout_when_enabled=True,strict_TAB_result=False,
        historical_paper_equivalence_certified=False,all_original_experiments_complete=False))


def replay(c, job):
    out = Path(job['artifact_directory']);source_path=out/'source_result.json';source=read(source_path)
    if not source['passed'] or source['job']!=job or source['queue_sha256']!=sha(ROOT/QUEUE) or source['actual_optimizer_updates']!=10000:
        raise ValueError('Every full original training result must be bound before replay')
    target = out/'native_replay';target.mkdir(exist_ok=False)
    results = {}
    for checkpoint in source['checkpoints']:
        if sha(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError('Frozen trained checkpoint changed')
        selection=checkpoint['selection']
        torch,np,native,args,model,logger=runtime(c,job,target,checkpoint['path'])
        result=metrics(c,torch,np,args,model,target/selection)
        previous=out/('corrected_best_metrics' if selection=='corrected_validation_minimum' else 'source_final_metrics')
        for name in ['future_prediction.npy','past_prediction.npy','future_native_truth64.npy','past_native_truth64.npy',
                     'future_matched_truth32.npy','past_matched_truth32.npy']:
            if not np.array_equal(np.load(previous/name,allow_pickle=False),np.load(target/selection/name,allow_pickle=False)):
                raise ValueError('Every complete predicted and true coordinate must replay bitwise on CPU')
        original=source['source_scores'][selection]
        if set(result['metrics'])!=set(original['metrics']) or any(not math.isclose(result['metrics'][k],v,
                rel_tol=c['metric_tolerance']['rtol'],abs_tol=c['metric_tolerance']['atol']) for k,v in original['metrics'].items()):
            raise ValueError('Every full directional W1/W2 metric must replay within frozen tolerance')
        results[selection]=dict(**result,all_prediction_and_truth_coordinates_bitwise_equal=True)
        del model,args
        gc.collect()
    exclusive(target/'report.json',dict(passed=True,job=job,queue_sha256=sha(ROOT/QUEUE),
        source_result_sha256=sha(source_path),actual_training_updates=10000,checkpoint_selections=results,
        historical_paper_equivalence_certified=False,all_original_experiments_complete=False))


def aggregate(records, jobs):
    if len(records)!=15 or len(jobs)!=15 or {r['job']['id'] for r in records}!={j['id'] for j in jobs}:
        raise ValueError('Only all15 full source-and-replay conditions may form a grid row')
    if {(j['seed'],j['leftout_timepoint']) for j in jobs}!={(s,t) for s in [42,43,44,45,46] for t in [1,2,3]}:
        raise ValueError('Every independent seed and every interior leaveout is required')
    if len({j['recipe_index'] for j in jobs}) != 1:
        raise ValueError('Distinct coefficient recipes must never be pooled or selected by test performance')
    expected={j['id']:j for j in jobs}
    if any(not r['passed'] or r['actual_training_updates']!=10000 or r['job']!=expected[r['job']['id']] for r in records):
        raise ValueError('Every full training/replay identity and 10000-step budget must be verified')
    values={}
    for selection in ['corrected_validation_minimum','source_last_scheduled_validation']:
        seed_values=[statistics.mean(r['checkpoint_selections'][selection]['primary_W1'] for r in records if r['job']['seed']==s) for s in [42,43,44,45,46]]
        mean,sd=statistics.mean(seed_values),statistics.stdev(seed_values);half=2.7764451051977987*sd/math.sqrt(5)
        values[selection]=dict(mean_W1=mean,seed_leaveout_means=seed_values,five_seed_mean_SD=sd,
            descriptive_df4_t95=[mean-half,mean+half],separate_all15_model_SD=statistics.stdev(r['checkpoint_selections'][selection]['primary_W1'] for r in records))
    return dict(full_models=15,seed_count=5,all_leftouts=[1,2,3],checkpoint_selections=values,
        selected_hyperparameters_by_test=False,paper_selected_coefficient_identity_certified=False,
        strict_TAB_result=False,all_original_experiments_complete=False)


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Preserve every original full180 controller and attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        required_grid_models=180,available_training_models=135,deferred_energy_models=45,completed_ids=[],
        failed_preserved={},all_original_experiments_complete=False)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def run(phase,job=None):
        key='full27_probe' if job is None else hashlib.sha256(job['id'].encode()).hexdigest()[:20]+'_'+phase
        attempt=base/'attempts'/key;attempt.mkdir(parents=True,exist_ok=False)
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            verify(c)
            command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--phase',phase]
            if job:command+=['--job-id',job['id']]
            with (attempt/'stdout.log').open('xb') as stdout,(attempt/'stderr.log').open('xb') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='-1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                update(dict(state='running_full_original_TrajectoryNet_'+phase,active_pid=child.pid,exact_child_handle=handle))
                code,usage=capped_wait(child,c,api,update)
        path=Path(c['artifact_root'])/'native_full27_probe/report.json' if job is None else Path(job['artifact_directory'])/('source_result.json' if phase=='train' else 'native_replay/report.json')
        passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
        exclusive(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
            report_path=str(path),report_sha256=sha(path) if path.exists() else None))
        update(dict(active_pid=None,resource_guard_aborted=usage['resource_guard_aborted']))
        if not passed:
            raise RuntimeError('Preserved original TrajectoryNet full attempt failed: '+key)
    try:
        while not author_ready(c):
            update(dict(state='waiting_actual_original_author_checkpoint_full_replay_guard'));time.sleep(10)
        run('probe')
        for job in c['jobs']:
            update(dict(active_job=job['id']));run('train',job);run('replay',job)
            state['completed_ids'].append(job['id']);update(dict(state='original10000step_training_and_native_full_replay_verified'))
            selected=[j for j in c['jobs'] if j['recipe_index']==job['recipe_index']]
            if {j['id'] for j in selected}.issubset(state['completed_ids']):
                records=[];guards=[]
                for j in selected:
                    key=hashlib.sha256(j['id'].encode()).hexdigest()[:20]
                    for phase,name in [('train','source_result.json'),('replay','native_replay/report.json')]:
                        g=base/'attempts'/(key+'_'+phase)/'guard_receipt.json';receipt=read(g);report=Path(j['artifact_directory'])/name
                        if not receipt['passed'] or receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or receipt['report_sha256']!=sha(report):
                            raise ValueError('Every complete original source/replay guard is required before row aggregation')
                        guards.append(dict(path=str(g),sha256=sha(g)))
                    records.append(read(Path(j['artifact_directory'])/'native_replay/report.json'))
                row=aggregate(records,selected);row.update(recipe=job['recipe'],recipe_index=job['recipe_index'],guards=guards,queue_sha256=sha(ROOT/QUEUE))
                path=base/'complete_rows'/f'recipe{job["recipe_index"]}.json';path.parent.mkdir(exist_ok=True)
                exclusive(path,row)
        update(dict(state='all135_bound_grid_conditions_complete_remaining45_energy_recipe_binding_required'))
        while True:
            # Energy identities remain in the queue, never silently removed or
            # mapped to an unrelated standalone RNODE image model.
            time.sleep(30)
            update(dict(state='waiting_source_bound_energy_Jacobian_recipe_for_remaining45'))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair',error=str(error)));raise
    finally:
        owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--phase',choices=['probe','train','replay']);p.add_argument('--job-id');a=p.parse_args()
    if a.register:exclusive(ROOT/QUEUE,registration());return
    c=read(ROOT/QUEUE);verify(c)
    if not a.phase:controller(c);return
    if a.phase=='probe':probe(c);return
    job=next(j for j in c['jobs'] if j['id']==a.job_id)
    (train if a.phase=='train' else replay)(c,job)


if __name__=='__main__':main()
