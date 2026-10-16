"""Run every defined DSB single-cell condition with actual original IPF loops."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
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
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as source

MODULE = 'scripts.flow_matching.run_dsb_singlecell_full70_v1'
QUEUE = 'configs/experiments/fm_dsb_singlecell_native_full70.v1.json'
MODEL = 'configs/models/fm_dsb_singlecell_native_IPF_E4_CPU.v1.json'
CORE = 'configs/reproducibility/fm_dsb_singlecell_all12_full_input_preflight.v2.json'
ADAPTER = 'src/iia_benchmark/models/dsb_singlecell_native_ipf_v1.py'
COMPONENTS = 'src/iia_benchmark/models/dsb_singlecell_source_components_v1.py'
ACQUISITION = 'configs/acquisition/fm_dsb_accelerate_021_overlay.v1.json'
DEFERRED = 'GEO13176_QC_full_raw_RNA'


def jobs_for(cases, seeds):
    jobs = []
    for case in cases:
        for seed in seeds:
            identity = f'dsb_singlecell_sourceE4__{case["dataset"]}__leave{case["leftout_timepoint"]}__seed{seed}'
            jobs.append(dict(id=identity, dataset=case['dataset'], leftout_timepoint=case['leftout_timepoint'],
                seed=seed, data=case.get('data'), artifact_directory='F:/aicoding/IIA_Data/experiments/dsb_singlecell_native_full70_v1/'
                + hashlib.sha256(identity.encode()).hexdigest()[:20]))
    return jobs


def registration():
    core, profile = read(ROOT / CORE), read(ROOT / MODEL)
    acquisition = read(ROOT / ACQUISITION)
    installed = read(Path(acquisition['output_directory']) / 'verified_installation.json')
    if not installed['passed'] or not installed['official_PyPI_sha256_verified'] or installed['base_environment_modified']:
        raise ValueError('Verified independent original Accelerate API overlay required')
    jobs = jobs_for(core['cases'], profile['training_seeds'])
    deferred = jobs_for([dict(dataset=DEFERRED, leftout_timepoint=i) for i in [1, 2]], profile['training_seeds'])
    if len(jobs) != 60 or len(deferred) != 10:
        raise ValueError('All60available and10known deferred native DSB conditions required')
    paths = [MODULE.replace('.', '/') + '.py', MODEL, CORE, ADAPTER, COMPONENTS, ACQUISITION,
             'tests/test_fm_dsb_singlecell_native_ipf_v1.py', 'tests/test_fm_dsb_singlecell_full70_queue_v1.py',
             str(Path(acquisition['output_directory']) / 'verified_installation.json'),
             'scripts/flow_matching/run_otcfm_CITE13176_GEO_native60_v1.py',
             'configs/reproducibility/fm_otcfm_CITE13176_GEO_full60_after_preprocessing.2026-10-16_v1.json']
    paths += [row['path'] for row in core['source_receipts']]
    return dict(jobs=jobs, deferred_jobs=deferred, required_models=70, available_models=60, deferred_models=10,
        profile=MODEL, source_core_preflight_configuration=CORE,
        source_core_preflight_report=str(Path(core['artifact_root']) / 'full12_source_cache_preflight.json'),
        source_core_preflight_guard=core['state_root'] + '/guard_receipt.json',
        environment_lock=core['environment_lock'], source_runtime_queue=core['source_runtime_queue'],
        python=core['python'], numerical_threads=2, settings=core['settings'], resource_lock=core['resource_lock'],
        overlay_receipt=str(Path(acquisition['output_directory']) / 'verified_installation.json'),
        overlay_directory=installed['isolated_import_directory'], source_root=profile['source_root'],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(paths))],
        state_root='experiments/runs/fm_dsb_singlecell_native_full70_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/dsb_singlecell_native_full70_v1',
        required_native_constructor_probe_cases_available=12, required_native_constructor_probe_cases_deferred=2,
        actual_updates_per_model=400040, training_batch=512, cache_particles=10000, langevin_steps=20,
        metric_tolerance=dict(rtol=2e-6, atol=2e-6), require_bitwise_stochastic_CPU_checkpoint_replay=True,
        paper_targets=dict(CITE=dict(mean=.953, std=.140), EB=dict(mean=.862, std=.023), Multiome=dict(mean=1.079, std=.117)),
        original_scope=core['original_scope'], all_original_experiments_complete=False,
        boundary=profile['boundary'])


def verify(c):
    if c != registration():
        raise ValueError('All70conditions,exact source dependency/core/budget/guard bindings required')
    installation = read(Path(c['overlay_receipt']))
    for row in installation['installed_files']:
        if sha(Path(c['overlay_directory']) / row['path']) != row['sha256']:
            raise ValueError('Pinned original Accelerate dependency changed')


def core_ready(c):
    path = ROOT / c['source_core_preflight_guard']
    if not path.exists():
        return False
    guard = read(path)
    if not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']:
        raise ValueError('Preserved core preflight failure requires isolated repair,never a silent retry')
    report = read(Path(c['source_core_preflight_report']))
    if (guard['report_sha256'] != sha(Path(c['source_core_preflight_report'])) or not report['passed']
            or report['configuration_sha256'] != sha(ROOT / CORE) or len(report['records']) != 12
            or report['required_cases'] != 12 or report['optimizer_updates'] != 0):
        raise ValueError('All12full core source preflights and exact original configuration required')
    return True


def all_jobs(c):
    target = ROOT / c['state_root'] / 'deferred_jobs_materialized.json'
    if target.exists():
        binding = read(target)
        if binding['source_queue_sha256'] != sha(ROOT / QUEUE):
            raise ValueError('Deferred original queue binding changed')
        for row in binding['preparation_receipts']:
            if sha(Path(row['path'])) != row['sha256']:
                raise ValueError('Immutable full deferred preprocessing evidence changed')
        if ([j['id'] for j in binding['jobs']] != [j['id'] for j in c['deferred_jobs']]
                or len(binding['jobs']) != 10):
            raise ValueError('All10frozen deferred conditions must be retained exactly')
        return c['jobs'] + binding['jobs']
    return c['jobs']


def materialize_deferred(c):
    from scripts.flow_matching import run_otcfm_CITE13176_GEO_native60_v1 as preparation
    ready = preparation.ready_preparation()
    if ready is None:
        return False
    prep, report, binding = ready
    jobs = [dict(j, data=dict(path=report['target_input_for_author_loader'], sha256=report['target_input_sha256'],
             cells=29394, days=report['days'], donor='13176', full_original_PC_dimensions=5,
             citation=prep['source_citation'], historical_paper_identity_certified=False)) for j in c['deferred_jobs']]
    receipts = [dict(path=binding[k], sha256=binding[k.replace('_path', '_sha256')])
                for k in ['preprocessing_report_path', 'preprocessing_guard_path']]
    exclusive(ROOT / c['state_root'] / 'deferred_jobs_materialized.json',
              dict(source_queue_sha256=sha(ROOT / QUEUE), jobs=jobs, preparation_receipts=receipts,
                   all_original_experiments_complete=False))
    return True


def runtime(c):
    torch, pl, distribution, _, OmegaConf = source.source_runtime(read(ROOT / c['source_runtime_queue']))
    modules = []
    for name, path in [('_iia_DSB_full_native_components', COMPONENTS), ('_iia_DSB_full_native_IPF', ADAPTER)]:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    components, adapter = modules
    trainer = adapter.load_native_trainer(ROOT / c['source_root'], c['overlay_directory'])
    return torch, pl, distribution, OmegaConf, components, adapter, trainer


def construct(c, job, directory, environment):
    torch, pl, distribution, OmegaConf, components, adapter, trainer = environment
    if sha(ROOT / job['data']['path']) != job['data']['sha256']:
        raise ValueError('Untouched full original input changed')
    pl.seed_everything(job['seed'], workers=True)
    dm = distribution.CustomTrajectoryDataModule(system=str(ROOT / job['data']['path']),
        train_val_test_split=(.8, .1, .1), max_dim=5, whiten=True, batch_size=512,
        num_workers=0, pin_memory=False, seed=job['seed'])
    counts = {str(t): len(x) for t, x in zip(dm.ulabels, dm.timepoint_data)}
    if counts != job['data']['days'] or sum(counts.values()) != job['data']['cells'] or dm.dim != 5:
        raise ValueError('All original observations,alltimepoints and5PCs required')
    indices = components.endpoint_indices(len(dm.timepoint_data), job['leftout_timepoint'])
    def endpoint(index):
        subset = dm.split_timepoint_data[index][0]
        return torch.as_tensor(dm.timepoint_data[index][list(subset.indices)])
    earlier, later = endpoint(indices['source_final_index']), endpoint(indices['source_initial_index'])
    profile = read(ROOT / MODEL)
    args = OmegaConf.create(profile['native_arguments'])
    args.CSV_log_dir = str(directory / 'native_logs')
    runner_type = adapter.runner_class(trainer, components, ROOT / c['source_root'], directory,
                                       earlier, later, job['seed'])
    runner = runner_type(args)
    if sum(p.numel() for p in runner.net.parameters()) != 28522:
        raise ValueError('Both full independent14261parameter networks required')
    return dm, runner, indices


def metrics(prediction, truth, independent=False):
    import numpy as np
    import torch
    from scipy.spatial.distance import cdist
    from src.models.components.distribution_distances import compute_distribution_distances
    import src.models.components.optimal_transport as transport
    original = transport.pot.emd2
    solves = []
    def observed(a, b, costs, **kwargs):
        if len(solves) >= 2 or kwargs.get('log'):
            raise ValueError('Two unchanged original fullW1/W2 solves required')
        value, log = original(a, b, costs, log=True, **kwargs)
        if int(log['result_code']) != 1 or log['warning'] is not None:
            raise ValueError('Original full-data LP must reach certified OPTIMAL')
        row = dict(power=len(solves) + 1, full_cost_shape=list(costs.shape),
                   objective=float(value), LP_result_code=int(log['result_code']))
        if independent and not solves:
            other_cost = cdist(prediction, truth, metric='euclidean')
            maximum = max(float(np.max(np.abs(other_cost[i:i+256] - costs[i:i+256])))
                          for i in range(0, len(prediction), 256))
            other, proof = original(a, b, other_cost, log=True, **kwargs)
            if int(proof['result_code']) != 1 or proof['warning'] is not None:
                raise ValueError('Independent full SciPy W1 LP must reach OPTIMAL')
            difference = abs(float(value) - float(other))
            bound = maximum + 1e-8 * max(1., abs(float(value)), abs(float(other)))
            if difference > bound:
                raise ValueError('Independent fullW1 exceeds the measured all-entry cost perturbation bound')
            row.update(independent_float64_W1=float(other), independent_LP_result_code=1,
                       cost_entry_maximum_difference=maximum, objective_difference=difference,
                       all_entry_cost_perturbation_bound=bound, independent_cost_entries=int(other_cost.size))
        solves.append(row)
        return value
    transport.pot.emd2 = observed
    try:
        names, values = compute_distribution_distances([torch.from_numpy(prediction)], [torch.from_numpy(truth)])
    finally:
        transport.pot.emd2 = original
    if len(solves) != 2 or len(names) != 8 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Every complete native heldout metric must be finite')
    return dict(zip(names, map(float, values))), solves


def worker(c, phase, job_id=None, tag=None):
    import numpy as np
    environment = runtime(c)
    torch = environment[0]
    profile = read(ROOT / MODEL)
    jobs = all_jobs(c)
    if phase == 'probe':
        selected = [j for j in jobs if j['seed'] == 42 and ((j['dataset'] == DEFERRED) == (tag == 'deferred'))]
        expected = 2 if tag == 'deferred' else 12
        if len(selected) != expected:
            raise ValueError('Every full native constructor/cache branch required before training')
        base = Path(c['artifact_root']) / ('native_' + tag + '_probe')
        base.mkdir(parents=True, exist_ok=False)
        records = []
        for job in selected:
            directory = base / hashlib.sha256(job['id'].encode()).hexdigest()[:20]
            directory.mkdir()
            os.chdir(directory)
            dm, runner, indices = construct(c, job, directory, environment)
            directions = []
            for role in ['b', 'f']:
                loader = runner.new_cacheloader(role, 1, False)
                x, target, times = next(loader)
                runner.accelerate(role)
                loss = torch.nn.functional.mse_loss(runner.net[role](x, runner.T - times), target)
                runner.accelerator.backward(loss)
                if len(x) != 512 or not torch.isfinite(loss) or not all(
                        p.grad is not None and torch.isfinite(p.grad).all() for p in runner.net[role].parameters()):
                    raise ValueError('Actual original Accelerator/cache/512batch gradients must pass')
                runner.net[role].zero_grad(set_to_none=True)
                directions.append(dict(direction=role, batch=512, loss=float(loss.detach()), optimizer_updates=0))
                loader.close()
                del loader, x, target, times, loss
            if sum(runner._iia_updates.values()) != 0:
                raise ValueError('Native constructor probe must never update a weight')
            with torch.no_grad():
                start = torch.as_tensor(dm.timepoint_data[indices['backward_start_index']])
                trajectory, _, _ = runner.langevin.record_langevin_seq(runner.net['b'], start, ipf_it=1, sample=True)
            prediction = trajectory[:,9,:].cpu().numpy()
            truth = dm.timepoint_data[job['leftout_timepoint']]
            if trajectory.shape != (len(start),20,5) or not np.isfinite(prediction).all():
                raise ValueError('Full actual native backward trajectory must be finite')
            probe_metrics, solves = metrics(prediction, truth, independent=True)
            records.append(dict(id=job['id'], full_timepoint_counts=job['data']['days'],
                                original_trainer_class='bridge.runners.ipf.IPFSequential',
                                original_accelerate_version='0.2.1', full_cache_particles=10000,
                                original_gamma_tensor_dtype=runner.gammas_dtype, directions=directions,
                                parameters=28522, optimizer_updates=0, benchmark_performance_claim=False,
                                complete_trajectory_shape=list(trajectory.shape), untrained_pipeline_metrics=probe_metrics,
                                full_native_and_independent_LP_proofs=solves))
            write(ROOT/c['state_root']/('native_probe_progress_'+tag+'.json'),
                  dict(completed_cases=len(records),required_cases=expected,optimizer_updates=0,benchmark_performance_claim=False))
            del dm, runner, trajectory, start, prediction, truth
            gc.collect()
        exclusive(base / 'report.json', dict(passed=True, records=records, required_cases=expected,
                   configuration_sha256=sha(ROOT / QUEUE), optimizer_updates=0, all_original_experiments_complete=False))
        return
    job = next(j for j in jobs if j['id'] == job_id)
    if sha(ROOT/job['data']['path']) != job['data']['sha256']:
        raise ValueError('Every original raw-input byte must match before training or replay')
    out = Path(job['artifact_directory'])
    if phase == 'train':
        out.mkdir(parents=True, exist_ok=False)
        os.chdir(out)
        started = time.monotonic()
        dm, runner, indices = construct(c, job, out, environment)
        runner.train()
        training_seconds = time.monotonic() - started
        if len(runner._iia_phases) != 40 or runner._iia_updates != dict(f=200020, b=200020):
            raise ValueError('All40original IPF passes and400040actual optimizer updates required')
        eval_seed = job['seed'] + profile['evaluation_seed_offset']
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        torch.save(dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate()),
                   out / 'evaluation_start_rng.pt')
        earlier = torch.as_tensor(dm.timepoint_data[indices['backward_start_index']])
        with torch.no_grad():
            trajectory, _, _ = runner.langevin.record_langevin_seq(runner.net['b'], earlier, ipf_it=20, sample=True)
        prediction = trajectory[:, 9, :].cpu().numpy()
        truth = dm.timepoint_data[job['leftout_timepoint']]
        if trajectory.shape != (len(earlier), 20, 5) or not np.isfinite(prediction).all():
            raise ValueError('All original source observations and native20-step SDE midpoint required')
        np.save(out / 'prediction.npy', prediction, allow_pickle=False)
        np.save(out / 'truth.npy', truth, allow_pickle=False)
        del trajectory, runner, dm, earlier
        gc.collect()
        values, solves = metrics(prediction, truth)
        artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*'))
                     if p.is_file()]
        exclusive(out / 'source_result.json', dict(passed=True, job=job, queue_sha256=sha(ROOT / QUEUE),
            selected_checkpoint=str(out / 'phase20_b_full_state.pt'), full_training_complete=True,
            actual_optimizer_updates=400040, training_seconds=training_seconds, parameters=28522,
            full_prediction_shape=list(prediction.shape), full_truth_shape=list(truth.shape),
            metrics=values, LP_solves=solves, artifacts=artifacts,
            historical_paper_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
        return
    result = read(out / 'source_result.json')
    if result['job'] != job or result['queue_sha256'] != sha(ROOT / QUEUE) or result['actual_optimizer_updates'] != 400040:
        raise ValueError('Full frozen source model completion required')
    for item in result['artifacts']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Every full source trained artifact must remain unchanged')
    replay = out / 'native_replay'
    replay.mkdir(exist_ok=False)
    torch, pl, distribution, _, components, _, _ = environment
    # The native backward Langevin method needs only the selected actual weights,
    # original20-step schedule and complete earlier endpoint observations.
    pl.seed_everything(job['seed'], workers=True)
    dm = distribution.CustomTrajectoryDataModule(system=str(ROOT / job['data']['path']),
        train_val_test_split=(.8,.1,.1), max_dim=5, whiten=True, batch_size=512,
        num_workers=0, pin_memory=False, seed=job['seed'])
    model = components.build_score_network(ROOT / c['source_root'])
    state = torch.load(result['selected_checkpoint'], map_location='cpu')
    model.load_state_dict(state['networks']['b'], strict=True)
    _, Langevin, _ = components.original_components(ROOT / c['source_root'])
    half = np.linspace(.001, .001, 10)
    gammas = torch.tensor(np.concatenate([half, np.flip(half)]))
    earlier_subset = dm.split_timepoint_data[job['leftout_timepoint'] - 1][0]
    earlier_train = torch.as_tensor(dm.timepoint_data[job['leftout_timepoint'] - 1][list(earlier_subset.indices)])
    process = Langevin(20, (5,), gammas, torch.distributions.Categorical(torch.ones(20)), device='cpu',
                       mean_final=earlier_train.mean(dim=0), var_final=earlier_train.var(), mean_match=False)
    rng = torch.load(out / 'evaluation_start_rng.pt', map_location='cpu')
    torch.set_rng_state(rng['torch_rng'])
    np.random.set_state(rng['numpy_rng'])
    random.setstate(rng['python_rng'])
    earlier = torch.as_tensor(dm.timepoint_data[job['leftout_timepoint'] - 1])
    with torch.no_grad():
        trajectory, _, _ = process.record_langevin_seq(model, earlier, ipf_it=20, sample=True)
    prediction = trajectory[:, 9, :].cpu().numpy()
    truth = dm.timepoint_data[job['leftout_timepoint']]
    if not np.array_equal(prediction, np.load(out / 'prediction.npy', allow_pickle=False)) or not np.array_equal(
            truth, np.load(out / 'truth.npy', allow_pickle=False)):
        raise ValueError('Restored original stochastic CPU checkpoint must match every prediction/truth coordinate bitwise')
    values, solves = metrics(prediction, truth, independent=True)
    if set(values) != set(result['metrics']) or any(not math.isclose(values[k], result['metrics'][k],
            rel_tol=c['metric_tolerance']['rtol'], abs_tol=c['metric_tolerance']['atol']) for k in values):
        raise ValueError('Every original native metric must pass the frozen2e-6tolerance')
    exclusive(replay / 'report.json', dict(passed=True, job=job, actual_full_native_checkpoint_replay=True,
        all_prediction_and_truth_coordinates_bitwise_equal=True, native_metrics=values, LP_solves=solves,
        source_result_sha256=sha(out / 'source_result.json'), queue_sha256=sha(ROOT / QUEUE),
        full_training_updates=400040, historical_paper_equivalence_certified=False,
        all_original_experiments_complete=False))


def aggregate_complete_row(records, required_jobs):
    if {r['job']['id'] for r in records} != {j['id'] for j in required_jobs} or len(records) != len(required_jobs):
        raise ValueError('No partial-seed or partial-leaveout row may be aggregated')
    seeds = sorted({j['seed'] for j in required_jobs})
    leftouts = sorted({j['leftout_timepoint'] for j in required_jobs})
    if seeds != [42,43,44,45,46] or len(records) != len(seeds) * len(leftouts):
        raise ValueError('Fiveindependent training seeds and allinterior leaveouts required')
    names = set(records[0]['native_metrics'])
    result = {}
    for name in sorted(names):
        means = [statistics.mean(r['native_metrics'][name] for r in records if r['job']['seed'] == seed) for seed in seeds]
        mean, sd = statistics.mean(means), statistics.stdev(means)
        half = 2.7764451051977987 * sd / math.sqrt(5)
        result[name] = dict(seed_leaveout_means=dict(zip(map(str,seeds),means)), mean=mean,
                            five_seed_mean_SD=sd, descriptive_df4_t95=[mean-half, mean+half],
                            separate_all_model_SD=statistics.stdev(r['native_metrics'][name] for r in records))
    return dict(full_models=len(records), required_models=len(required_jobs), full_seeds=seeds, all_leftouts=leftouts,
                metrics=result, uncertainty_unit='training-seed mean across allleaveouts;separate allmodelSD is not interchangeable',
                historical_paper_equivalence_certified=False, all_original_experiments_complete=False)


def complete_rows(c, completed):
    jobs = all_jobs(c)
    for dataset in sorted({j['dataset'] for j in jobs}):
        selected = [j for j in jobs if j['dataset'] == dataset]
        if not {j['id'] for j in selected}.issubset(completed):
            continue
        path = ROOT / c['state_root'] / 'complete_rows' / (dataset + '.json')
        if path.exists():
            continue
        records, receipts = [], []
        for job in selected:
            key = hashlib.sha256(job['id'].encode()).hexdigest()[:20]
            report = Path(job['artifact_directory']) / 'native_replay/report.json'
            source_report = Path(job['artifact_directory']) / 'source_result.json'
            for phase, artifact in [('train', source_report), ('replay', report)]:
                guard_path = ROOT / c['state_root'] / 'attempts' / (key + '_' + phase) / 'guard_receipt.json'
                guard = read(guard_path)
                if not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted'] or guard['report_sha256'] != sha(artifact):
                    raise ValueError('Complete row requires everyactual source and replayguard')
                receipts.append(dict(path=str(guard_path), sha256=sha(guard_path)))
            record = read(report)
            if not record['passed'] or record['job'] != job or record['source_result_sha256'] != sha(source_report):
                raise ValueError('Complete row requires everyfull native replay identity')
            records.append(record)
        row = aggregate_complete_row(records, selected)
        family = 'CITE' if dataset == DEFERRED or 'CITE' in dataset else ('EB' if 'EB' in dataset else 'Multiome')
        row.update(dataset=dataset, queue_sha256=sha(ROOT / QUEUE), source_and_replay_receipts=receipts,
                   paper_reference_target=c['paper_targets'][family],
                   paper_mean_difference=row['metrics']['1-Wasserstein']['mean'] - c['paper_targets'][family]['mean'],
                   published_SD_aggregation_identity_certified=False, strict_TAB_result=False,
                   matched_CFM_cross_method_comparison_complete=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        exclusive(path, row)


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades()
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None or (base / 'status.json').exists():
        if owner:
            owner.close()
        raise FileExistsError('Preserve every original full70controller and attempt')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
                 required_models=70, completed_ids=[], failed_preserved={}, all_original_experiments_complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    def run(phase, job=None, tag=None):
        key = (tag + '_probe') if phase == 'probe' else hashlib.sha256(job['id'].encode()).hexdigest()[:20] + '_' + phase
        attempt = base / 'attempts' / key
        attempt.mkdir(parents=True, exist_ok=False)
        with api['resource_slot'](ROOT / c['resource_lock'], c['settings'], update):
            verify(c)
            cmd = [c['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--phase', phase]
            cmd += ['--tag', tag] if tag else ['--job-id', job['id']]
            with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=cmd)
                update(dict(state='running_full_native_DSB_' + phase, active_pid=child.pid, exact_child_handle=handle))
                code, usage = capped_wait(child, c, api, update)
        path = (Path(c['artifact_root']) / ('native_' + tag + '_probe') / 'report.json') if tag else (
            Path(job['artifact_directory']) / ('source_result.json' if phase == 'train' else 'native_replay/report.json'))
        passed = code == 0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
        exclusive(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                  exact_child_handle=handle, report_path=str(path), report_sha256=sha(path) if path.exists() else None))
        update(dict(active_pid=None))
        if not passed:
            raise RuntimeError('Preserved full native DSB attempt failed: ' + key)
    try:
        while not core_ready(c):
            update(dict(state='waiting_actual_all12_full_source_core_preflight'))
            time.sleep(10)
        run('probe', tag='available')
        for job in c['jobs']:
            update(dict(active_job=job['id']))
            run('train', job)
            run('replay', job)
            state['completed_ids'].append(job['id'])
            update(dict(state='full_source_training_and_native_replay_verified'))
            complete_rows(c, state['completed_ids'])
        while len(all_jobs(c)) == 60:
            if materialize_deferred(c):
                break
            update(dict(state='available60_complete_waiting_all29394cell_GEO_preprocessing_for_remaining10'))
            time.sleep(10)
        run('probe', tag='deferred')
        for job in all_jobs(c)[60:]:
            update(dict(active_job=job['id']))
            run('train', job)
            run('replay', job)
            state['completed_ids'].append(job['id'])
            update(dict(state='full_source_training_and_native_replay_verified'))
            complete_rows(c, state['completed_ids'])
        if len(state['completed_ids']) != 70:
            raise ValueError('Every known full model must complete training and native replay')
        update(dict(state='full70_training_replay_rows_complete_matched_method_comparisons_and_other_scope_pending'))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair', error=str(error)))
        raise
    finally:
        owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register', action='store_true')
    p.add_argument('--phase', choices=['probe', 'train', 'replay'])
    p.add_argument('--job-id')
    p.add_argument('--tag', choices=['available', 'deferred'])
    a = p.parse_args()
    if a.register:
        exclusive(ROOT / QUEUE, registration())
        return
    c = read(ROOT / QUEUE)
    verify(c)
    if a.phase:
        worker(c, a.phase, a.job_id, a.tag)
    else:
        controller(c)


if __name__ == '__main__':
    main()
