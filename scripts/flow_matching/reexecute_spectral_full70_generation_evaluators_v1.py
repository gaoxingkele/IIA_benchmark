"""Full70 checkpoint generation and learned-evaluator reexecution.

Historical pre-sampling RNG was not saved. A separately declared new generation
seed is therefore used and saved. Two complete original samples must replay
bitwise from that snapshot; historical sample identity is never inferred.
"""
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as source
from scripts.flow_matching import run_spectral_original_job as native
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.reexecute_spectral_full70_generation_evaluators_v1'
CONFIG = 'configs/reproducibility/fm_spectral_full70_generation_evaluator_reexecution.v1.json'
TEST = 'tests/test_fm_spectral_full70_generation_evaluator_reexecution_v1.py'


def registration():
    parent = read(ROOT / source.CONFIG)
    q = read(ROOT / parent['original_queue'])
    policy = read(ROOT / q['protocol'])
    paths = [MODULE.replace('.', '/') + '.py', TEST, source.CONFIG,
        source.MODULE.replace('.', '/') + '.py', parent['original_queue'],
        q['environment_lock'], q['protocol'], 'scripts/flow_matching/run_spectral_original_job.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py']
    return dict(source_artifact_audit=source.CONFIG, original_queue=parent['original_queue'],
        required_conditions=70, required_cohorts=14, original_training_seeds=[42, 43, 44, 45, 46],
        new_sampling_seed_offset=1000000, full_same_RNG_generations_each=2,
        evaluation_repeats=policy['evaluation_repeats'], all_four_original_evaluators=True,
        historical_sampling_RNG_recovered=False, historical_generated_bytes_equivalence_claim=False,
        new_model_training_updates=0, new_training_seed_slots=0,
        python=q['python'], settings=q['settings'], resource_lock=q['resource_lock'],
        spectral_flow_minimum_commit_headroom_bytes=q['spectral_sampling_minimum_commit_headroom_bytes'],
        state_root='experiments/runs/fm_spectral_full70_generation_evaluator_reexecution_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/spectral_full70_generation_evaluator_reexecution_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        original_scope=dict(parent['original_scope'], paper_scalar_targets=702), citation=policy['citation'],
        data_citation=policy['data_archive_registry'], full_original_scope_complete=False)


def verify(c):
    if c != registration():
        raise ValueError('Full70 new-RNG generative reexecution contract changed')
    parent = read(ROOT / c['source_artifact_audit'])
    cases, _ = source.verify(parent)
    cohorts = Counter((x['job']['model'], x['job']['dataset'], x['job']['data_track']) for x in cases.values())
    if len(cases) != 70 or len(cohorts) != 14 or set(cohorts.values()) != {5}:
        raise ValueError('Every original70 condition and14 five-training-seed cohorts required')
    return parent, cases


def evaluator_nodes():
    """Select the complete unchanged evaluator section, including graph seeds."""
    main = ast.parse(inspect.getsource(native.main)).body[0]
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Import)
                 and any(a.name == 'tensorflow' for a in n.names))
    stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets))
    return copy.deepcopy(main.body[start:stop])


def original_evaluators():
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in
        ['output', 'policy', 'samples', 'reference', 'torch', 'np', 'seed_everything']],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    result = ast.parse('return dict(metrics=metrics, evaluation_seconds=evaluation_seconds)').body[0]
    function = ast.FunctionDef(name='execute', args=args, body=evaluator_nodes() + [result], decorator_list=[])
    unit = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(native.__dict__)
    exec(compile(unit, __file__, 'exec'), namespace)
    return namespace['execute']


def capture_RNG(torch, np, cuda=True):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_CPU=torch.get_rng_state().clone(),
        torch_CUDA=[s.clone() for s in torch.cuda.get_rng_state_all()] if cuda else [])


def same_RNG(a, b, torch, np):
    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0]
        and np.array_equal(a['numpy'][1], b['numpy'][1]) and a['numpy'][2:] == b['numpy'][2:]
        and torch.equal(a['torch_CPU'], b['torch_CPU'])
        and len(a['torch_CUDA']) == len(b['torch_CUDA'])
        and all(torch.equal(x, y) for x, y in zip(a['torch_CUDA'], b['torch_CUDA'])))


def restore_RNG(state, torch, np, cuda=True):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_CPU'])
    if cuda:
        torch.cuda.set_rng_state_all(state['torch_CUDA'])
    if not same_RNG(state, capture_RNG(torch, np, cuda), torch, np):
        raise ValueError('Every complete RNG state must restore exactly')


def assert_same_samples(first, second, expected_shape, np):
    if (list(first.shape) != expected_shape or first.shape != second.shape or first.dtype != second.dtype
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or first.tobytes() != second.tobytes()):
        raise ValueError('Two full original generations must match every byte, including all surplus rows')


def assert_tree_equal(actual, expected, torch):
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
            raise ValueError('Frozen model/EMA/optimizer tensor changed')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Complete checkpoint state keys changed')
        for key in expected:
            assert_tree_equal(actual[key], expected[key], torch)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('Complete checkpoint state sequence changed')
        for x, y in zip(actual, expected):
            assert_tree_equal(x, y, torch)
    elif actual != expected:
        raise ValueError('Frozen checkpoint controller value changed')


def ready_proof(parent, case, identity):
    state = read(ROOT / parent['state_root'] / 'status.json')
    item = state['verified'].get(identity)
    if item is None:
        return None
    path = ROOT / item['proof']
    proof = read(path)
    ready = source.source_ready(case)
    if (sha(path) != item['proof_sha256'] or not proof['passed'] or proof['canonical_id'] != identity
            or ready is None or sha(ready[0]) != proof['original_result_sha256']
            or sha(ready[1]) != proof['actual_terminal_receipt_sha256']):
        raise ValueError('Actual full training/source artifact audit required before reexecution')
    return proof, path


def worker(c, parent, case, identity):
    import importlib.metadata
    proof, proof_path = ready_proof(parent, case, identity)
    q = read(ROOT / case['queue'])
    canonical_q = read(ROOT / c['original_queue'])
    native.verify(canonical_q)
    job = case['job']
    if not native.complete(job):
        raise ValueError('Actual original full source experiment does not verify')
    source_result_path = ROOT / job['output_directory'] / 'result.json'
    source_result = read(source_result_path)
    for path, expected in [(job['frozen_samples'], job['data_sha256']),
                           (job['reference_path'], job['reference_sha256'])]:
        if sha(ROOT / path) != expected:
            raise ValueError('Full original frozen dataset bytes changed')
    policy = read(ROOT / canonical_q['protocol'])
    lock = read(ROOT / canonical_q['environment_lock'])
    for package, expected in lock['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Original native numerical environment changed: ' + package)
    out = Path(c['artifact_root']) / identity
    out.mkdir(parents=True, exist_ok=False)
    workspace = out / 'author_workspace'
    shutil.copytree(ROOT / policy['author_source'], workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(ROOT / 'src'))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from engine.solver import Trainer
    from engine.logger import Logger
    from utils.io_utils import seed_everything, instantiate_from_config
    from models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
    from iia_benchmark.models.fm_spectral_author import FrozenAuthorGenerationDataset
    torch.set_num_threads(2)
    torch.cuda.set_device(c['settings']['gpu_index'])
    seed_everything(job['seed'])
    config = yaml.safe_load((ROOT / job['author_yaml']).read_text(encoding='utf-8'))
    if config['solver']['max_epochs'] != job['training_updates']:
        raise ValueError('Original full architecture/training configuration required')
    arguments = SimpleNamespace(config=str(ROOT / job['author_yaml']), gpu=c['settings']['gpu_index'],
        name=identity + '__full_generation_reexecution_v1', output=str(out), tensorboard=False,
        seed=job['seed'], milestone=10, test_batch_size=job['generation_batch_size'], opts=None,
        save_dir=str(out / 'author_output'))
    logger = Logger(arguments)
    logger.save_config(config)
    model = instantiate_from_config(config['model']).cuda()
    if sum(p.numel() for p in model.parameters()) != proof['actual_model_parameters']:
        raise ValueError('Original complete trained architecture required')
    data = FrozenAuthorGenerationDataset(ROOT / job['frozen_samples'])
    loader = DataLoader(data, batch_size=config['dataloader']['batch_size'], shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    trainer = Trainer(config, arguments, model, {'dataloader': loader, 'dataset': data}, logger)
    checkpoint = Path(next(a['path'] for a in source_result['artifacts'] if a['role'] == 'checkpoint'))
    checkpoint_sha = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['step'] != job['training_updates'] or checkpoint.name != 'checkpoint-10.pt':
        raise ValueError('Exact completed original checkpoint required')
    trainer.results_folder = checkpoint.parent
    trainer.load(10)
    trainer.results_folder = out / 'author_output/checkpoints'
    for name, current in [('model', trainer.model.state_dict()), ('ema', trainer.ema.state_dict()), ('opt', trainer.opt.state_dict())]:
        assert_tree_equal(current, saved[name], torch)
    def forbidden_training(*args, **kwargs):
        raise RuntimeError('No trained generator optimizer update is permitted during reexecution')
    trainer.train = forbidden_training
    trainer.opt.step = forbidden_training
    new_seed = c['new_sampling_seed_offset'] + job['seed']
    seed_everything(new_seed)
    start_RNG = capture_RNG(torch, np)
    torch.save(start_RNG, out / 'complete_generation_start_RNG.pt')
    write(out / 'progress.json', dict(phase='first_full_generation', full_generation_batch=job['generation_batch_size']))
    started = time.perf_counter()
    first = trainer.sample(num=len(data), size_every=job['generation_batch_size'], shape=[data.window, data.var_num])
    first = unnormalize_to_zero_to_one(first)
    first_end_RNG = capture_RNG(torch, np)
    first_seconds = time.perf_counter() - started
    np.save(out / 'first_full_generated.npy', first, allow_pickle=False)
    saved_RNG = torch.load(out / 'complete_generation_start_RNG.pt', map_location='cpu', weights_only=False)
    if not same_RNG(start_RNG, saved_RNG, torch, np):
        raise ValueError('Saved complete sampling RNG changed on disk')
    restore_RNG(saved_RNG, torch, np)
    write(out / 'progress.json', dict(phase='second_full_generation_from_identical_saved_RNG'))
    started = time.perf_counter()
    second = trainer.sample(num=len(data), size_every=job['generation_batch_size'], shape=[data.window, data.var_num])
    second = unnormalize_to_zero_to_one(second)
    second_seconds = time.perf_counter() - started
    second_end_RNG = capture_RNG(torch, np)
    expected = [job['generation_batch_size'] * (len(data) // job['generation_batch_size'] + 1), *job['data_shape'][1:]]
    assert_same_samples(first, second, expected, np)
    if not same_RNG(first_end_RNG, second_end_RNG, torch, np):
        raise ValueError('Full replay consumed different RNG streams')
    np.save(out / 'second_full_generated.npy', second, allow_pickle=False)
    torch.save(dict(first=first_end_RNG, second=second_end_RNG), out / 'complete_generation_end_RNG.pt')
    for name, current in [('model', trainer.model.state_dict()), ('ema', trainer.ema.state_dict()), ('opt', trainer.opt.state_dict())]:
        assert_tree_equal(current, saved[name], torch)
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError('Original checkpoint bytes changed')
    reference = out / 'reference_data.npy'
    shutil.copyfile(ROOT / job['reference_path'], reference)
    if sha(reference) != job['reference_sha256']:
        raise ValueError('Complete original evaluation reference changed')
    parameter_count = sum(p.numel() for p in model.parameters())
    del second, saved, trainer, model, loader, data
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = original_evaluators()(out, policy, first, reference, torch, np, seed_everything)
    from scripts.flow_matching.audit_spectral_completed_generation_v1 import evaluator_statistics
    checked = evaluator_statistics(evaluation['metrics'])
    differences = {k: dict(original_mean=proof['metrics'][k]['mean'],
        new_generation_mean=v['mean'], difference=v['mean'] - proof['metrics'][k]['mean']) for k, v in checked.items()}
    files = ['first_full_generated.npy', 'second_full_generated.npy', 'reference_data.npy',
             'complete_generation_start_RNG.pt', 'complete_generation_end_RNG.pt']
    result = dict(passed=True, canonical_id=identity, model=job['model'], dataset=job['dataset'],
        data_track=job['data_track'], seed=job['seed'], separate_sampling_seed=new_seed,
        original_result_sha256=sha(source_result_path), original_artifact_proof_sha256=sha(proof_path),
        original_checkpoint_sha256=checkpoint_sha, original_checkpoint_unchanged=sha(checkpoint) == checkpoint_sha,
        original_source_and_frozen_dataset_verified=True, generator_parameters=parameter_count,
        actual_new_generator_training_updates=0, every_model_EMA_optimizer_state_unchanged=True,
        full_generation_shape=expected, all_generated_payload_bytes_replayed=True,
        complete_RNG_restoration_and_equal_consumption=True, full_original_sampling_calls=2,
        original_generation_batch=job['generation_batch_size'], generation_field_chunking=False,
        original_sampler_grid_and_solver_unchanged=True, historical_sampling_RNG_recovered=False,
        environment=dict(torch=torch.__version__, CUDA=torch.version.cuda, numpy=np.__version__,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic),
        historical_original_generated_bytes_equivalence_claim=False,
        first_generation_seconds=first_seconds, second_generation_seconds=second_seconds,
        evaluation_seconds=evaluation['evaluation_seconds'], all_four_original_evaluators_reexecuted=True,
        all_learned_evaluators_independently_retrained=True,
        original_evaluation_repeats=5, metrics=checked, original_vs_new_generation=differences,
        historical_author_equivalence_certified=False, strict_TAB_result=False,
        new_independent_training_seed_slots=0, all_original_experiments_complete=False,
        artifacts=[dict(path=str(out / n), sha256=sha(out / n)) for n in files])
    if not result['original_checkpoint_unchanged']:
        raise ValueError('Original model changed during evaluators')
    source.exclusive(out / 'full_generation_and_evaluator_reexecution.json', result)


def case_settings(c, job):
    settings = dict(c['settings'])
    if job['model'] == 'spectral_flow':
        settings['minimum_commit_headroom_bytes'] = max(settings['minimum_commit_headroom_bytes'],
            c['spectral_flow_minimum_commit_headroom_bytes'])
    return settings


def full_cohort_comparisons(cases, verified):
    groups = {}
    for identity, case in cases.items():
        j = case['job']
        key = '/'.join([j['model'], j['dataset'], j['data_track']])
        groups.setdefault(key, {})[j['seed']] = identity
    output = {}
    for key, planned in groups.items():
        present = {seed: verified[identity] for seed, identity in planned.items() if identity in verified}
        row = dict(verified_training_seeds=sorted(present), required_training_seeds=[42, 43, 44, 45, 46],
                   complete=set(present) == {42, 43, 44, 45, 46})
        if row['complete']:
            row['paired_training_seed_metric_differences'] = {}
            for metric in ['context_fid', 'correlational', 'discriminative', 'predictive']:
                values = [present[seed]['original_vs_new_generation'][metric]['difference'] for seed in [42, 43, 44, 45, 46]]
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                margin = 2.7764451051977987 * sd / math.sqrt(5)
                row['paired_training_seed_metric_differences'][metric] = dict(mean=mean, sample_SD=sd,
                    t95_interval=[mean - margin, mean + margin], df=4, new_generation_minus_original=True)
        output[key] = row
    if len(output) != 14:
        raise ValueError('All14 complete five-seed comparison groups required')
    return dict(cohorts=output, complete_cohorts=sum(v['complete'] for v in output.values()),
        interval_scope='Descriptive paired differences of five trained-model evaluator means on one frozen dataset; not historical RNG recovery or unseen-data generalization',
        all_original_experiments_complete=False)


def controller(c, parent, cases):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    from scripts.flow_matching.run_spectral_original_queue import guarded_api
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    install_cached_loader(lambda value: None)
    api = guarded_api()
    api['guarded_wait'] = resource_api()['guarded_wait']
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=False)
    out = Path(c['artifact_root'])
    out.mkdir(parents=True, exist_ok=False)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full70 generation/evaluator reexecution already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_conditions=70,
        required_cohorts=14, active_pid=None, verified_ids=[], failed_preserved={},
        new_generator_optimizer_updates=0, new_training_seed_slots=0,
        historical_original_generation_equivalence_claim=False, all_original_experiments_complete=False)
    verified = {}
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    update(dict(state='waiting_original_full_source_and_artifact_proofs'))
    try:
        while len(verified) + len(state['failed_preserved']) < 70:
            for identity, case in cases.items():
                if identity in verified or identity in state['failed_preserved']:
                    continue
                try:
                    if ready_proof(parent, case, identity) is None:
                        continue
                    settings = case_settings(c, case['job'])
                    update(dict(active_condition=identity))
                    with api['resource_slot'](ROOT / c['resource_lock'], settings, update):
                        gpu = api['gpu_snapshot'](settings)
                        if gpu['gpu_free_bytes'] < settings['minimum_gpu_free_bytes']:
                            update(dict(state='waiting_for_original_GPU_memory', **gpu))
                            continue
                        attempt = base / 'attempts' / identity
                        attempt.mkdir(parents=True, exist_ok=False)
                        command = [str(ROOT / c['python']), '-B', '-X', 'utf8', '-u', '-m', MODULE, '--case', identity]
                        with (attempt / 'stdout.log').open('xb') as stdout, (attempt / 'stderr.log').open('xb') as stderr:
                            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                    CUDA_VISIBLE_DEVICES=str(c['settings']['gpu_index'])),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            child_handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                            update(dict(state='running_full_generation_and_all_evaluators', active_pid=child.pid,
                                        exact_child_handle=child_handle))
                            code, usage = api['guarded_gpu_wait'](child, settings, api, update)
                        report_path = out / identity / 'full_generation_and_evaluator_reexecution.json'
                        passed = code == 0 and not usage['resource_guard_aborted'] and report_path.exists() and read(report_path)['passed']
                        source.exclusive(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code,
                            exact_child_handle=child_handle, **usage,
                            report_sha256=sha(report_path) if report_path.exists() else None))
                        if not passed:
                            raise ValueError('Full generation or original full evaluator rerun failed; preserve attempt without retry')
                        report = read(report_path)
                        for artifact in report['artifacts']:
                            if sha(artifact['path']) != artifact['sha256']:
                                raise ValueError('Full reexecuted generation/RNG artifact changed')
                        verified[identity] = report
                        state['verified_ids'].append(identity)
                        write(out / 'full14_cohort_comparisons.json', full_cohort_comparisons(cases, verified))
                except Exception as error:
                    state['failed_preserved'][identity] = repr(error)
                update(dict(active_pid=None, state='scanning_all70_full_source_proofs'))
            update(dict(state='waiting_original_full_source_and_artifact_proofs'))
            time.sleep(10)
        update(dict(state='all70_reexecution_attempts_terminal_other_original_experiments_pending'))
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        source.exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    parent, cases = verify(c)
    if args.probe_only:
        print(json.dumps(dict(required_conditions=len(cases), required_cohorts=14,
            evaluator_repeats=5, full_generations_each=2, training_updates_added=0,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    if args.case:
        worker(c, parent, cases[args.case], args.case)
    else:
        controller(c, parent, cases)


if __name__ == '__main__':
    main()
