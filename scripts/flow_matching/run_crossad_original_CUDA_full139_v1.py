"""Full139 original CrossAD conditions on an explicit, separate CUDA protocol.

All source data, budgets and author/TAB diagnostic metrics remain unchanged.
The original CPU queue remains intact. Failed attempts require isolated repair.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_crossad_original_cuda_first_full_v1 as first

MODULE = 'scripts.flow_matching.run_crossad_original_CUDA_full139_v1'
CONFIG = 'configs/experiments/fm_crossad_original_CUDA_full139.v1.json'
MODEL = 'configs/models/fm_crossad_original_CUDA_full139.v1.json'
PHASES = ('CUDA_probe', 'CPU_reference', 'author', 'replay')


def registration():
    base = read(ROOT / first.CONFIG)
    source = read(ROOT / first.preflight.QUEUE)
    jobs = []
    for original in source['jobs']:
        job = copy.deepcopy(original)
        canonical = job['id']
        job.update(id=canonical + '__original_CUDA_FP32_full139_v1',
                   canonical_CPU_condition_reference=canonical,
                   device_protocol='original_CUDA_FP32_no_activation_checkpoint',
                   activation_checkpointing=False, runner=MODULE,
                   output_directory='experiments/runs/fm_crossad_original_CUDA_full139_v1/jobs/' + canonical + '/author_job')
        jobs.append(job)
    # Fixed order independent of any outcome; finish one dataset before the next.
    order = {name: i for i, name in enumerate(['GECCO', 'MSL', 'SMAP', 'PSM', 'SMD', 'SWAN', 'SWAT'])}
    jobs.sort(key=lambda j: (order[j['dataset']], bool(j['reconstructed_ablation']),
                            j['mode'] == 'release_checkpoint', j['seed'], j.get('ablation_row', 0)))
    files = {r['path']: r['sha256'] for r in base['source_receipts']}
    files[first.CONFIG] = sha(ROOT / first.CONFIG)
    for job in source['jobs']:
        for item in job['frozen_files']:
            if item['path'] in files and files[item['path']] != item['sha256']:
                raise ValueError('Inconsistent original frozen source receipt')
            files[item['path']] = item['sha256']
    for name in [MODULE.replace('.', '/') + '.py', MODEL,
                 'tests/test_fm_crossad_original_CUDA_full139_v1.py',
                 'src/iia_benchmark/models/fm_crossad_ablation.py']:
        actual = sha(ROOT / name)
        if name in files and files[name] != actual:
            raise ValueError('Previously registered source may not be rebound: ' + name)
        files[name] = actual
    return dict(schema_version=1, jobs=jobs, required_original_conditions=139,
                original_queue=first.preflight.QUEUE,
                original_queue_sha256=sha(ROOT / first.preflight.QUEUE),
                source_receipts=[dict(path=p, sha256=h) for p, h in sorted(files.items())],
                settings=copy.deepcopy(base['settings']), host_guards=base['host_guards'],
                GPU=base['GPU'], cpu_threads=base['cpu_threads'], python=base['python'],
                resource_lock=base['resource_lock'], array_tolerance=base['array_tolerance'],
                cpu_reference_root=read(ROOT / first.preflight.CONFIG)['cpu_reference_root'],
                delegated_first_original_condition=base['canonical_CPU_condition_reference'],
                delegated_first_configuration=first.CONFIG,
                first_full_source_and_checkpoint_replay_required=True,
                state_root='experiments/runs/fm_crossad_original_CUDA_full139_v1',
                artifact_root='F:/aicoding/IIA_Data/experiments/crossad_original_CUDA_full139_v1',
                new_independent_training_seed_slots=0, original_CPU_conditions_replaced=False,
                reconstructed_ablation_author_equivalence=False,
                original_scope=base['original_scope'], strict_TAB_result=False,
                paper_equivalence=False, all_original_experiments_complete=False)


def verify(c):
    if c != registration():
        raise ValueError('All139 original recipes, seeds, sources, paths and resource gates required')
    for item in c['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen source changed: ' + item['path'])
    canonical = [j['canonical_CPU_condition_reference'] for j in c['jobs']]
    source = read(ROOT / c['original_queue'])['jobs']
    if len(canonical) != 139 or len(set(canonical)) != 139 or set(canonical) != {j['id'] for j in source}:
        raise ValueError('No main model, released checkpoint, seed or ablation may be omitted')


def case_paths(c, job):
    canonical = job['canonical_CPU_condition_reference']
    return ROOT / c['state_root'] / 'jobs' / canonical, Path(c['artifact_root']) / canonical


def build(c, job, cpu=False):
    from iia_benchmark.models.fm_crossad_author_runtime import build_model
    runtime = ROOT / (c['cpu_reference_root'] if cpu else c['settings']['runtime_root'])
    if job['reconstructed_ablation']:
        from iia_benchmark.models.fm_crossad_ablation import build_ablation
        return build_ablation(runtime, job['model_parameters'], job['ablation_row']).float()
    return build_model(runtime, job['model_parameters']).float()


def load_data(c, job, cpu=False):
    from iia_benchmark.models.fm_crossad_author_runtime import install_wide_cache
    runtime = ROOT / (c['cpu_reference_root'] if cpu else c['settings']['runtime_root'])
    install_wide_cache(runtime, ROOT / c['settings']['cache_root'])
    provider = __import__('importlib').import_module('data_provider.data_provider')
    window, batch = job['model_parameters']['seq_len'], job['train_parameters']['batch_size']
    data, loader = provider.data_provider(str(ROOT / c['settings']['raw_root']), job['dataset'], batch, window, 1, 'train')
    expected = int(job['train_length'] * .8) - window + 1
    if len(data) != expected or len(loader) != math.ceil(expected / batch):
        raise ValueError('Complete source windows and original last partial batch must be retained')
    return data, loader


def native_probe(c, job, cpu=False):
    import numpy as np
    import torch
    _, artifact = case_paths(c, job)
    out = artifact / 'full_native_preflight'
    if cpu:
        torch.set_num_threads(c['cpu_threads'])
        torch.manual_seed(job['seed'])
        report = read(out / 'CUDA.json')
        for name, key in [('input.npy', 'input_sha256'), ('initialization.pth', 'initialization_sha256'),
                          ('CUDA_scores.npy', 'scores_sha256')]:
            if sha(out / name) != report[key]:
                raise ValueError('Original complete probe data/state changed')
        model = build(c, job, cpu=True)
        model.load_state_dict(torch.load(out / 'initialization.pth', map_location='cpu', weights_only=False), strict=True)
        model.eval()
        values = np.load(out / 'input.npy', allow_pickle=False)
        expected = np.load(out / 'CUDA_scores.npy', allow_pickle=False)
        with torch.no_grad():
            actual = model.infer(torch.from_numpy(values), None, None, None)[0].numpy()
        equal = actual.shape == expected.shape and actual.dtype == expected.dtype
        if equal:
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            bound = c['array_tolerance']['atol'] + c['array_tolerance']['rtol'] * np.abs(expected.astype(np.float64))
            equal = bool(np.isfinite(actual).all() and np.all(difference <= bound))
        else:
            difference = bound = None
        np.save(out / 'CPU_scores.npy', actual, allow_pickle=False)
        exclusive(out / 'CPU_GPU.json', dict(passed=equal, all_entries=int(expected.size),
                  complete_native_shape=list(expected.shape), full_arrays_compared=True,
                  fixed_tolerance=c['array_tolerance'], maximum_absolute_error=float(difference.max()) if difference is not None else None,
                  maximum_normalized_error=float((difference / bound).max()) if bound is not None else None,
                  CPU_scores_sha256=sha(out / 'CPU_scores.npy'), CUDA_report_sha256=sha(out / 'CUDA.json'),
                  optimizer_updates=0, benchmark_performance=False))
        if not equal:
            raise ValueError('Complete CPU/GPU scores exceed unchanged fixed array tolerance')
        return
    torch, np = first.precision(dict(c, job=job))
    data, loader = load_data(c, job)
    batch, window = job['train_parameters']['batch_size'], job['model_parameters']['seq_len']
    values = np.stack([data[i][0] for i in range(batch)]).astype(np.float32)
    if values.shape != (batch, window, data.train.shape[1]):
        raise ValueError('Full original batch, sequence and channels required')
    model = build(c, job).to('cuda:0')
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / 'input.npy', values, allow_pickle=False)
    torch.save(model.state_dict(), out / 'initialization.pth')
    model.eval()
    x = torch.from_numpy(values).to('cuda:0')
    with torch.no_grad():
        a = model.infer(x, None, None, None)[0]
        b = model.infer(x, None, None, None)[0]
    if a.shape != x.shape or not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError('Complete source inference must be finite and repeat identically')
    np.save(out / 'CUDA_scores.npy', a.cpu().numpy(), allow_pickle=False)
    del a, b
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(x, None, None, None)[0]  # Exact original author train() objective.
    if not torch.isfinite(loss):
        raise ValueError('Original full-batch objective must remain finite')
    loss.backward()
    torch.cuda.synchronize()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients):
        raise ValueError('Actual source full-batch backward must remain finite FP32')
    if torch.cuda.max_memory_allocated() > c['GPU']['maximum_allocated_bytes']:
        raise ValueError('Registered original GPU cap exceeded')
    exclusive(out / 'CUDA.json', dict(passed=True, full_native_shape=list(values.shape),
              full_training_windows=len(data), full_batches_per_epoch=len(loader),
              model_parameters=sum(p.numel() for p in model.parameters()), gradient_tensors=len(gradients),
              finite_full_backward=True, inference_repeat_bitwise_equal=True, optimizer_updates=0,
              peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
              input_sha256=sha(out / 'input.npy'), initialization_sha256=sha(out / 'initialization.pth'),
              scores_sha256=sha(out / 'CUDA_scores.npy'), original_CUDA_source=True,
              reconstructed_ablation=job['reconstructed_ablation'], ablation_row=job.get('ablation_row'),
              benchmark_performance=False, full_training_still_required=True))


def source_or_replay(c, job, replay=False):
    torch, np = first.precision(dict(c, job=job))
    from scripts.flow_matching import run_crossad_author_job as original
    from iia_benchmark.models.fm_crossad_author_runtime import import_author, load_release
    state, artifact = case_paths(c, job)
    output = ROOT / job['output_directory']
    if not output.is_junction() or output.resolve() != artifact / 'author_job':
        raise ValueError('All new large artifacts require the fresh registered F-drive junction')
    run, scores = first.configured_functions(original)
    if not replay:
        if job['reconstructed_ablation']:
            module = import_author(ROOT / c['settings']['runtime_root'])
            module.Exp_Anomaly_Detection._build_model = lambda self: build(c, job)
        run(job, c['settings'])
        source = read(output / 'result.json')
        if source['details']['declared_train_parameters'] != job['train_parameters']:
            raise ValueError('Full original recipe must remain unchanged')
        exclusive(state / 'source_device_proof.json', dict(passed=True, source_result_sha256=sha(output / 'result.json'),
                  original_CUDA_source=True, activation_checkpointing=False, autocast=False, FP32_precision='ieee',
                  torch_version=torch.__version__, CUDA_build=torch.version.cuda,
                  peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
                  reconstructed_ablation_author_equivalence=False, new_independent_training_seed_slots=0))
        return
    source = read(output / 'result.json')
    if source['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
        raise ValueError('Original source condition identity changed')
    for item in source['artifacts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('A complete source artifact changed')
    data, loader = load_data(c, job)
    model = build(c, job).to('cuda:0')
    checkpoint = output / 'workspace/configs' / job['dataset'] / 'checkpoints_0/checkpoint.pth'
    loaded = load_release(model, checkpoint)
    model.eval()
    saved = np.load(output / 'native_scores.npz', allow_pickle=False)
    target = artifact / 'native_replay'
    target.mkdir(parents=True, exist_ok=False)
    proof, actual = [], {}
    for role, values in [('train', data.train), ('validation', data.val), ('test', data.test)]:
        score = scores(model, values, job['model_parameters']['seq_len'], job['train_parameters']['batch_size'])
        reference = saved[role]
        if score.shape != reference.shape or score.dtype != reference.dtype or score.tobytes() != reference.tobytes():
            raise ValueError('Every original full-population score must replay byte-for-byte')
        np.save(target / (role + '.npy'), score, allow_pickle=False)
        actual[role] = score
        proof.append(dict(role=role, points=len(score), all_score_bytes_equal=True,
                          score_file_sha256=sha(target / (role + '.npy'))))
    labels = data.test_label.reshape(-1).astype(np.uint8)
    if labels.tobytes() != saved['labels'].tobytes() or not np.array_equal(saved['source_rows'], np.arange(len(labels)) + job['train_length']):
        raise ValueError('Every source label and original row identity must match')
    detail = source['details']
    if detail['parameter_count'] != sum(p.numel() for p in model.parameters()):
        raise ValueError('All original algorithm parameters required')
    if job['mode'] == 'release_checkpoint':
        if detail['epochs_completed'] != 0 or detail['training_forward_calls'] != 0:
            raise ValueError('Published checkpoint test must not train a new model')
    elif (not 1 <= detail['epochs_completed'] <= job['train_parameters']['train_epochs'] or
          detail['training_forward_calls'] != detail['epochs_completed'] * len(loader)):
        raise ValueError('All original complete epoch batches and native earlystop must be retained')
    repeated = original.evaluate_saved({job['dataset']: (labels, actual['test'])}, actual['validation'], actual['train'],
               job['raw_sha256'], sha(checkpoint), c['settings']['evaluation'], job['seed'])
    first.compare_nested(source['metrics']['native_validation_control'], repeated, c['array_tolerance'])
    exclusive(target / 'report.json', dict(passed=True, id=job['id'], canonical_CPU_condition_reference=job['canonical_CPU_condition_reference'],
              source_result_sha256=sha(output / 'result.json'), checkpoint=loaded, all_roles=proof,
              all_labels_source_rows_and_tail_points_verified=True, actual_epochs=detail['epochs_completed'],
              full_batches_per_epoch=len(loader), training_forward_calls=detail['training_forward_calls'],
              all_native_validation_metrics_and_uncertainty_recomputed=True, native_validation_control=repeated,
              reconstructed_ablation=job['reconstructed_ablation'], author_equivalence=False,
              strict_TAB_result=False, new_independent_training_seed_slots=0, all_original_experiments_complete=False))


def first_completion(c):
    b = read(ROOT / c['delegated_first_configuration'])
    delegated = next(j for j in c['jobs'] if j['canonical_CPU_condition_reference'] == c['delegated_first_original_condition'])
    for key in ('model_parameters', 'train_parameters', 'seed', 'raw_sha256', 'train_length', 'mode'):
        if delegated[key] != b['job'][key]:
            raise ValueError('Delegated first proof must match the exact original condition')
    state = ROOT / b['state_root']
    status = read(state / 'status.json')
    if status['completed_conditions'] != 1:
        if status['state'] == 'failed_preserved_requires_isolated_repair':
            raise RuntimeError('First original CUDA attempt failed; preserve it and repair in isolation')
        return None
    proof = []
    for phase in ('train', 'replay'):
        path = state / (phase + '.guard_receipt.json')
        guard = read(path)
        if not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted'] or sha(guard['report_path']) != guard['report_sha256']:
            raise ValueError('First actual complete training and full fresh checkpoint replay must pass')
        proof.append(dict(phase=phase, guard_sha256=sha(path), report_sha256=guard['report_sha256']))
    return dict(canonical_CPU_condition_reference=c['delegated_first_original_condition'],
                actual_original_CUDA_condition_complete=True, actual_guards=proof, original_CPU_condition_marked_complete=False)


def controller(c):
    import psutil
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    state = ROOT / c['state_root']
    state.mkdir(parents=True, exist_ok=True)
    lock = exclusive_lock(state / 'controller.lock')
    if lock is None or (state / 'status.json').exists():
        if lock:
            lock.close()
        raise FileExistsError('Never overwrite or restart any full139 attempt')
    status = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
                  completed_original_CUDA_conditions=[], original_CPU_condition_completion_credit=0,
                  required_original_conditions=139, all_original_experiments_complete=False)
    def update(value):
        status.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(state / 'status.json', status)
    try:
        while (proof := first_completion(c)) is None:
            update(dict(state='waiting_actual_first_full_training_and_checkpoint_replay', dependency_configuration=c['delegated_first_configuration']))
            time.sleep(10)
        exclusive(state / 'delegated_first_full_completion.json', proof)
        status['completed_original_CUDA_conditions'].append(c['delegated_first_original_condition'])
        for job in c['jobs']:
            canonical = job['canonical_CPU_condition_reference']
            if canonical == c['delegated_first_original_condition']:
                continue
            verify(c)
            case, artifact = case_paths(c, job)
            case.mkdir(parents=True, exist_ok=False)
            artifact.mkdir(parents=True, exist_ok=False)
            output = ROOT / job['output_directory']
            target = artifact / 'author_job'
            target.mkdir(exist_ok=False)
            subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(output), str(target)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if not output.is_junction() or output.resolve() != target:
                raise ValueError('Fresh F-drive artifact junction failed')
            with resource_slot(ROOT / c['resource_lock'], c['host_guards'], update):
                while first.preflight.gpu_snapshot()['free_bytes'] < c['GPU']['minimum_free_bytes']:
                    update(dict(state='waiting_registered_VRAM', active_condition=canonical))
                    time.sleep(10)
                for phase in PHASES:
                    command = [str(ROOT / c['python']), '-B', '-X', 'utf8', '-u', '-m', MODULE,
                               '--phase', phase, '--canonical-id', canonical]
                    with (case / (phase + '.stdout.log')).open('xb') as stdout, (case / (phase + '.stderr.log')).open('xb') as stderr:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=dict(os.environ,
                           PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                           CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           OMP_NUM_THREADS=str(c['cpu_threads']), MKL_NUM_THREADS=str(c['cpu_threads'])))
                        handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                        update(dict(state='running_' + phase, active_condition=canonical, active_pid=child.pid, exact_child_handle=handle))
                        code, usage = first.preflight.guarded_GPU_wait(child, dict(GPU=c['GPU'], settings=c['host_guards']), update)
                    report = {'CUDA_probe': artifact / 'full_native_preflight/CUDA.json',
                              'CPU_reference': artifact / 'full_native_preflight/CPU_GPU.json',
                              'author': output / 'result.json', 'replay': artifact / 'native_replay/report.json'}[phase]
                    passed = code == 0 and not usage['resource_guard_aborted'] and report.exists()
                    if passed:
                        passed = read(report).get('passed', read(report).get('status') == 'completed')
                    exclusive(case / (phase + '.guard.json'), dict(passed=passed, exit_code=code, **usage,
                              report_path=str(report), report_sha256=sha(report) if report.exists() else None, exact_child_handle=handle))
                    update(dict(active_pid=None))
                    if not passed:
                        raise RuntimeError('Preserved full139 condition requires isolated repair: ' + canonical + '/' + phase)
            status['completed_original_CUDA_conditions'].append(canonical)
            update(dict(state='between_full_conditions'))
        update(dict(state='all139_original_CUDA_conditions_and_fresh_replays_complete_other_original_experiments_still_pending'))
    except Exception as error:
        update(dict(state='failed_preserved_requires_isolated_repair', error=repr(error)))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--canonical-id')
    args = parser.parse_args()
    if args.register:
        exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    verify(c)
    if args.phase:
        job = next(j for j in c['jobs'] if j['canonical_CPU_condition_reference'] == args.canonical_id)
        if args.phase in ('CUDA_probe', 'CPU_reference'):
            native_probe(c, job, cpu=args.phase == 'CPU_reference')
        else:
            source_or_replay(c, job, replay=args.phase == 'replay')
    else:
        controller(c)


if __name__ == '__main__':
    main()
