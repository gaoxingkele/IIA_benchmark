"""Run the pinned full-width author implementation in an isolated CPU workspace.

Original source, datasets and existing smaller local tracks are never modified.
Source test-set checkpoint selection, pooled calibration, PA and tail dropping
are deliberately retained and disclosed. This is not the strict TAB track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.next.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned source patch does not match exactly once: ' + old[:90])
    return source.replace(old, new, 1)


def patched(relative, source):
    """Only device/API compatibility and append-only experiment observations."""
    if relative == 'model/attn.py':
        source = replace_once(source, 'torch.zeros((window_size, window_size)).cuda()',
                              'torch.zeros((window_size, window_size))')
        source = replace_once(source, 'sigma.shape[1], 1, 1).cuda()',
                              'sigma.shape[1], 1, 1).to(queries.device)')
    elif relative == 'solver.py':
        if source.count('np.Inf') != 2:
            raise ValueError('Unexpected infinity compatibility sites')
        source = source.replace('np.Inf', 'np.inf')
        source = replace_once(source, '        train_steps = len(self.train_loader)',
                              '        self.observed_epochs = []\n        train_steps = len(self.train_loader)')
        source = replace_once(source, '            early_stopping(vali_loss1, vali_loss2, self.model, path)',
                              '            self.observed_epochs.append(dict(epoch=epoch+1, steps=train_steps, '
                              'train_loss=float(train_loss), test_selection_loss1=float(vali_loss1), '
                              'test_selection_loss2=float(vali_loss2)))\n'
                              '            early_stopping(vali_loss1, vali_loss2, self.model, path)')
        source = replace_once(source, '        print("Threshold :", thresh)',
                              '        threshold_energy = test_energy.copy()\n        print("Threshold :", thresh)')
        source = replace_once(source, '        gt = test_labels.astype(int)',
                              '        gt = test_labels.astype(int)\n        predictions_no_pa = pred.copy()')
        source = replace_once(source, '        return accuracy, precision, recall, f_score',
                              '        np.savez(os.path.join(self.model_save_path, "author_scores.npz"), '
                              'train_energy=train_energy, threshold_energy=threshold_energy, '
                              'test_energy=test_energy, labels=gt, predictions_no_pa=predictions_no_pa, '
                              'predictions_author_pa=pred, threshold=np.asarray(thresh))\n'
                              '        return accuracy, precision, recall, f_score')
    return source


def verify(config, root=ROOT):
    if config['required_model'] != dict(d_model=512, d_ff=512, n_heads=8, e_layers=3,
                                         dropout=0.0, window=100, batch_size=256, k=3,
                                         learning_rate=0.0001, temperature=50):
        raise ValueError('Full original architecture/training/scoring parameters required')
    if config['seeds'] != [1103, 1104, 1105, 1106, 1107] or len(config['jobs']) != 25:
        raise ValueError('All five datasets and all five registered seeds must remain')
    expected = {(d, s) for d in ('SMD', 'MSL', 'SMAP', 'PSM', 'SWaT') for s in config['seeds']}
    if {(j['dataset'], j['seed']) for j in config['jobs']} != expected:
        raise ValueError('Incomplete original dataset/seed matrix')
    if (config['settings']['minimum_available_memory_bytes'] < 32*1024**3
            or config['settings']['minimum_commit_headroom_bytes'] < 40*1024**3
            or config['settings']['emergency_commit_headroom_bytes'] < 16*1024**3):
        raise ValueError('Original full-author admission/emergency memory protection required')
    for pin in config['source_receipts']:
        if sha(root / pin['path']) != pin['sha256']:
            raise ValueError('Frozen source changed: ' + pin['path'])
    for job in config['jobs']:
        if job['id'] != f"anomaly_transformer_author__full_width__{job['dataset'].lower()}__seed{job['seed']}":
            raise ValueError('Canonical original full-width identity changed')
        if job['epochs'] != (10 if job['dataset'] == 'SMD' else 3):
            raise ValueError('Author maximum epochs changed')
        expected_ratio = 0.5 if job['dataset']=='SMD' else (0.1 if job['dataset']=='SWaT' else 1.0)
        if job['anomaly_ratio'] != expected_ratio:
            raise ValueError('Author per-dataset anomaly ratio changed')
        if job['dataset_config'] != f"configs/datasets/mtsad_{job['dataset'].lower()}.json":
            raise ValueError('Original dataset configuration binding changed')
        if job['status'] != ('source_loader_unavailable' if job['dataset'] == 'SWaT' else 'registered_full_author_runtime'):
            raise ValueError('Unreviewed source loader or unsupported status')
    return True


def check_result(config, job):
    base = Path(config['artifact_root']) / job['id']
    result = read(base / 'result.json')
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if result['status'] != 'completed' or result['experiment_sha256'] != fingerprint:
        raise ValueError('Existing result has a different identity')
    for record in result['artifacts']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Existing full-author artifact changed')
    return result


def run_worker(config, job, config_path):
    verify(config)
    if job['status'] != 'registered_full_author_runtime':
        raise ValueError('Original snapshot does not implement the SWaT loader')
    base = Path(config['artifact_root']) / job['id']
    # Never replace interrupted numerical runs, raw files, or existing artifacts.
    base.mkdir(parents=True, exist_ok=False)
    workspace = base / 'source'
    original = ROOT / config['source_directory']
    shutil.copytree(original, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patches = []
    for relative in ('model/attn.py', 'solver.py'):
        path = workspace / relative
        before = sha(path)
        transformed = patched(relative, path.read_text(encoding='utf-8'))
        path.write_text(transformed, encoding='utf-8', newline='\n')
        patches.append(dict(path=relative, original_sha256=before, runtime_sha256=sha(path)))
    write(base / 'runtime_patch_manifest.json', dict(patches=patches,
          architecture_and_loss_unchanged=True, source_selection_and_calibration_unchanged=True))
    import random
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_recall_fscore_support
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    if torch.cuda.is_available():
        raise ValueError('This frozen reproduction track is CPU only')
    sys.path.insert(0, str(workspace))
    from solver import Solver
    dataset_config = read(ROOT / job['dataset_config'])
    data_path = (ROOT / config['data_root'] / dataset_config['directory']).resolve()
    data_audit = []
    for name in dataset_config['files'].values():
        path = data_path / name
        data_audit.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    solver = Solver(dict(lr=0.0001, num_epochs=job['epochs'], k=3, win_size=100,
                         input_c=dataset_config['features'], output_c=dataset_config['features'],
                         batch_size=256, dataset=job['dataset'], data_path=str(data_path),
                         model_save_path=str(base / 'checkpoints'), anormly_ratio=job['anomaly_ratio'],
                         pretrained_model=None))
    if solver.device.type != 'cpu':
        raise ValueError('Unexpected numerical device')
    expected_shapes = dataset_config['expected_shapes']
    actual_shapes = dict(train=list(solver.train_loader.dataset.train.shape),
                         test=list(solver.test_loader.dataset.test.shape))
    if actual_shapes != expected_shapes:
        raise ValueError('Source data are incomplete or changed')
    model = solver.model
    if model.embedding.value_embedding.tokenConv.out_channels != 512:
        raise ValueError('Author model width changed')
    parameter_count = sum(p.numel() for p in model.parameters())
    updates = [0]
    hook = solver.optimizer.register_step_post_hook(lambda *_args: updates.__setitem__(0, updates[0]+1))
    started = time.perf_counter()
    solver.train()
    training_seconds = time.perf_counter() - started
    hook.remove()
    epochs = solver.observed_epochs
    if not epochs or updates[0] != sum(e['steps'] for e in epochs):
        raise ValueError('Actual full training step coverage is unproven')
    if len(epochs) > job['epochs'] or (len(epochs) < job['epochs'] and len(epochs) < 4):
        raise ValueError('Training stopped before the original patience rule permits')
    started = time.perf_counter()
    author_metrics = solver.test()
    score_seconds = time.perf_counter() - started
    scorepath = base / 'checkpoints/author_scores.npz'
    with np.load(scorepath, allow_pickle=False) as arrays:
        scores, labels = arrays['test_energy'], arrays['labels']
        expected_test_points = actual_shapes['test'][0] // 100 * 100
        expected_train_points = len(solver.train_loader.dataset) * 100
        if len(scores) != expected_test_points or len(arrays['train_energy']) != expected_train_points:
            raise ValueError('Full original score coverage missing')
        if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
            raise ValueError('Nonfinite original scores')
        threshold = float(np.percentile(np.concatenate([arrays['train_energy'], arrays['threshold_energy']]),
                                        100-job['anomaly_ratio']))
        if threshold != float(arrays['threshold']) or not np.array_equal(arrays['predictions_no_pa'], (scores > threshold).astype(int)):
            raise ValueError('Author pooled calibration/prediction mismatch')
        def metrics(pred):
            p, r, f, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
            return dict(accuracy=float(accuracy_score(labels, pred)), precision=float(p), recall=float(r), f1=float(f))
        adjusted, unadjusted = metrics(arrays['predictions_author_pa']), metrics(arrays['predictions_no_pa'])
        if not all(math.isclose(a, adjusted[k], abs_tol=1e-12, rel_tol=1e-12)
                   for a, k in zip(author_metrics, ('accuracy', 'precision', 'recall', 'f1'))):
            raise ValueError('Author metric output differs from full saved predictions')
        has_both = len(np.unique(labels)) == 2
        ranking = dict(auroc=float(roc_auc_score(labels, scores)) if has_both else None,
                       average_precision=float(average_precision_score(labels, scores)) if has_both else None)
    checkpoint = base / 'checkpoints' / (job['dataset'] + '_checkpoint.pth')
    artifacts = [checkpoint, scorepath, base / 'runtime_patch_manifest.json']
    write(base / 'result.json', dict(status='completed', job=job,
          experiment_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
          configuration_sha256=sha(config_path), parameter_count=parameter_count,
          epochs=epochs, optimizer_updates=updates[0], training_seconds=training_seconds,
          score_seconds=score_seconds, data_audit=data_audit, actual_shapes=actual_shapes,
          train_score_points=expected_train_points, test_score_points=expected_test_points,
          dropped_test_tail_points=actual_shapes['test'][0]-expected_test_points,
          author_PA=adjusted, no_PA_same_author_threshold=unadjusted, ranking=ranking,
          source_test_features_used_for_checkpoint_selection=True,
          source_test_scores_used_for_threshold_calibration=True,
          historical_paper_equivalence_certified=False, strict_TAB_result=False,
          full_independent_checkpoint_replay_complete=False,
          artifacts=[dict(path=str(p), sha256=sha(p)) for p in artifacts],
          boundary=config['boundary']))


def controller(config, config_path):
    from scripts.flow_matching.heavy_resources import guarded_wait
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    import psutil
    verify(config)
    state_root = ROOT / config['state_root']
    lock = exclusive_lock(state_root / 'controller.lock')
    if lock is None:
        raise RuntimeError('Full-author lane already has an owner')
    common = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                  required_original_jobs=25, all_original_experiments_complete=False)
    def update(value):
        write(state_root / 'status.json', dict(common, **value))
    for job in config['jobs']:
        if job['status'] == 'source_loader_unavailable':
            write(state_root / 'unavailable' / (job['id']+'.json'),
                  dict(job=job, reason='Pinned original source has no SWaT loader; no replacement certified', completed=False))
            continue
        base = Path(config['artifact_root']) / job['id']
        receipt = state_root / 'receipts' / (job['id']+'.json')
        if (base / 'result.json').exists():
            check_result(config, job)
            continue
        if base.exists() or receipt.exists():
            update(dict(state='preserved_prior_attempt_requires_review', active_pid=None, job=job['id']))
            continue
        def waiting(value):
            update(dict(value, job=job['id'], active_pid=None))
        with resource_slot(ROOT / config['resource_lock'], config['settings'], waiting, poll_seconds=10):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                               OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']))
            state_root.mkdir(parents=True, exist_ok=True)
            with (state_root / (job['id']+'.log')).open('xb') as log:
                process = subprocess.Popen([config['python'], '-B', '-X', 'utf8', '-u', '-m',
                     'scripts.flow_matching.run_anomaly_transformer_author_full_v1', '--config', config_path,
                     '--worker', '--job-id', job['id']], cwd=ROOT, env=environment,
                     stdout=log, stderr=subprocess.STDOUT)
                update(dict(state='running_full_author_original', job=job['id'], active_pid=process.pid,
                            active_create_time=psutil.Process(process.pid).create_time()))
                code, peaks = guarded_wait(process, config['settings'], poll_seconds=2)
            passed = code == 0 and not peaks['resource_guard_aborted']
            if passed:
                check_result(config, job)
            write(receipt, dict(job=job['id'], exit_code=code, passed=passed, **peaks))
    completed = []
    for job in config['jobs']:
        receipt = state_root / 'receipts' / (job['id']+'.json')
        if receipt.exists() and read(receipt).get('passed'):
            check_result(config, job)
            completed.append(job['id'])
    update(dict(state='registered_matrix_attempts_terminal_material_and_review_gaps_remain', active_pid=None,
                completed_original_models=completed,
                missing_source_loader_jobs=[j['id'] for j in config['jobs'] if j['status']=='source_loader_unavailable']))
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, registered_full_width_jobs=25,
                              source_loader_available_jobs=20, missing_SWaT_source_loader_jobs=5,
                              torch_imported='torch' in sys.modules, all_original_experiments_complete=False)))
    elif args.worker:
        job = next(j for j in config['jobs'] if j['id'] == args.job_id)
        run_worker(config, job, ROOT / args.config)
    else:
        controller(config, args.config)


if __name__ == '__main__':
    main()
