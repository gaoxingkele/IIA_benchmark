"""Execute full official CrossAD training/checkpoint tests and native-tail controls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.models.fm_crossad_author_runtime import import_author, install_wide_cache, load_release
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_experiment import verify_job, evaluate_saved


def native_scores(model, values, window, batch_size):
    import torch
    if len(values) < window:
        raise ValueError('Original CrossAD source segment is shorter than its window')
    blocks = len(values) // window
    chunks = []
    with torch.no_grad():
        for first in range(0, blocks, batch_size):
            batch = np.stack([values[i*window:(i+1)*window] for i in range(first, min(first+batch_size, blocks))])
            score, _ = model.infer(torch.as_tensor(batch, dtype=torch.float32), None, None, None)
            chunks.append(score.mean(dim=-1).cpu().numpy().reshape(-1))
        remainder = len(values) % window
        if remainder:
            score, _ = model.infer(torch.as_tensor(values[-window:][None], dtype=torch.float32), None, None, None)
            chunks.append(score.mean(dim=-1).cpu().numpy().reshape(-1)[-remainder:])
    scores = np.concatenate(chunks)
    if scores.shape != (len(values),) or not np.isfinite(scores).all():
        raise ValueError('Incomplete/nonfinite native CrossAD scores')
    return scores


def run(job, settings):
    import torch
    import pandas as pd
    torch.set_num_threads(settings['cpu_threads'])
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    verify_job(ROOT, job)
    output = ROOT / job['output_directory']
    if (output / 'result.json').exists() or (output / 'workspace').exists():
        raise FileExistsError('CrossAD previous artifacts preserved')
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / 'workspace'
    configs = workspace / 'configs' / job['dataset']
    configs.mkdir(parents=True)
    for name, data in [('model_configs_0.json', job['model_parameters']), ('train_configs.json', job['train_parameters'])]:
        write_json(configs / name, data)
    module = import_author(ROOT / settings['runtime_root'])
    cache_audits = install_wide_cache(ROOT / settings['runtime_root'], ROOT / settings['cache_root'])
    args = SimpleNamespace(configs_path=str(workspace/'configs'), save_path=str(workspace/'test_results'),
             root_path=str(ROOT/settings['raw_root']), data=job['dataset'], data_origin='DADA',
             use_gpu=False, gpu_type='cpu', use_multi_gpu=False, gpu=0, devices='0',
             device=torch.device('cpu'), metrics=settings['author_metrics'], method='spot', t=settings['spot_q'])
    exp = module.Exp_Anomaly_Detection(args, id=0)
    # Preserve source torch.load semantics while making author CUDA archives CPU-readable.
    original_load = torch.load
    def cpu_load(*args, **kwargs):
        kwargs.setdefault('map_location', 'cpu'); kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = cpu_load
    losses, validation_losses = [], []
    def capture(module, inputs, outputs):
        if module.training:
            value = float(outputs[0].detach())
            if not np.isfinite(value):
                raise ValueError('Nonfinite CrossAD author training loss')
            losses.append(value)
    hook = exp.model.register_forward_hook(capture)
    original_vali = exp.vali
    def capture_vali(loader):
        value = original_vali(loader)
        validation_losses.append(float(value))
        write_json(output/'training_trace.json', {'training_forward_calls': len(losses), 'validation_losses': validation_losses,
                    'epochs_finished': len(validation_losses), 'declared_max_epochs': job['train_parameters']['train_epochs']})
        return value
    exp.vali = capture_vali
    started = time.monotonic()
    checkpoint = Path(exp.model_save_path) / 'checkpoint.pth'
    if job['mode'] == 'release_checkpoint':
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / job['release_checkpoint'], checkpoint)
        weight_audit = load_release(exp.model, checkpoint)
        training_seconds = 0.0
    else:
        exp.train()
        training_seconds = time.monotonic()-started
        weight_audit = {'author_best_checkpoint_reloaded': True, 'checkpoint_sha256': sha(checkpoint),
                        'framework_default_strict_train_load': True, 'epochs_completed': len(validation_losses)}
    hook.remove()
    stages = output / 'stages'
    def stage(name, details, paths):
        artifacts = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]
        write_json(stages/name/'receipt.json', {'status': 'completed', 'details': details, 'artifacts': artifacts})
    stage('weights', weight_audit, [checkpoint])
    started = time.monotonic()
    exp.test()  # Complete original source inference and default oracle/range/VUS metrics.
    author_test_seconds = time.monotonic()-started
    result_dir = Path(exp.rst_save_path)
    stage('author_test', {'seconds': author_test_seconds, 'metrics': args.metrics}, sorted(result_dir.glob('*.npy'))+sorted(result_dir.glob('*.csv')))
    exp.evaluate_spot(t=settings['spot_q'])  # Original default-q SPOT and affiliation, separately recorded.
    stage('author_spot', {'q': settings['spot_q']}, sorted(result_dir.glob('*affiliation*.csv'))+sorted(result_dir.glob('*spot*.csv')))
    model = exp.model.eval()
    train_set, _ = exp._get_data('train')
    roles = [('train', train_set.train), ('validation', train_set.val), ('test', train_set.test)]
    started = time.monotonic()
    scores = {role: native_scores(model, values, job['model_parameters']['seq_len'], job['train_parameters']['batch_size']) for role, values in roles}
    labels = train_set.test_label.reshape(-1).astype(np.uint8)
    original_scores = np.load(result_dir/'a_test_energy.npy').mean(axis=-1)
    if not np.array_equal(original_scores, scores['test'][:len(original_scores)]):
        # Torch channel-mean and numpy channel-mean may round differently.
        if not np.allclose(original_scores, scores['test'][:len(original_scores)], rtol=1e-6, atol=1e-7):
            raise ValueError('Native control differs from original complete-window source inference')
    scores_path = output / 'native_scores.npz'
    np.savez_compressed(scores_path, **scores, labels=labels, source_rows=np.arange(len(labels))+job['train_length'])
    native_score_seconds = time.monotonic()-started
    strict = evaluate_saved({job['dataset']: (labels, scores['test'])}, scores['validation'], scores['train'],
                            job['raw_sha256'], sha(checkpoint), settings['evaluation'], job['seed'])
    author_metrics = {p.stem: pd.read_csv(p).to_dict(orient='records') for p in result_dir.glob('*.csv')}
    details = {'algorithm_weights': weight_audit, 'native_points': {r: len(v) for r, v in roles},
               'retained_channels': train_set.train.shape[1], 'original_scored_test_points': len(original_scores),
               'original_unscored_tail_points': len(labels)-len(original_scores),
               'native_tail_policy': 'Preserve original nonoverlap scores, append only tail positions from end-aligned full window',
               'source_scaler_fit_boundary': 'All author nominal training rows, including later validation portion',
               'training_stride': 1, 'declared_train_parameters': job['train_parameters'],
               'epochs_completed': len(validation_losses), 'training_forward_calls': len(losses),
               'training_seconds': training_seconds, 'author_test_seconds': author_test_seconds,
               'native_score_seconds': native_score_seconds, 'parameter_count': sum(p.numel() for p in model.parameters()),
               'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    write_json(output/'execution_details.json', details)
    stage('native_control', details, [scores_path, output/'execution_details.json'])
    artifacts = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in
                 [checkpoint, scores_path, output/'execution_details.json']+sorted(result_dir.glob('*.npy'))+sorted(result_dir.glob('*.csv'))]
    receipt = {'id': job['id'], 'status': 'completed', 'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
               'dataset': job['dataset'], 'seed': job['seed'], 'mode': job['mode'], 'artifacts': artifacts,
               'metrics': {'original_author': author_metrics, 'native_validation_control': strict},
               'details': details, 'boundary': settings['boundary'], 'strict_TAB_result': False, 'paper_equivalence': False}
    write_json(output/'result.json', receipt)
    print(json.dumps({'id': job['id'], 'status': 'completed', 'epochs_completed': len(validation_losses)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--queue', type=Path, required=True); parser.add_argument('--job-id', required=True)
    args = parser.parse_args(); queue = json.loads(args.queue.read_text(encoding='utf-8'))
    run(next(j for j in queue['jobs'] if j['id'] == args.job_id), queue['settings'])
