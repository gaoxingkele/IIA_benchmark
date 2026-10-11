"""Every native test image, per-category paper metrics, immutable score attempt."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT, read, sha, validate_history, verify_receipts,
)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def context(queue, case):
    """Validate original full checkpoint and model/data bindings before inference."""
    verify_receipts(queue['source_receipts'])
    training = read(ROOT / case['training_configuration'])
    verify_receipts(training['source_receipts'])
    dataset = read(ROOT / training['dataset_configuration'])
    verify_receipts(dataset['source_receipts']); verify_receipts(dataset['array_receipts'])
    environment = read(ROOT / queue['environment_configuration'])
    if (sha(sys.executable) != environment['python_sha256']
            or sys.version != environment['python_version']
            or any(importlib.metadata.version(p) != v for p, v in environment['packages'].items())):
        raise ValueError('Frozen full evaluation environment differs')
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Registered GPU evaluator required, no CPU substitution')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    checkpoint_path = ROOT / case['checkpoint']
    checkpoint_sha = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    binding = dict(job_sha256=sha(ROOT / case['training_configuration']),
        dataset_configuration_sha256=sha(ROOT / training['dataset_configuration']),
        model_configuration_sha256=sha(ROOT / training['model_configuration']),
        optimizer_configuration_sha256=sha(ROOT / training['optimizer_configuration']))
    n = training['expected_training_images_per_epoch']
    if (checkpoint['binding'] != binding or checkpoint['seed'] != case['seed']
            or checkpoint['completed_epochs'] != case['checkpoint_epoch']
            or checkpoint['global_steps'] != case['checkpoint_epoch'] * ((n + 31) // 32)):
        raise ValueError('Full native original-budget checkpoint identity differs')
    validate_history(checkpoint['history'], case['checkpoint_epoch'], n)
    verify_receipts(checkpoint['epoch_receipts'])
    model_config = read(ROOT / training['model_configuration'])
    model = FlowMismatchUNet(**model_config['parameters'], asset_root=ROOT)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.to(device).eval()
    derived_path = next(x['path'] for x in dataset['source_receipts']
                        if Path(x['path']).name == 'full_derived_image_preprocessing.json')
    names = read(ROOT / derived_path)['original_categories']
    by_field = {x['field']: x for x in dataset['array_receipts'] if x['split'] == 'test'}
    arrays = {field: np.load(receipt['path'], mmap_mode='r', allow_pickle=False)
              for field, receipt in by_field.items()}
    expected = dataset['expected_native_splits']['test']
    if (set(arrays) != {'images', 'masks', 'labels', 'categories'}
            or arrays['images'].shape != (expected, 3, 128, 128)
            or arrays['masks'].shape != (expected, 128, 128)
            or arrays['labels'].shape != (expected,) or arrays['categories'].shape != (expected,)
            or not np.isin(arrays['labels'], (0, 1)).all()
            or set(np.unique(arrays['categories'])) != set(range(len(names)))):
        raise ValueError('Every complete native original test array and category required')
    return model, arrays, names, binding, checkpoint_sha, torch, np, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--case-id', required=True)
    parser.add_argument('--attempt-directory', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    case = next(x for x in queue['cases'] if x['id'] == args.case_id)
    output = (ROOT / case['output_directory']).resolve()
    attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output / 'attempts') or not attempt.is_dir():
        raise ValueError('Fresh preserved evaluation attempt directory required')
    model, arrays, names, binding, checkpoint_sha, torch, np, device = context(queue, case)
    from iia_benchmark.models.fm_image_evaluation import paired_probes
    from iia_benchmark.models.fm_supplemental import flow_mismatch
    from scripts.flow_matching.flow_mismatching_image_metrics_v1 import per_category_metrics
    n = len(arrays['images']); parameters = case['score_parameters']
    paths = parameters['paths']; times = parameters['times']
    heatmap_path = attempt / 'complete_native_test_heatmaps.npy'
    score_path = attempt / 'complete_native_test_image_scores.npy'
    probe_path = attempt / 'complete_native_test_probe_seeds.npy'
    heatmaps = np.lib.format.open_memmap(heatmap_path, mode='w+', dtype=np.float32, shape=(n, 128, 128))
    scores = np.lib.format.open_memmap(score_path, mode='w+', dtype=np.float32, shape=(n,))
    probes = np.lib.format.open_memmap(probe_path, mode='w+', dtype=np.int64, shape=(n, paths))
    elapsed = []
    with torch.inference_mode():
        for index in range(n):
            image = torch.from_numpy(np.array(arrays['images'][index:index + 1], copy=True)).to(device)
            labels = (torch.tensor([int(arrays['categories'][index])], device=device)
                      if case['variant'] == 'conditional' else None)
            noise, ids = paired_probes(image, training_seed=case['seed'], dataset=case['dataset'],
                                      native_image_index=index, paths=paths)
            if index == 0:
                # Explicit local full-first-image warm-up, never omitted from actual coverage.
                flow_mismatch(model, image, labels=labels, seeds=noise, **parameters)
            torch.cuda.synchronize(device); start = time.perf_counter()
            heat, score = flow_mismatch(model, image, labels=labels, seeds=noise, **parameters)
            torch.cuda.synchronize(device); elapsed.append(time.perf_counter() - start)
            if not torch.isfinite(heat).all() or not torch.isfinite(score).all():
                raise FloatingPointError('Nonfinite native full-image score')
            heatmaps[index] = heat[0].cpu().numpy(); scores[index] = float(score[0]); probes[index] = ids
            if (index + 1) % 100 == 0:
                print(json.dumps(dict(case_id=case['id'], full_test_images_scored=index + 1, expected=n)), flush=True)
    heatmaps.flush(); scores.flush(); probes.flush()
    metrics = per_category_metrics(heatmaps, scores, arrays['masks'], arrays['labels'], arrays['categories'],
        names, fpr_limit=queue['metrics']['aupro_fpr_limit'], connectivity=queue['metrics']['region_connectivity'])
    elapsed = np.asarray(elapsed, dtype=np.float64)
    result = dict(case=case, queue_configuration=args.queue, queue_sha256=sha(ROOT / args.queue),
        captured_utc=datetime.now(timezone.utc).isoformat(), training_binding=binding,
        original_checkpoint_sha256=checkpoint_sha, complete_original_test_images=n,
        theoretical_test_forward_evaluations=n * paths * times,
        full_test_batched_velocity_calls=n * times, additional_full_first_image_warmup_calls=times,
        native_test_order='complete original derived array order; no subset or reorder',
        original_model_parameters=sum(p.numel() for p in model.parameters()),
        original_full_state_dictionary_loaded_strictly=True, local_metrics=metrics,
        timing=dict(local_gpu_FPS=float(n / elapsed.sum()),
            complete_test_forward_seconds=float(elapsed.sum()), median_image_seconds=float(np.median(elapsed)),
            p95_image_seconds=float(np.quantile(elapsed, .95)), gpu=torch.cuda.get_device_name(),
            precision='FP32 parameters/inputs with TF32; original inference precision unreported',
            includes='synchronized full scorer GPU forwards, residuals, reductions and image score',
            excludes='probe generation, transfers, model load, metrics, IO and warm-up',
            original_A100_80GB_efficiency_equivalence_certified=False),
        output_receipts=[dict(path=str(p), sha256=sha(p)) for p in (heatmap_path, score_path, probe_path)],
        native_truth_array_receipts=[x for x in read(ROOT / read(ROOT / case['training_configuration'])['dataset_configuration'])['array_receipts'] if x['split'] == 'test'],
        full_independent_metric_and_model_replay_passed=False, original_author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False,
        boundary='Full native test scoring candidate. Independent every-image model replay and original-truth metric audit still required; paired probe and AUPRO/inference choices are disclosed local conventions.')
    write_new(attempt / 'full_native_score_candidate.json', result)
    write_new(output / 'score_candidate_location.json', dict(candidate=str(attempt / 'full_native_score_candidate.json'),
        candidate_sha256=sha(attempt / 'full_native_score_candidate.json')))
    print(json.dumps(dict(case_id=case['id'], full_test_images=n, all_categories=len(names), full_audit_pending=True)))


if __name__ == '__main__':
    main()
