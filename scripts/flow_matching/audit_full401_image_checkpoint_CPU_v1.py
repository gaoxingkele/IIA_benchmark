"""Verify the complete saved401-epoch model/Adam and CPU RNG states, no training."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import random
import sys

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts, validate_history
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def verify(config):
    verify_receipts(config['source_receipts'])
    verify_receipts([config['checkpoint_receipt']])
    old = read(ROOT / config['original_full25_configuration'])
    from scripts.flow_matching.run_full_image_training_recovery_queue_v3 import verify_registration
    registered = verify_registration(old)
    if config['job_configuration'] not in registered or len(registered) != 25:
        raise ValueError('All original20FM and5WT jobs and unchanged canonical seed required')
    reference = read(ROOT / config['reference_readonly_resource_configuration'])
    if config['settings'] != reference['settings']:
        raise ValueError('Existing complete-checkpoint CPU RAM/commit/private limits retained')
    job = read(ROOT / config['job_configuration'])
    if config['completed_epochs'] != 401 or config['required_original_epochs'] != 2000:
        raise ValueError('Keep all completed epochs and the complete original training budget')
    if Path(config['checkpoint_receipt']['path']).resolve() != (ROOT / job['output_directory'] / 'resume.pt').resolve():
        raise ValueError('Exact original rolling checkpoint required; never use probe weights')
    return job


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)): return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def worker(config, path):
    job = verify(config)
    environment = read(ROOT / job['environment_configuration'])
    if (sha(sys.executable) != environment['python_sha256'] or sys.version != environment['python_version']
            or any(importlib.metadata.version(name) != version for name, version in environment['packages'].items())):
        raise ValueError('Exact frozen original native checkpoint environment required')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    checkpoint = torch.load(config['checkpoint_receipt']['path'], map_location='cpu', weights_only=False)
    dataset = read(ROOT / job['dataset_configuration'])
    verify_receipts(dataset['source_receipts']); verify_receipts(dataset['array_receipts'])
    binding = dict(job_sha256=sha(ROOT / config['job_configuration']),
        dataset_configuration_sha256=sha(ROOT / job['dataset_configuration']),
        model_configuration_sha256=sha(ROOT / job['model_configuration']),
        optimizer_configuration_sha256=sha(ROOT / job['optimizer_configuration']))
    if checkpoint['binding'] != binding or checkpoint['seed'] != job['seed']:
        raise ValueError('Original job, raw-data, model, optimizer and seed binding required')
    n = job['expected_training_images_per_epoch']
    if checkpoint['completed_epochs'] != 401 or checkpoint['global_steps'] != 401*((n+31)//32):
        raise ValueError('Every complete original epoch and optimizer update required')
    validate_history(checkpoint['history'], 401, n)
    verify_receipts(checkpoint['epoch_receipts'])
    if len(checkpoint['epoch_receipts']) != 401:
        raise ValueError('All401 actual full-epoch receipts required')
    model_config = read(ROOT / job['model_configuration'])
    optimizer_config = read(ROOT / job['optimizer_configuration'])
    model = FlowMismatchUNet(**model_config['parameters'], asset_root=ROOT)
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer = build_verified_dinomaly_stableadamw(model.parameters(), **optimizer_config['parameters'], asset_root=ROOT)
    optimizer.load_state_dict(checkpoint['optimizer'])
    expected = next(row for row in environment['full_architectures'] if row['dataset'] == job['dataset'] and row['variant'] == job['variant'])
    if sum(value.numel() for value in model.parameters()) != expected['parameter_count']:
        raise ValueError('Every native full-backbone parameter required')
    if not equal(model.state_dict(), checkpoint['model']) or not equal(optimizer.state_dict(), checkpoint['optimizer']):
        raise ValueError('All original model and optimizer variables must restore bitwise')
    for value in model.state_dict().values():
        if not torch.isfinite(value).all(): raise ValueError('All original model tensors must be finite')
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Every native optimizer tensor must be finite')
    state = checkpoint['rng']
    if set(state) != {'python','numpy','torch_cpu','torch_cuda','order'} or len(state['torch_cuda']) != 1:
        raise ValueError('All original sampler/CPU/native CUDA/Python/NumPy states required')
    if any(value.device.type != 'cpu' or value.dtype != torch.uint8 for value in state['torch_cuda']):
        raise ValueError('Retain full original CUDA byte states without executing a GPU restore')
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'])
    order = torch.Generator(device='cpu'); order.set_state(state['order'])
    current = np.random.get_state()
    if (random.getstate() != state['python'] or current[0] != state['numpy'][0]
            or not np.array_equal(current[1], state['numpy'][1]) or current[2:] != state['numpy'][2:]
            or not torch.equal(torch.get_rng_state(), state['torch_cpu']) or not torch.equal(order.get_state(), state['order'])):
        raise ValueError('Every original CPU/Python/NumPy/order RNG state must restore bitwise')
    if torch.cuda.is_initialized(): raise ValueError('CPU audit must never initialize CUDA or contend with training GPU')
    verify_receipts([config['checkpoint_receipt']])
    output = ROOT / config['output_root']; output.mkdir(parents=True, exist_ok=False)
    exclusive(output / 'full401_native_model_optimizer_CPU_rng_restore.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), configuration=path, configuration_sha256=sha(ROOT/path),
        canonical_job=job, checkpoint_receipt=config['checkpoint_receipt'], full_native_parameters=expected['parameter_count'],
        completed_original_epochs=401, verified_original_optimizer_updates=checkpoint['global_steps'],
        required_original_epochs=2000, remaining_original_epochs=1599, complete_epoch_receipts_verified=401,
        all_model_and_StableAdamW_variables_bitwise_restored=True, all_model_and_optimizer_tensors_finite=True,
        Python_NumPy_native_CPU_and_order_RNG_states_bitwise_restored=True,
        saved_original_CUDA_generator_states_retained=1, CUDA_initialized=False, CUDA_RNG_restoration_verified=False,
        full_native_GPU_restore_and_training_continuation_still_required=True,
        original_full25_jobs_retained=True, new_optimizer_updates_executed=0, new_benchmark_results=0,
        original_author_equivalence_certified=False, all_original_experiments_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); worker(read(ROOT/args.config), args.config)
