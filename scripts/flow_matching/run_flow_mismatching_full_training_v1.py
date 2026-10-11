"""Full 2,000-epoch local-choice replication, never a synthetic benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import uuid

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT, read, sha, validate_history, validate_job, verify_receipts,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def capture_rng(torch, np, order):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                order=order.get_state())


def restore_rng(state, torch, np, order):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])
    order.set_state(state['order'].cpu())


def save_checkpoint(path, value, torch):
    """Two rolling epoch slots plus immutable 100-epoch snapshots, raw untouched."""
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        previous = path.with_name('resume_previous.pt')
        previous_temp = path.with_name('resume_previous.' + uuid.uuid4().hex + '.tmp')
        os.link(path, previous_temp)
        os.replace(previous_temp, previous)
    os.replace(temporary, path)


def train_epoch(model, optimizer, images, categories, *, conditional, batch_size,
                order, device, bf16, torch, np, objective):
    """No tail drop: each original normal training row occurs exactly once."""
    indices = torch.randperm(len(images), generator=order).numpy().astype('<i8')
    if not np.array_equal(np.sort(indices), np.arange(len(images))):
        raise ValueError('Training sampler is not a complete permutation')
    loss_sum, steps = 0.0, 0
    model.train()
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        batch = torch.from_numpy(np.array(images[rows], dtype=np.float32, copy=True)).to(device)
        labels = (torch.from_numpy(np.array(categories[rows], dtype=np.int64, copy=True)).to(device)
                  if conditional else None)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = objective(model, batch, labels=labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite original CFM training loss')
        loss.backward()
        # No extra gradient norm clipping, loss scaling, EMA, scheduler or accumulation.
        if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None]).all():
            raise FloatingPointError('Nonfinite original CFM gradient')
        optimizer.step()
        loss_sum += float(loss.detach()) * len(rows)
        steps += 1
    return dict(images_visited=len(indices), unique_images_visited=len(indices),
                optimizer_steps=steps, mean_training_loss=loss_sum / len(indices),
                permutation_sha256=hashlib.sha256(indices.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--attempt-directory', required=True)
    args = parser.parse_args()
    job_path = (ROOT / args.config).resolve()
    job = read(job_path)
    verify_receipts(job['source_receipts'])
    environment = read(ROOT / job['environment_configuration'])
    if (sha(sys.executable) != environment['python_sha256']
            or sys.version != environment['python_version']
            or any(importlib.metadata.version(p) != v for p, v in environment['packages'].items())):
        raise ValueError('Frozen Python or training package environment differs')
    dataset = read(ROOT / job['dataset_configuration'])
    model_config = read(ROOT / job['model_configuration'])
    optimizer_config = read(ROOT / job['optimizer_configuration'])
    n = validate_job(job, model_config, optimizer_config, dataset)
    verify_receipts(dataset['source_receipts'])
    # Bind all eight independently audited arrays without loading test data into training.
    verify_receipts(dataset['array_receipts'])
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    from iia_benchmark.models.fm_supplemental import image_cfm_objective

    torch.set_num_threads(job['training']['cpu_threads'])
    torch.set_num_interop_threads(job['training']['cpu_threads'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Registered CUDA bfloat16 training required; CPU/FP32 fallback forbidden')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(job['seed']); np.random.seed(job['seed'])
    torch.manual_seed(job['seed']); torch.cuda.manual_seed_all(job['seed'])
    order = torch.Generator(device='cpu').manual_seed(job['seed'])
    by_field = {x['field']: x for x in dataset['array_receipts'] if x['split'] == 'train'}
    arrays = {k: np.load(by_field[k]['path'], mmap_mode='r', allow_pickle=False)
              for k in ('images', 'categories', 'labels')}
    images, categories, labels = arrays['images'], arrays['categories'], arrays['labels']
    if (images.shape != (n, 3, 128, 128) or images.dtype != np.float32
            or categories.shape != (n,) or categories.dtype != np.int64
            or labels.shape != (n,) or np.any(labels != 0)
            or np.any(categories < 0) or np.any(categories >= dataset['expected_original_category_count'])
            or len(np.unique(categories)) != dataset['expected_original_category_count']):
        raise ValueError('Full original normal training array contract differs')
    model = FlowMismatchUNet(**model_config['parameters'], asset_root=ROOT).to(device)
    optimizer = build_verified_dinomaly_stableadamw(
        model.parameters(), **optimizer_config['parameters'], asset_root=ROOT)
    output = (ROOT / job['output_directory']).resolve()
    attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output / 'attempts') or not attempt.is_dir():
        raise ValueError('Fresh registered attempt directory required')
    checkpoint_path = output / 'resume.pt'
    checkpoint_capacity = next(x['conservative_full_checkpoint_bytes']
        for x in environment['full_architectures']
        if x['dataset'] == job['dataset'] and x['variant'] == job['variant'])
    job_sha = sha(job_path)
    binding = dict(job_sha256=job_sha, dataset_configuration_sha256=sha(ROOT / job['dataset_configuration']),
                   model_configuration_sha256=sha(ROOT / job['model_configuration']),
                   optimizer_configuration_sha256=sha(ROOT / job['optimizer_configuration']))
    start, history, epoch_receipts, global_steps = 0, [], [], 0
    if checkpoint_path.exists():
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if old['binding'] != binding or old['seed'] != job['seed']:
            raise ValueError('Resume checkpoint binding differs')
        start, history = old['completed_epochs'], old['history']
        validate_history(history, start, n)
        epoch_receipts = old['epoch_receipts']; verify_receipts(epoch_receipts)
        global_steps = old['global_steps']
        if global_steps != start * ((n + 31) // 32) or not 0 <= start <= 2000:
            raise ValueError('Resume update budget differs')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        restore_rng(old['rng'], torch, np, order)
        del old
    write(attempt / 'environment_and_resume.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), binding=binding,
        torch_version=torch.__version__, numpy_version=np.__version__,
        python=sys.version, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), parameter_count=sum(p.numel() for p in model.parameters()),
        starting_epoch=start, starting_global_steps=global_steps,
        implementation='pinned related Meta U-Net and native Dinomaly StableAdamW; local single GPU',
        original_author_equivalence_certified=False, test_arrays_loaded=0,
        raw_or_derived_dataset_files_written=0))
    for epoch in range(start + 1, 2001):
        record = train_epoch(model, optimizer, images, categories,
            conditional=job['variant'] == 'conditional', batch_size=32, order=order,
            device=device, bf16=True, torch=torch, np=np, objective=image_cfm_objective)
        record['epoch'] = epoch
        history.append(record); global_steps += record['optimizer_steps']
        epoch_path = attempt / ('epoch_%04d.json' % epoch)
        write(epoch_path, record)
        epoch_receipts.append(dict(path=str(epoch_path), sha256=sha(epoch_path)))
        value = dict(binding=binding, seed=job['seed'], completed_epochs=epoch,
            global_steps=global_steps, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=capture_rng(torch, np, order), history=history, epoch_receipts=epoch_receipts)
        if shutil.disk_usage(output).free < checkpoint_capacity * 2 + 2 * 1024 ** 3:
            raise OSError('Insufficient capacity for full checkpoint; last complete epoch preserved')
        save_checkpoint(checkpoint_path, value, torch)
        if epoch % 100 == 0:
            snapshot = output / ('epoch_%04d.pt' % epoch)
            if snapshot.exists():
                if sha(snapshot) != sha(checkpoint_path):
                    raise ValueError('Immutable periodic checkpoint already differs')
            else:
                os.link(checkpoint_path, snapshot)
        write(output / 'progress.json', dict(completed_epochs=epoch, global_steps=global_steps,
            registered_epochs=2000, mean_training_loss=record['mean_training_loss'],
            attempt_directory=str(attempt), benchmark_performance=False))
        print(json.dumps(dict(global_steps=global_steps, **record)), flush=True)
    validate_history(history, 2000, n)
    snapshots = [output / ('epoch_%04d.pt' % epoch) for epoch in range(100, 2001, 100)]
    write(output / 'training_completed.json', dict(job_id=job['id'], seed=job['seed'],
        job_configuration=job_path.relative_to(ROOT).as_posix(),
        binding=binding, completed_epochs=2000, global_steps=global_steps, history=history,
        epoch_receipts=epoch_receipts,
        checkpoint_receipts=[dict(path=str(p), sha256=sha(p)) for p in snapshots],
        benchmark_performance=False, original_author_equivalence_certified=False,
        boundary='Full training completion only; original test scores, checkpoint replay and uncertainty audit still required.'))


if __name__ == '__main__':
    main()
