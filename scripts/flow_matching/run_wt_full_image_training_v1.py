"""Complete2000epoch image-domain WT reverse-FM; arXiv2605.23070/p22.

Full original common training budget/backbone retained. Only the objective and
declared WT per-image preprocessing differ. Shared original sources unchanged.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import json
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,validate_history,verify_receipts
from scripts.flow_matching.wt_full_image_training_contract_v1 import validate_wt_job
from scripts.flow_matching.run_flow_mismatching_full_training_v1 import write,capture_rng,restore_rng,save_checkpoint,train_epoch


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
    n = validate_wt_job(job, model_config, optimizer_config, dataset)
    verify_receipts(dataset['source_receipts'])
    # Bind all eight independently audited arrays without loading test data into training.
    verify_receipts(dataset['array_receipts'])
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    from iia_benchmark.models.fm_wt_image_training import wt_uniform_objective
    from iia_benchmark.models.fm_reflect_wt_native import load_verified_wt_flow
    from functools import partial
    native_core=load_verified_wt_flow(**model_config['native_source'],asset_root=ROOT)
    objective=partial(wt_uniform_objective,native_core=native_core,
        epsilon=model_config['method_parameters']['epsilon'])

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
        method='WT image-domain reverse FM Eqs21-23',
        native_WT_source=model_config['native_source'],
        WT_statistics=model_config['method_parameters'],
        original_author_equivalence_certified=False, test_arrays_loaded=0,
        raw_or_derived_dataset_files_written=0))
    for epoch in range(start + 1, 2001):
        record = train_epoch(model, optimizer, images, categories,
            conditional=job['variant'] == 'conditional', batch_size=32, order=order,
            device=device, bf16=True, torch=torch, np=np, objective=objective)
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
    write(output / 'training_completed.json', dict(job_id=job['id'], seed=job['seed'], method='WT_Flow_controlled_image_domain',
        native_components_configuration_sha256=sha(ROOT/job['native_components_configuration']),
        original_WT_best_feature_setting_reproduced=False,
        job_configuration=job_path.relative_to(ROOT).as_posix(),
        binding=binding, completed_epochs=2000, global_steps=global_steps, history=history,
        epoch_receipts=epoch_receipts,
        checkpoint_receipts=[dict(path=str(p), sha256=sha(p)) for p in snapshots],
        benchmark_performance=False, original_author_equivalence_certified=False,
        boundary='Full2000epoch controlled WT image-domain training only; original feature-domain best settings not reproduced; full native scoring, replay and uncertainty still required.'))


if __name__ == '__main__':
    main()
