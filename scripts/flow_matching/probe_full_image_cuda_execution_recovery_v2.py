"""Actual full32x3x128x128 CUDA update and long-path IO, capacity only, never performance."""
import argparse
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
from pathlib import Path
import random
import sys
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts, validate_job
from scripts.flow_matching.wt_full_image_training_contract_v1 import validate_wt_job
from scripts.flow_matching.full_image_training_execution_recovery_v2 import (
    enable_native_activation_recomputation, write_json, save_checkpoint, native_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--case-id',required=True)
    parser.add_argument('--attempt-directory',required=True);args=parser.parse_args()
    config=read(ROOT/args.config);verify_receipts(config['source_receipts'])
    case=next(c for c in config['capacity_cases'] if c['id']==args.case_id)
    job=read(ROOT/case['original_training_configuration']);verify_receipts(job['source_receipts'])
    dataset=read(ROOT/job['dataset_configuration']);model_config=read(ROOT/job['model_configuration'])
    optimizer_config=read(ROOT/job['optimizer_configuration'])
    validator=validate_wt_job if case['objective']=='WT' else validate_job
    validator(job,model_config,optimizer_config,dataset)
    environment=read(ROOT/job['environment_configuration'])
    if (sha(sys.executable)!=environment['python_sha256'] or sys.version!=environment['python_version']
            or any(importlib.metadata.version(p)!=v for p,v in environment['packages'].items())):
        raise ValueError('Original frozen full CUDA training environment required')
    verify_receipts(dataset['source_receipts']);verify_receipts(dataset['array_receipts'])
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import torch
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    from iia_benchmark.models.fm_supplemental import image_cfm_objective
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Original CUDA BF16 required')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    model=FlowMismatchUNet(**model_config['parameters'],asset_root=ROOT)
    counts=enable_native_activation_recomputation(model)
    if counts!=dict(ResBlock=54,AttentionBlock=29) or sum(p.numel() for p in model.parameters())!=45289539:
        raise ValueError('Complete original64channel/3block/15class backbone required')
    model.to('cuda:0').train()
    optimizer=build_verified_dinomaly_stableadamw(model.parameters(),**optimizer_config['parameters'],asset_root=ROOT)
    by_field={x['field']:x for x in dataset['array_receipts'] if x['split']=='train'}
    images=np.load(by_field['images']['path'],mmap_mode='r',allow_pickle=False)
    categories=np.load(by_field['categories']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.load(by_field['labels']['path'],mmap_mode='r',allow_pickle=False)
    if images.shape!=(3629,3,128,128) or np.any(labels!=0):raise ValueError('Full original normal training source required')
    order=torch.Generator(device='cpu').manual_seed(job['seed'])
    indices=torch.randperm(len(images),generator=order).numpy()[:32]
    batch=torch.from_numpy(np.array(images[indices],copy=True)).to('cuda:0')
    classes=torch.from_numpy(np.array(categories[indices],copy=True)).to('cuda:0')
    objective=image_cfm_objective
    if case['objective']=='WT':
        from iia_benchmark.models.fm_wt_image_training import wt_uniform_objective
        from iia_benchmark.models.fm_reflect_wt_native import load_verified_wt_flow
        native_core=load_verified_wt_flow(**model_config['native_source'],asset_root=ROOT)
        objective=partial(wt_uniform_objective,native_core=native_core,epsilon=model_config['method_parameters']['epsilon'])
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,batch,labels=classes)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite complete native batch32 loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite complete native batch32 gradient')
    optimizer.step();torch.cuda.synchronize()
    output=(ROOT/case['output_directory']).resolve();attempt=Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output/'attempts') or not attempt.is_dir():raise ValueError('Registered preserved capacity attempt required')
    checkpoint=attempt/'full_batch32_capacity_checkpoint.pt'
    save_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),original_training_configuration=case['original_training_configuration'],
        capacity_probe_only=True,model_performance_measured=False),torch)
    restored=torch.load(native_path(checkpoint),map_location='cpu',weights_only=False)
    if set(restored['model'])!=set(model.state_dict()) or not restored['capacity_probe_only']:
        raise ValueError('Actual full checkpoint IO/reload must agree')
    for key,value in model.state_dict().items():
        if not torch.equal(value.cpu(),restored['model'][key]):raise ValueError('Full actual capacity checkpoint weight changed')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),case=case,configuration=args.config,
        configuration_sha256=sha(ROOT/args.config),original_job_sha256=sha(ROOT/case['original_training_configuration']),
        original_source_array_receipts=list(by_field.values()),actual_batch_shape=list(batch.shape),
        original_full_normal_training_source_images=len(images),normal_training_indices=indices.tolist(),
        full_model_parameters=45289539,native_recomputed_blocks=counts,bfloat16=True,TF32=True,
        actual_optimizer_updates=1,full_batch_forward_backward_optimizer_and_checkpoint_IO_passed=True,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(),checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),
        loss=float(loss.detach()),capacity_probe_only=True,full_original_training_executed=False,
        additional_canonical_training_seed_slots=0,benchmark_performance=False,author_equivalence_certified=False,
        all_original_experiments_complete=False)
    path=attempt/'full_batch32_cuda_capacity_verified.json';write_json(path,report)
    if (output/'capacity_probe_result.json').exists():raise FileExistsError('Preserve original capacity proof')
    write_json(output/'capacity_probe_result.json',dict(path=str(path),sha256=sha(path),passed=True))
    print(dict(case_id=case['id'],batch_shape=list(batch.shape),native_blocks=counts,peak_allocated=report['cuda_peak_allocated_bytes'],
        full_original_training_executed=False,benchmark_performance=False))


if __name__=='__main__':main()
