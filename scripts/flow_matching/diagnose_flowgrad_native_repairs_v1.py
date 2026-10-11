"""Full128px artificial-field diagnostics; never original-data performance."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from iia_benchmark.models.fm_native_guidance import load_verified_flowgrad,unit_time_dynamic,dflow_initial_noise_flowgrad
from iia_benchmark.models.fm_native_guidance_repaired import FlowGradPartitionRepair,load_repaired_dflow_adapter
from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import digest,json_write
from scripts.flow_matching.heavy_resources import has_headroom,resource_snapshot

ROOT=Path(__file__).resolve().parents[2]


def relative_error(observed,expected):
    return float((observed-expected).norm()/expected.norm().clamp_min(1e-30))


def full_autograd(velocity,initial,target,steps):
    z=initial.detach().clone().requires_grad_(True);state=z
    dynamic=unit_time_dynamic(velocity)
    for i in range(steps):
        actual_time=(torch.ones(initial.shape[0],device=initial.device)*i/steps*(1-1e-3)+1e-3)*999
        state=state+dynamic(state,actual_time)/steps
    gradient=torch.autograd.grad((state-target).square().mean(),z)[0]
    return gradient.detach(),state.detach()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for receipt in config['source_receipts']:
        if digest(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen numerical diagnostic dependency changed')
    if not has_headroom(resource_snapshot(),config['resources']):
        raise RuntimeError('Diagnostic resource reservation not available; preserve full compute and retry explicitly')
    out=ROOT/config['output_directory'];out.mkdir(parents=True,exist_ok=False)
    arrays=ROOT/config['artificial_array_directory'];arrays.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config['cpu_threads'])
    source=config['native_source'];native=load_verified_flowgrad(source['path'],source['sha256'],asset_root=ROOT)
    base=config['base_adapter'];adapter=load_repaired_dflow_adapter(base['path'],base['sha256'],asset_root=ROOT).adapter
    generator=torch.Generator().manual_seed(config['artificial_fixture_seed'])
    shape=(config['fixture_batch'],3,128,128)
    initial=torch.randn(shape,generator=generator,dtype=torch.float64)*.1+.2
    target=torch.randn(shape,generator=generator,dtype=torch.float64)*.01+.7
    for name,value in [('initial',initial),('target',target)]:np.save(arrays/(name+'.npy'),value.numpy())
    states={'zero_field':lambda x,t:torch.zeros_like(x),
            'autonomous_linear':lambda x,t:.3*x+.2,
            'non_autonomous_linear':lambda x,t:2*t[:,None,None,None]*x+.2}
    rows=[];began=time.monotonic();source_bytes=(ROOT/base['path']).read_bytes();author_bytes=(ROOT/source['path']).read_bytes()
    for name in config['fields']:
        velocity=states[name]
        for steps in config['steps']:
            gradient,endpoint=full_autograd(velocity,initial,target,steps)
            for threshold in config['straightness_thresholds']:
                fixed=FlowGradPartitionRepair(native)
                original_trajectory,original_partition=native.generate_traj(unit_time_dynamic(velocity),initial,N=steps,straightness_threshold=threshold)
                fixed_trajectory,fixed_partition=fixed.generate_traj(unit_time_dynamic(velocity),initial,N=steps,straightness_threshold=threshold)
                if not all(torch.equal(a,b) for a,b in zip(original_trajectory,fixed_trajectory)):
                    raise ValueError('Partition repair changed original full forward state')
                common=dict(steps=steps,iterations=1,learning_rate=config['diagnostic_learning_rate'],straightness_threshold=threshold,optimizer='sgd_diagnostic')
                raw=dflow_initial_noise_flowgrad(velocity,target,initial,native_core=native,**common)
                repaired=adapter(velocity,target,initial,native_core=fixed,**common)
                g_raw=(initial-raw['optimized_initial_noise'])/config['diagnostic_learning_rate']
                g_fixed=(initial-repaired['optimized_initial_noise'])/config['diagnostic_learning_rate']
                if not torch.isfinite(g_raw).all() or not torch.isfinite(g_fixed).all():raise ValueError('Numerical gradient diagnostics must be finite')
                row=dict(artificial_field=name,steps=steps,straightness_threshold=threshold,
                    original_partition_blocks=len(original_partition['indices']),repaired_partition_blocks=len(fixed_partition['indices']),
                    complete_original_forward_states_bit_identical=True,additional_forward_calls=0,
                    original_gradient_relative_error=relative_error(g_raw,gradient),repaired_gradient_relative_error=relative_error(g_fixed,gradient),
                    repaired_zero_energy_finite_ratios=fixed.last_repair_audit['zero_same_velocity_ratios_defined_as_zero'],
                    original_anomaly_dataset_performance=False)
                if threshold==0 and row['repaired_gradient_relative_error']>config['full_partition_relative_error_tolerance']:
                    raise ValueError('Full non-approximated repaired VJP fails independent all-step gradient tolerance')
                rows.append(row);json_write(out/'progress.json',dict(completed_cases=len(rows),total_cases=27,latest=row,benchmark_performance=False))
                print(json.dumps(dict(completed_cases=len(rows),field=name,steps=steps,threshold=threshold,original_relative_error=row['original_gradient_relative_error'],repaired_relative_error=row['repaired_gradient_relative_error'])),flush=True)
    if source_bytes!=(ROOT/base['path']).read_bytes() or author_bytes!=(ROOT/source['path']).read_bytes():
        raise ValueError('Original source or frozen local v1 was changed')
    with (out/'all_artificial_gradient_cases.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
        configuration_sha256=digest(ROOT/args.config),actual_complete_cases=len(rows),complete_image_fixture_shape=list(shape),
        artificial_fixture_seed=config['artificial_fixture_seed'],actual_elapsed_seconds=time.monotonic()-began,
        all_original_forward_states_bit_identical=True,original_author_and_local_v1_sources_unchanged=True,
        full_partition_relative_error_tolerance=config['full_partition_relative_error_tolerance'],all_cases=rows,
        original_native_source=source,original_local_adapter=base,
        paired_fixture_arrays=[dict(path=str(p.resolve()),sha256=digest(p)) for p in arrays.glob('*.npy')],
        existing_original_native_model_training_seed_slots_added=0,benchmark_performance=False,
        strict_TAB_equivalent=False,original_anomaly_author_protocol_reproduced=False,
        boundary='Artificial full-resolution mathematical diagnostic across complete forward trajectories. Relative gradient errors and partitions are numerical evidence only. Approximate nonuniform gradients may improve or worsen; no paper anomaly metrics, trained full model, original data benchmark or new training seed claimed.')
    json_write(out/'full_artificial_gradient_repair_diagnostic.json',report)
    print(json.dumps(dict(passed=True,actual_cases=len(rows),full_original_anomaly_experiment_complete=False)))


if __name__=='__main__':main()
