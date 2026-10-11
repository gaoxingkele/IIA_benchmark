"""Independently verify all27 artificial cases using analytic linear flow."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from iia_benchmark.models.fm_native_guidance import load_verified_flowgrad,dflow_initial_noise_flowgrad
from iia_benchmark.models.fm_native_guidance_repaired import FlowGradPartitionRepair,load_repaired_dflow_adapter
from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import digest,json_write

ROOT=Path(__file__).resolve().parents[2]


def analytic_gradient(field,initial,target,steps):
    # Derivative is a product of scalar affine-step factors, no autograd/VJP.
    factor=1.;offset=0.
    for i in range(steps):
        rf_time=float(((torch.ones(1)*i/steps*(1-1e-3)+1e-3)*999).double()[0]/999)
        a,b=(0.,0.) if field=='zero_field' else ((.3,.2) if field=='autonomous_linear' else (2*rf_time,.2))
        step_factor=1+a/steps
        factor*=step_factor;offset=offset*step_factor+b/steps
    endpoint=initial*factor+offset
    return (2*(endpoint-target)*factor/initial.numel()).detach()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for receipt in cfg['source_receipts']:
        if digest(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen independent repair audit dependency changed')
    native_cfg=json.loads((ROOT/cfg['diagnostic_configuration']).read_text(encoding='utf-8'))
    report=json.loads((ROOT/cfg['original_diagnostic']).read_text(encoding='utf-8'))
    if not report['passed'] or report['actual_complete_cases']!=27 or report['benchmark_performance']:
        raise ValueError('Complete mathematical diagnostics required, never benchmark metrics')
    for artifact in report['paired_fixture_arrays']:
        if digest(Path(artifact['path']))!=artifact['sha256']:raise ValueError('Paired artificial image bytes changed')
    arrays=ROOT/native_cfg['artificial_array_directory']
    initial=torch.from_numpy(np.load(arrays/'initial.npy'));target=torch.from_numpy(np.load(arrays/'target.npy'))
    torch.set_num_threads(native_cfg['cpu_threads'])
    source=native_cfg['native_source'];base=native_cfg['base_adapter']
    core=load_verified_flowgrad(source['path'],source['sha256'],asset_root=ROOT)
    repaired=load_repaired_dflow_adapter(base['path'],base['sha256'],asset_root=ROOT).adapter
    fields={'zero_field':lambda x,t:torch.zeros_like(x),'autonomous_linear':lambda x,t:.3*x+.2,
            'non_autonomous_linear':lambda x,t:2*t[:,None,None,None]*x+.2}
    output=[]
    for row in report['all_cases']:
        field=row['artificial_field'];steps=row['steps'];threshold=row['straightness_threshold']
        gradient=analytic_gradient(field,initial,target,steps)
        common=dict(steps=steps,iterations=1,learning_rate=native_cfg['diagnostic_learning_rate'],straightness_threshold=threshold,optimizer='sgd_diagnostic')
        original=dflow_initial_noise_flowgrad(fields[field],target,initial,native_core=core,**common)
        fixed=repaired(fields[field],target,initial,native_core=FlowGradPartitionRepair(core),**common)
        errors={}
        for name,result in [('original',original),('repaired',fixed)]:
            observed=(initial-result['optimized_initial_noise'])/native_cfg['diagnostic_learning_rate']
            error=float((observed-gradient).norm()/gradient.norm())
            if abs(error-row[name+'_gradient_relative_error'])>cfg['reported_error_absolute_tolerance']:
                raise ValueError('All original reported gradient errors must agree with independent analytic proof')
            errors[name+'_independent_relative_error']=error
        if threshold==0 and errors['repaired_independent_relative_error']>native_cfg['full_partition_relative_error_tolerance']:
            raise ValueError('Repaired all-step gradient disagrees with closed-form solution')
        output.append(dict(artificial_field=field,steps=steps,threshold=threshold,**errors))
    result=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),verified_cases=len(output),
        independent_derivative='Closed-form product of affine Euler step derivatives; no torch.autograd or VJP used to derive expected gradients',
        full128px_paired_fixture_images_verified=True,all_original_reported_errors_reproduced=True,
        original_diagnostic_path=cfg['original_diagnostic'],original_diagnostic_sha256=digest(ROOT/cfg['original_diagnostic']),
        configuration=args.config,configuration_sha256=digest(ROOT/args.config),
        reported_error_absolute_tolerance=cfg['reported_error_absolute_tolerance'],all_cases=output,
        benchmark_performance=False,new_original_training_seed_slots=0,
        boundary='Independent mathematical audit of all complete artificial gradient cases; not trained model or original image benchmark performance.')
    out=ROOT/cfg['output']
    if out.exists():raise FileExistsError('Preserve every independent audit attempt')
    json_write(out,result);print(json.dumps(dict(passed=True,verified_cases=len(output),benchmark_performance=False)))


if __name__=='__main__':main()
