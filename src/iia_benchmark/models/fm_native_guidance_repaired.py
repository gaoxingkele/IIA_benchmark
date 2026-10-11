"""Explicit local repair branch; original source and v1 bindings stay intact.

Correct the first-boundary wrap and undefined zero-energy straightness ratios.
Keep the author's complete forward trajectory, controls, RF time and VJP rule.
Repaired source-space VJPs use the actual recorded forward time arguments.
No claim of equality with the unavailable anomaly author's adaptation.
"""
import ast
import hashlib
from pathlib import Path
import types
from collections.abc import Callable

import torch

from iia_benchmark.models.fm_native_guidance import unit_time_dynamic
from iia_benchmark.models.fm_supplemental_author import _resolve_asset


class FlowGradPartitionRepair:
    """Wrap the original forward function; no model calls are added.

    The first velocity has no preceding neighbor. At a zero denominator,
    identical velocities have distance0; a nonzero difference has infinite
    relative distance and forces a segment boundary. No arbitrary epsilon
    or user-visible training/compute budget is introduced.
    """
    def __init__(self, native_core):
        self.original=native_core
        self.source_sha256=native_core.source_sha256
        self.last_scaled_times=[]
        self.last_repair_audit={}

    @torch.no_grad()
    def generate_traj(self,dynamic,z0,u=None,N=100,straightness_threshold=None):
        if N<1:raise ValueError('Full original positive step budget required')
        if straightness_threshold is not None and (not torch.isfinite(torch.tensor(straightness_threshold)) or straightness_threshold<0):
            raise ValueError('Finite nonnegative straightness threshold required')
        predictions=[];times=[]
        def capture(state,scaled_time):
            value=dynamic(state,scaled_time)
            if value.shape!=state.shape or not torch.isfinite(value).all():
                raise ValueError('Complete original velocity must have matching shape and finite values')
            predictions.append(value.detach().clone().cpu())
            times.append(scaled_time.detach().clone())
            return value
        trajectory=self.original.generate_traj(capture,z0,u=u,N=N,straightness_threshold=None)
        if len(predictions)!=N or len(trajectory)!=N+1:
            raise ValueError('Complete original forward trajectory must be retained')
        self.last_scaled_times=times
        if straightness_threshold is None:return trajectory
        partition={'indices':[],'length':{}}
        start=0;length=0;accumulated=0.;zero_same=0;zero_changed=0;ratios=[]
        for i,prediction in enumerate(predictions):
            denominator=float(prediction.square().sum())
            neighbor_ratios=[]
            for neighbor in (i-1,i+1):
                if not 0<=neighbor<N:
                    neighbor_ratios.append(0.);continue
                numerator=float((predictions[neighbor]-prediction).square().sum())
                if denominator==0:
                    if numerator==0:ratio=0.;zero_same+=1
                    else:ratio=float('inf');zero_changed+=1
                else:ratio=numerator/denominator
                neighbor_ratios.append(ratio)
            distance=max(neighbor_ratios)
            ratios.append('infinity_at_zero_velocity_with_nonzero_neighbor' if distance==float('inf') else distance)
            accumulated+=distance;length+=1
            if accumulated>straightness_threshold or i==N-1:
                partition['indices'].append(start);partition['length'][start]=length
                start=i+1;length=0;accumulated=0.
        if sum(partition['length'].values())!=N or partition['indices'][0]!=0:
            raise ValueError('Repaired partition must cover all original steps')
        self.last_repair_audit=dict(first_boundary_wrap_removed=True,
            zero_same_velocity_ratios_defined_as_zero=zero_same,
            zero_velocity_nonzero_neighbor_boundaries=zero_changed,
            per_step_straightness=ratios,forward_calls=N,extra_forward_model_calls=0,
            original_RF_time_and_epsilon_preserved=True,original_source_bytes_modified=False)
        return trajectory,partition


def load_repaired_dflow_adapter(base_adapter_source_path:str|Path,expected_base_sha256:str,
                               *,asset_root:str|Path|None=None):
    """Apply precisely two AST repairs to a frozen original local v1 adapter.

    Reuse every optimizer/loss/compute check unchanged. Replace recomputed
    VJP time with the actual original forward call's argument and accurately
    mark the repair branch. All patches are counted; unexpected source fails.
    """
    path=_resolve_asset(base_adapter_source_path,asset_root)
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected_base_sha256:
        raise ValueError('Frozen v1 source-space adapter SHA256 differs')
    tree=ast.parse(source.decode('utf-8'),filename=str(path))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dflow_initial_noise_flowgrad']
    if len(selected)!=1:raise ValueError('Exactly one original source-space adapter required')
    function=selected[0];time_patches=0;flag_patches=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='native_time':
            node.value=ast.parse('native_core.last_scaled_times[j].detach().clone().to(z.device)',mode='eval').body
            time_patches+=1
        if isinstance(node,ast.keyword) and node.arg=='native_first_step_wrap_preserved':
            if not isinstance(node.value,ast.Constant) or node.value.value is not True:
                raise ValueError('Original repair provenance flag contract changed')
            node.value=ast.Constant(False);flag_patches+=1
    if (time_patches,flag_patches)!=(1,1):
        raise ValueError('Exactly two mechanical source-space repairs required')
    namespace={'torch':torch,'Tensor':torch.Tensor,'Callable':Callable,'unit_time_dynamic':unit_time_dynamic}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    derived=namespace['dflow_initial_noise_flowgrad']
    def adapter(*args,**kwargs):
        core=kwargs.get('native_core')
        if not isinstance(core,FlowGradPartitionRepair):
            raise ValueError('Explicit repaired native core required; never mislabel the original branch')
        result=derived(*args,**kwargs)
        result.update(reproduction_status='explicit_local_boundary_zero_energy_and_forward_time_repairs_original_anomaly_protocol_unverified',
            repair_audit=dict(core.last_repair_audit),base_adapter_sha256=expected_base_sha256,
            original_source_modified=False,author_anomaly_implementation_verified=False)
        return result
    return types.SimpleNamespace(adapter=adapter,base_adapter_sha256=expected_base_sha256,
        controlled_AST_patches=2,original_source_file_modified=False)
