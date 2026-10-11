"""Immutable original FlowGrad core and explicit D-Flow source-space adapter.

Original FlowGrad optimizes controls u at multiple times. D-Flow optimizes the
initial noise only. Those are distinct APIs here; the source-space adapter is
paper-derived, not the unavailable Flow Mismatching author's implementation.
"""
import ast
import hashlib
from pathlib import Path
import time
import types
from collections.abc import Callable

import torch
from torch import Tensor

from iia_benchmark.models.fm_supplemental_author import _resolve_asset


def load_verified_flowgrad(source_path: str | Path, expected_sha256: str,
                          *, asset_root: str | Path | None = None):
    """Load precisely the author's trajectory/optimization defs, unchanged.

    AST selection avoids importing CLIP/LPIPS/image editing dependencies for
    the gradient core. Neither original function body nor source bytes change.
    """
    path=_resolve_asset(source_path,asset_root)
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected_sha256:
        raise ValueError('Immutable original FlowGrad function source SHA256 differs')
    tree=ast.parse(source.decode('utf-8'),filename=str(path))
    names={'generate_traj','flowgrad_optimization'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes}!=names:
        raise ValueError('Both original complete FlowGrad function definitions required')
    selected=ast.Module(body=nodes,type_ignores=[])
    namespace={'torch':torch,'time':time,'__name__':'_iia_original_flowgrad'}
    exec(compile(ast.fix_missing_locations(selected),str(path),'exec'),namespace)
    return types.SimpleNamespace(generate_traj=namespace['generate_traj'],
        flowgrad_optimization=namespace['flowgrad_optimization'],source_path=str(path),source_sha256=expected_sha256)


def unit_time_dynamic(velocity: Callable, labels: Tensor | None = None):
    """Adapt only original RF t*999 argument to the unit-time image U-Net."""
    def dynamic(x,original_time):
        t=original_time.to(device=x.device,dtype=x.dtype)/999
        if t.numel()==1:t=t.reshape(1).expand(x.shape[0])
        if labels is not None and labels.shape!=(x.shape[0],):
            raise ValueError('One fixed category label per complete image required')
        result=velocity(x,t,labels) if labels is not None else velocity(x,t)
        if result.shape!=x.shape:raise ValueError('Native velocity output shape differs')
        return result
    return dynamic


def dflow_initial_noise_flowgrad(velocity: Callable, images: Tensor, initial_noise: Tensor,
        *, native_core, steps: int, iterations: int, learning_rate: float,
        straightness_threshold: float, optimizer: str,
        lbfgs_inner_iterations: int | None = None, labels: Tensor | None = None):
    """D-Flow Eq.26 with MSE, native FlowGrad forward and VJP partition.

    Only initial_noise is optimized; model parameters never accumulate .grad.
    D-Flow p4 uses L-BFGS with line search; the separately named SGD option is
    a LOCAL diagnostic. Mandatory inner iterations cannot be copied from the
    original unrelated inverse tasks into the anomaly comparison unnoticed.
    Native RF epsilon/partition (including first-step wrap) are preserved.
    Strong-Wolfe is the explicit local PyTorch line-search choice, not a verified
    author implementation. No certified anomaly performance/equivalence claim.
    """
    if images.ndim!=4 or initial_noise.shape!=images.shape:
        raise ValueError('Complete paired [batch,channel,height,width] tensors required')
    if steps<1 or iterations<1 or learning_rate<=0 or straightness_threshold<0:
        raise ValueError('Positive explicit compute budgets and learning rate required')
    if optimizer not in ('lbfgs','sgd_diagnostic'):
        raise ValueError('Explicit lbfgs or separately named local sgd_diagnostic required')
    if optimizer=='lbfgs' and (lbfgs_inner_iterations is None or lbfgs_inner_iterations<1):
        raise ValueError('Original anomaly LBFGS inner budget is unreported; freeze it explicitly')
    dynamic=unit_time_dynamic(velocity,labels)
    z=initial_noise.detach().clone().requires_grad_(True);target=images.detach();trace=[]
    solver=(torch.optim.LBFGS([z],lr=learning_rate,max_iter=lbfgs_inner_iterations,
        line_search_fn='strong_wolfe') if optimizer=='lbfgs' else None)
    # torch.enable_grad ensures correctness if a caller used inference no_grad.
    def evaluate_source_gradient(iteration):
        trajectory,partition=native_core.generate_traj(dynamic,z,N=steps,straightness_threshold=straightness_threshold)
        with torch.enable_grad():
            endpoint=trajectory[-1].detach().to(z.device).requires_grad_(True)
            loss=(endpoint-target).square().mean()
            lam=torch.autograd.grad(loss,endpoint)[0].detach()
            for j in reversed(partition['indices']):
                state=trajectory[j].detach().to(z.device).requires_grad_(True)
                native_time=z.new_tensor([j/steps*(1-1e-3)+1e-3])*999
                interval=partition['length'][j]/steps
                def mapping(value):
                    shaped=value.reshape_as(z)
                    return (shaped+dynamic(shaped,native_time)*interval).reshape(-1)
                _,vjp=torch.autograd.functional.vjp(mapping,state.reshape(-1),v=lam.reshape(-1))
                lam=vjp.detach().reshape_as(z)
        if not torch.isfinite(loss) or not torch.isfinite(lam).all():
            raise ValueError('Native original FlowGrad trajectory/source gradient is nonfinite')
        trace.append(dict(iteration=iteration,closure_evaluation=len(trace)+1,pre_update_MSE=float(loss.detach()),
                          selected_indices=list(partition['indices']),
                          intervals=dict(partition['length']),gradient_L2=float(lam.norm()),
                          gradient_target='initial_noise_only',native_first_step_wrap_preserved=True,
                          optimizer=optimizer))
        return loss.detach(),lam
    for iteration in range(iterations):
        if solver is not None:
            def closure():
                solver.zero_grad(set_to_none=True)
                loss,gradient=evaluate_source_gradient(iteration)
                z.grad=gradient
                return loss
            with torch.enable_grad():solver.step(closure)
        else:
            _,gradient=evaluate_source_gradient(iteration)
            with torch.no_grad():z.add_(gradient,alpha=-learning_rate)
    final=native_core.generate_traj(dynamic,z,N=steps)[-1].detach().to(z.device)
    heatmap=torch.linalg.vector_norm(target-final,dim=1)
    return dict(optimized_initial_noise=z.detach(),reconstruction=final,heatmap=heatmap,
        optimization_trace=trace,closure_evaluations=len(trace),optimizer=optimizer,
        line_search='strong_wolfe_local_choice' if solver is not None else None,
        final_MSE=float((final-target).square().mean()),
        original_function_source_sha256=native_core.source_sha256,
        reproduction_status='paper_derived_source_space_adapter_native_FlowGrad_core_original_anomaly_settings_unreported',
        author_anomaly_implementation_verified=False,strict_TAB_equivalent=False)
