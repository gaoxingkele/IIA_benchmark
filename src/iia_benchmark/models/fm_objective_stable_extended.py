"""Extend failed final Sinkhorn iterations without changing the FM objective.

Every successful v2 plan is returned untouched. Only the documented v2
non-convergence failure invokes the same solver with a larger final cap;
epsilon, tolerances, uniform marginals and all intermediate stages stay fixed.
"""
from dataclasses import dataclass,field
from types import FunctionType
import numpy as np
from . import fm_objective_stable as original

FAILURE_PREFIX='Epsilon-continuation Sinkhorn did not converge: marginal error='


def extended_plan(x0,x1,regularization,tolerance=1e-6,max_iterations=20000,extended_iterations=200000,audit=None):
    if extended_iterations<max_iterations:raise ValueError('Extended cap may not reduce the original final iteration budget')
    failed=None
    try:
        plan=original.stable_plan(x0,x1,regularization,tolerance,max_iterations)
        extended=False
    except RuntimeError as error:
        if not str(error).startswith(FAILURE_PREFIX):raise
        failed=str(error)
        plan=original.stable_plan(x0,x1,regularization,tolerance,extended_iterations)
        extended=True
    n=len(plan)
    marginal=max(float(np.max(np.abs(plan.sum(0)-1/n))),float(np.max(np.abs(plan.sum(1)-1/n))))
    if not np.isfinite(plan).all() or np.any(plan<0) or marginal>tolerance:raise RuntimeError('Original marginal tolerance still not met')
    if audit is not None:
        audit.append(dict(batch_size=n,regularization=float(regularization),tolerance=float(tolerance),
            original_final_cap=max_iterations,extended_final_cap=extended_iterations,used_extended_cap=extended,
            original_failure=failed,actual_absolute_marginal_error=marginal,uniform_or_exact_OT_fallback=False))
    return plan


@dataclass
class ExtendedStableCFMObjectiveDetector(original.StableCFMObjectiveDetector):
    """The exact original training code object with one private solver binding."""
    sinkhorn_max_final_iterations:int=200000
    transport_solver_audit_:list=field(default_factory=list,init=False,repr=False)

    def __post_init__(self):
        super().__post_init__()
        if type(self.sinkhorn_max_final_iterations) is not int or self.sinkhorn_max_final_iterations<20000:
            raise ValueError('Preserve at least the original20000 final Sinkhorn iterations')

    def fit(self,windows):
        self.transport_solver_audit_=[]
        def coupling(x0,x1,regularization):
            return extended_plan(x0,x1,regularization,extended_iterations=self.sinkhorn_max_final_iterations,audit=self.transport_solver_audit_)
        body=original.StableCFMObjectiveDetector.fit
        private=FunctionType(body.__code__,dict(body.__globals__,stable_plan=coupling),body.__name__,body.__defaults__,body.__closure__)
        return private(self,windows)
