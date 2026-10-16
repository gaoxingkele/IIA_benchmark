import numpy as np
import pytest
import torch
from iia_benchmark.models import fm_objective_stable as original
from iia_benchmark.models.fm_objective_stable_extended import extended_plan,ExtendedStableCFMObjectiveDetector,FAILURE_PREFIX


def test_every_original_success_and_rng_state_are_returned_bitwise_unchanged():
    g=torch.Generator().manual_seed(14);a=torch.randn((4,5),generator=g);b=torch.randn((4,5),generator=g)
    state=torch.random.get_rng_state().clone();expected=original.stable_plan(a,b,2.);audit=[]
    actual=extended_plan(a,b,2.,audit=audit)
    assert np.array_equal(actual,expected) and torch.equal(torch.random.get_rng_state(),state)
    assert not audit[0]['used_extended_cap'] and audit[0]['tolerance']==1e-6 and audit[0]['regularization']==2.


def test_failed_solver_only_extends_its_final_cap_and_keeps_problem(monkeypatch):
    calls=[];a=torch.zeros((2,3));b=torch.ones((2,3));expected=np.full((2,2),.25)
    def solver(x,y,reg,tol,cap):
        assert x is a and y is b;calls.append((reg,tol,cap))
        if cap==20000:raise RuntimeError(FAILURE_PREFIX+'1.03e-06')
        return expected
    monkeypatch.setattr(original,'stable_plan',solver);audit=[]
    assert extended_plan(a,b,2.,audit=audit) is expected
    assert calls==[(2.,1e-6,20000),(2.,1e-6,200000)]
    assert audit[0]['used_extended_cap'] and not audit[0]['uniform_or_exact_OT_fallback']


def test_unrelated_errors_and_persistent_nonconvergence_are_not_hidden(monkeypatch):
    def unrelated(*args):raise RuntimeError('Different tensor error')
    monkeypatch.setattr(original,'stable_plan',unrelated)
    with pytest.raises(RuntimeError,match='Different tensor error'):extended_plan(torch.zeros((2,1)),torch.zeros((2,1)),2.)
    def failure(*args):raise RuntimeError(FAILURE_PREFIX+'1e-5')
    monkeypatch.setattr(original,'stable_plan',failure)
    with pytest.raises(RuntimeError,match='did not converge'):extended_plan(torch.zeros((2,1)),torch.zeros((2,1)),2.)


@pytest.mark.parametrize('bad',[np.array([[.2,.2],[.2,.2]]),np.array([[np.nan,.25],[.25,.25]]),np.array([[-.25,.75],[.75,-.25]])])
def test_bad_marginals_nonfinite_or_negative_mass_never_become_accepted(monkeypatch,bad):
    monkeypatch.setattr(original,'stable_plan',lambda *args:bad)
    with pytest.raises(RuntimeError):extended_plan(torch.zeros((2,1)),torch.zeros((2,1)),2.)


def test_preserves_full_training_losses_weights_and_scores_on_original_successes():
    values=np.random.default_rng(12).normal(size=(4,3,2)).astype(np.float32)
    p=dict(objective='sf2m',sigma=1.,window=3,hidden_size=8,epochs=2,batch_size=4,monte_carlo_samples=1,device='cpu')
    torch.set_num_threads(2)
    before=original.StableCFMObjectiveDetector(**p).fit(values)
    after=ExtendedStableCFMObjectiveDetector(**p).fit(values)
    assert before.training_losses_==after.training_losses_
    assert all(torch.equal(v,after._model.state_dict()[k]) for k,v in before._model.state_dict().items())
    assert np.array_equal(before.score(values),after.score(values))
    assert not any(x['used_extended_cap'] for x in after.transport_solver_audit_)
    assert after.score.__func__ is before.score.__func__


def test_original_training_module_is_not_globally_patched_and_budget_cannot_shrink():
    func=original.StableCFMObjectiveDetector.fit;solver=original.stable_plan
    ExtendedStableCFMObjectiveDetector(objective='sb',sigma=1.)
    assert original.StableCFMObjectiveDetector.fit is func and original.stable_plan is solver
    with pytest.raises(ValueError):ExtendedStableCFMObjectiveDetector(objective='sb',sigma=1.,sinkhorn_max_final_iterations=19999)
    with pytest.raises(ValueError):extended_plan(torch.zeros((2,1)),torch.zeros((2,1)),2.,extended_iterations=19999)
