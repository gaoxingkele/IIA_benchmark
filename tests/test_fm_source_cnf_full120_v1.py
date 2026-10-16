import ast
import copy
import inspect
from types import SimpleNamespace
import pytest
import torch

from iia_benchmark.models import otcfm_cnf_manual_training_v1 as training
from scripts.flow_matching import run_source_cnf_full120_v1 as s
from scripts.flow_matching import audit_source_cnf_full120_v1 as audit


def test_all_five_full_data_tracks_all_leaveouts_both_parameter_tracks_and_five_seeds():
    jobs=s.scientific_cases()
    assert len(jobs)==len({j['id'] for j in jobs})==120
    assert len({j['dataset'] for j in jobs})==5
    from collections import defaultdict
    groups=defaultdict(list)
    for j in jobs:
        groups[(j['dataset'],j['protocol'],j['leaveout_timepoint'])].append(j)
        r=j['scientific_protocol']
        assert r['batch_size']==1024 and r['max_epochs']==1000 and r['validation_every_n_epochs']==10
        assert r['early_stopping_patience_validation_checks']==100 and r['maximum_training_wall_seconds'] is None
        assert r['ode_solver']=='dopri5' and r['ode_atol']==r['ode_rtol']==1e-5
        assert r['weight_decay']==(.01 if j['protocol']==s.PROTOCOLS[0] else 1e-5)
        assert not j['validation_compatibility']['training_and_test_loaders_changed']
    assert len(groups)==24 and all({j['seed'] for j in rows}=={42,43,44,45,46} for rows in groups.values())


@pytest.mark.parametrize('points,left',[(4,1),(4,2),(5,1),(5,2),(5,3)])
@pytest.mark.parametrize('decay',[.01,1e-5])
def test_one_AdamW_update_matches_full_joint_objective_and_excludes_only_heldout(points,left,decay):
    torch.manual_seed(42)
    batch=[torch.randn(1024,5,dtype=torch.float64) for _ in range(points)]
    def run(streamed):
        parameter=torch.nn.Parameter(torch.linspace(.1,.5,5,dtype=torch.float64))
        optimizer=torch.optim.AdamW([parameter],lr=.001,weight_decay=decay)
        visited=[]
        def term(index):
            return ((batch[index]*parameter).sin().square().mean()+parameter.square().mean())
        def step(b,training,endpoint_index):
            visited.append(endpoint_index);return parameter.sum()*0,term(endpoint_index)
        logs={}
        model=SimpleNamespace(is_trajectory=True,hparams=SimpleNamespace(leaveout_timepoint=left),
            optimizers=lambda:optimizer,manual_backward=lambda loss:loss.backward(),
            parameters=lambda:iter([parameter]),step=step,log_dict=lambda values,**kwargs:logs.update(values))
        if streamed:value=training.training_step(model,batch,0)
        else:
            optimizer.zero_grad();value=torch.stack([term(i) for i in reversed(range(points)) if i!=left]).mean()
            value.backward();optimizer.step()
        return parameter.detach(),parameter.grad,optimizer.state_dict(),visited,value
    joint,streamed=run(False),run(True)
    torch.testing.assert_close(joint[0],streamed[0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(joint[1],streamed[1],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(joint[4],streamed[4],rtol=1e-12,atol=1e-12)
    assert streamed[3]==[i for i in reversed(range(points)) if i!=left]
    for k in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(joint[2]['state'][0][k],streamed[2]['state'][0][k],rtol=1e-12,atol=1e-12)
    assert int(streamed[2]['state'][0]['step'])==1


def test_native_replay_ast_changes_only_registered_optimizer_decay_assertion():
    original=ast.parse(inspect.getsource(audit.parent.worker));changed=audit.replay_tree()
    class Restore(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.unparse(node)=="group['weight_decay'] != job['scientific_protocol']['weight_decay']":
                return ast.parse("group['weight_decay'] != 1e-5",mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(original)==ast.dump(Restore().visit(changed))


def test_never_install_twice_or_train_reduced_batch():
    model=SimpleNamespace(_iia_endpoint_backward_installed=True,automatic_optimization=True)
    training.install(model);assert not model.automatic_optimization
    with pytest.raises(ValueError):training.install(model)
    invalid=SimpleNamespace(is_trajectory=True,hparams=SimpleNamespace(leaveout_timepoint=1))
    with pytest.raises(ValueError):training.training_step(invalid,[torch.zeros(128,5)]*4,0)


def test_exact_likelihood_evaluation_uses_no_grad_and_cannot_silently_enable_inference_mode():
    received={}
    native=SimpleNamespace(Trainer=lambda **kwargs:received.update(kwargs))
    s.trainer_facade(native).Trainer(precision=32)
    assert received==dict(inference_mode=False,precision=32)
    with pytest.raises(ValueError):s.trainer_facade(native).Trainer(inference_mode=True)


def test_all12_actual_full_input_preflight_completed_with_real_gradients():
    c=s.read(s.ROOT/s.preflight.CONFIG)
    report=s.read(__import__('pathlib').Path(c['artifact_root'])/'full12_native_likelihood_preflight.json')
    guard=s.read(s.ROOT/c['state_root']/'guard_receipt.json')
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
    assert guard['report_sha256']==s.sha(__import__('pathlib').Path(c['artifact_root'])/'full12_native_likelihood_preflight.json')
    assert len(report['records'])==12
    for row in report['records']:
        assert row['training_batch']==1024 and row['optimizer_updates']==0
        assert row['requested_leftout_gradient_zero'] and row['all_retained_timepoint_gradients_nonzero_finite']
