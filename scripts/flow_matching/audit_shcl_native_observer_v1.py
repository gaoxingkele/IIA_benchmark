"""Full64x110x25 observer plumbing; original benchmark training remains pending."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching import shcl_native_observer_v1 as observation
from scripts.flow_matching.shcl_author_audits_v1 import environment
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def worker(config,config_path):
    for pin in config['source_receipts']:
        if observation.native.sha(observation.native.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen observer fixture changed')
    if config['environment']!=environment(): raise ValueError('Exact native numerical environment required')
    import torch
    import numpy as np
    torch.set_num_threads(2)
    model_config=observation.native.read(observation.native.ROOT/config['model_config'])
    base=observation.native.ROOT/config['output_root']; base.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1); model,loss=observation.native.build_model(model_config)
    torch.manual_seed(1); baseline,_=observation.native.build_model(model_config)
    assert equal(model.state_dict(),baseline.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001); original_optimizer=torch.optim.Adam(baseline.parameters(),lr=.0001)
    x=torch.randn(64,110,25,generator=torch.Generator().manual_seed(20261012))
    labels=(torch.arange(64*110).reshape(64,110,1)%7==0).float()
    job=dict(id='shcl_native_observer_full_shape_fixture',num_epochs=1,expected_optimizer_updates=1,
        expected_validation_batches=1,expected_threshold_windows=64,expected_test_windows=64,
        source_dropped_tail_rows=0,anormly_ratio=2,synthetic_fixture=True)
    obs=observation.Observer(job,base); before=torch.get_rng_state().clone()
    baseline.train(); original_optimizer.zero_grad(); unobserved=loss(baseline(x)).mean(); unobserved.backward(); original_optimizer.step()
    baseline_after=torch.get_rng_state().clone(); torch.set_rng_state(before)
    model.train(); optimizer.zero_grad(); obs.before_forward(x,labels,'train')
    observed=loss(model(x)).mean(); observed.backward(); optimizer.step()
    solver=SimpleNamespace(model=model,optimizer=optimizer,dataset='SMAP',model_save_path=str(base/'native_checkpoint'))
    obs.after_step(solver)
    if (not torch.equal(observed,unobserved) or not equal(model.state_dict(),baseline.state_dict())
            or not equal(optimizer.state_dict(),original_optimizer.state_dict()) or not torch.equal(torch.get_rng_state(),baseline_after)):
        raise ValueError('Observation must preserve every complete native parameter/Adam state/loss/RNG after the full-batch step')
    model.eval(); obs.before_forward(x,labels,'validation')
    with torch.no_grad(): validation=loss(model(x)).mean()
    checkpoint=Path(solver.model_save_path); checkpoint.mkdir()
    torch.save(model.state_dict(),checkpoint/'SMAP_checkpoint.pth')
    obs.after_epoch(solver,0,observed.detach(),validation)
    before=obs.rng(); obs.finish_train(solver); obs.unchanged_rng(before)
    training=observation.native.read(base/'result.json')
    assert training['synthetic_operator_only_not_benchmark_performance'] and not training['original_training_complete']
    (base/'result.json').rename(base/'training_observer_fixture.json')
    for phase in ('threshold','test'):
        obs.before_forward(x,labels,phase)
        with torch.no_grad(): value=loss(model(x)); score=torch.softmax(value,dim=-1).cpu().numpy()
        obs.after_score(value,score,labels)
        if phase=='threshold': threshold=float(np.percentile(score.reshape(-1),98))
        else: scores=score.reshape(-1)
    truth=labels.numpy().reshape(-1).astype(int); prediction=(scores>threshold).astype(int)
    obs.before_adjustment(prediction,truth,threshold)
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support
    p,r,f,_=precision_recall_fscore_support(truth,prediction,average='binary')
    values=dict(accuracy=float(accuracy_score(truth,prediction)),precision=float(p),recall=float(r),f_score=float(f),threshold=threshold)
    before=obs.rng(); obs.finish_test(solver,values); obs.unchanged_rng(before)
    test=observation.native.read(base/'result.json')
    assert test['synthetic_operator_only_not_benchmark_performance'] and not test['full_native_test_complete']
    (base/'result.json').rename(base/'test_observer_fixture.json')
    exclusive(base/'full_observer_fidelity_fixture.json',dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=config_path,configuration_sha256=observation.native.sha(observation.native.ROOT/config_path),
        environment=environment(),full_input_shape=[64,110,25],all_native_parameters_and_Adam_state_bitwise_unchanged_by_observation=True,
        source_loss_and_parent_Torch_RNG_bitwise_equal=True,serialized_checkpoint_and_all_frame_observation_RNG_unchanged=True,
        original_forward_loss_optimizer_computation_unchanged=True,
        synthetic_optimizer_updates=1,original_benchmark_optimizer_updates=0,
        synthetic_operator_only_not_benchmark_performance=True,new_benchmark_results=0,
        full_original_train49_and_test364_commands_still_pending=True,all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); worker(observation.native.read(observation.native.ROOT/args.config),args.config)


if __name__=='__main__':main()
