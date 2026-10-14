"""Full architecture restore fixtures validate recovery, never benchmark quality."""
import ast
from copy import deepcopy
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_cfm_ts_original import CFMTSPaperTrajectory, PaperTrajectoryField
from scripts.flow_matching import run_cfm_saved_full_score_recovery_v1 as recovery
from scripts.flow_matching import run_cfm_ts_original_job as original


def fixture(tmp_path):
    model = CFMTSPaperTrajectory(method='gp', target_mode='gaussian_conditional', epochs=400, batch_size=258)
    field = PaperTrajectoryField(1, **model.network_parameters)
    job = dict(id='diagnostic_restore_fixture_only', epochs=400, expected_optimizer_updates=62400,
        training_items=40000, batch_size=258, observed_dimensions=1)
    checkpoint = dict(job=job, job_sha256=original.fingerprint(job), state_dict=field.state_dict(),
        network_parameters=model.network_parameters, optimizer_updates=62400,
        training_losses=[0.5]*400, preprocessing=[{'fixture': True}])
    path = tmp_path/'model.pt'
    torch.save(checkpoint, path)
    return model, checkpoint, dict(original_job=job, checkpoint_path=str(path))


def test_restore_actual_full_architecture_without_fit_or_optimizer(tmp_path, monkeypatch):
    model, checkpoint, config = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery must never train or preprocess targets')
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    monkeypatch.setattr(model, 'fit', forbidden)
    monkeypatch.setattr(model, 'prepare_targets', forbidden)
    recovery.restore(model, config)
    assert not model.model.training
    assert sum(p.numel() for p in model.model.parameters()) > 150000
    assert model.optimizer_updates_ == 62400 and model.training_items_ == 40000
    assert model.loss_history_ == checkpoint['training_losses']
    assert all(torch.equal(v, model.model.state_dict()[k]) for k, v in checkpoint['state_dict'].items())
    assert np.isfinite(model.simulate([0.], np.linspace(0, .01, 5))).all()


@pytest.mark.parametrize('corruption', ['budget', 'history', 'job', 'nonfinite'])
def test_reject_partial_or_invalid_saved_checkpoint(tmp_path, corruption):
    model, checkpoint, config = fixture(tmp_path)
    checkpoint = deepcopy(checkpoint)
    if corruption == 'budget': checkpoint['optimizer_updates'] -= 1
    if corruption == 'history': checkpoint['training_losses'].pop()
    if corruption == 'job': checkpoint['job_sha256'] = '0'*64
    if corruption == 'nonfinite': next(iter(checkpoint['state_dict'].values())).fill_(float('inf'))
    torch.save(checkpoint, config['checkpoint_path'])
    with pytest.raises(ValueError): recovery.restore(model, config)


def test_replace_only_fit_keep_every_original_scoring_statement():
    import inspect
    source = ast.parse(inspect.getsource(original.run))
    changed = recovery.scoring_ast(original)
    differences = [(a, b) for a, b in zip(source.body[0].body, changed.body[0].body) if ast.dump(a) != ast.dump(b)]
    assert len(differences) == 1
    assert ast.unparse(differences[0][0]).startswith('model.fit(')
    assert ast.unparse(differences[0][1]) == '_restore(model, _recovery_config)'
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit'
        for n in ast.walk(changed))


def test_registered_full285_route_and_original_guards():
    config = recovery.cold.original.read(recovery.ROOT/'configs/runtime/fm_cfm_saved_score_recovery.v1.json')
    primary = recovery.verify(config)
    from scripts.flow_matching.run_cfm_saved_score_full_replay_v1 import namespace
    runtime = recovery.cold.original.read(recovery.ROOT/'configs/runtime/fm_cfm_saved_score_full_replay.v1.json')
    routes, _, records = namespace()['verify'](runtime)
    assert len(routes) == 285 and recovery.IDENTITY not in records
    assert len([r for r, old in zip(routes, primary['routes']) if r != old]) == 1
    assert config['settings'] == primary['settings']
    assert config['settings']['minimum_commit_headroom_bytes'] == 30*1024**3
