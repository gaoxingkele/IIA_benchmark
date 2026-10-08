import json

import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_reflow import ReflowCFMDetector, pair_digest, transport_endpoints


def test_constant_transport_preserves_coupling_and_has_zero_curvature():
    origins = torch.tensor([[1., 2.], [-3., 4.]])
    def velocity(state):
        return torch.tensor([2., -1.]).expand(len(state), -1)
    for solver in ('euler', 'heun'):
        terminal, evidence = transport_endpoints(velocity, origins, 4, solver)
        torch.testing.assert_close(terminal, origins + torch.tensor([2., -1.]))
        assert abs(evidence['discrete_excess_action_mean']) < 1e-12


def test_heun_resolves_linear_ode_and_rejects_bad_velocity():
    x = torch.ones((2, 1))
    terminal, _ = transport_endpoints(lambda state: state[:, :-1], x, 100, 'heun')
    np.testing.assert_allclose(terminal, np.e, rtol=3e-5)
    with pytest.raises(FloatingPointError):
        transport_endpoints(lambda state: state[:, :-1] * float('nan'), x, 3, 'euler')


def test_reflow_teacher_targets_and_all_stage_evidence_are_persisted():
    values = np.random.default_rng(10).normal(size=(9, 3, 2)).astype('float32')
    model = ReflowCFMDetector(objective='rectified', window=3, hidden_size=8,
                             epochs=1, reflow_epochs=1, reflow_stages=3,
                             reflow_ode_steps=4, batch_size=4, monte_carlo_samples=1,
                             score_times=(.5,), seed=31, device='cpu').fit(values)
    evidence = model._reflow_evidence.state_dict()
    records = json.loads(bytes(evidence['metadata_json_utf8'].tolist()))
    assert len(model.training_losses_) == 3 and len(records) == 3
    for stage in (2, 3):
        z0, z1 = evidence[f'stage{stage}_z0'], evidence[f'stage{stage}_z1']
        assert records[stage - 1]['pair_sha256'] == pair_digest(z0, z1)
        teacher = torch.nn.Sequential(torch.nn.Linear(7, 8), torch.nn.SiLU(),
                                      torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 6))
        teacher.load_state_dict({r['name']: evidence[r['buffer']] for r in records[stage - 1]['teacher_parameters']})
        reconstructed = []
        for start in range(0, len(z0), 4):
            reconstructed.append(transport_endpoints(teacher, z0[start:start + 4], 4, 'heun')[0])
        torch.testing.assert_close(torch.cat(reconstructed), z1, rtol=0, atol=0)
    assert not torch.equal(evidence['stage2_z1'], evidence['stage3_z1'])
    score = model.score(values)
    assert score.shape == (9, 3) and np.isfinite(score).all()


def test_reflow_is_deterministic_and_not_independent_endpoint_training():
    values = np.random.default_rng(7).normal(size=(5, 2, 1)).astype('float32')
    params = dict(objective='rectified', window=2, hidden_size=4, epochs=1,
                  reflow_epochs=1, reflow_ode_steps=2, batch_size=3,
                  monte_carlo_samples=1, score_times=(.5,), device='cpu', seed=19)
    a, b = ReflowCFMDetector(**params).fit(values), ReflowCFMDetector(**params).fit(values)
    np.testing.assert_array_equal(a.score(values), b.score(values))
    assert a.stage_records_ == b.stage_records_
    with pytest.raises(ValueError):
        ReflowCFMDetector(objective='independent')
