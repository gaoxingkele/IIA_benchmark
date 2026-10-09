import copy
import math
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.flow_matching.prepare_cfm_ts_original_data import field, generate_case
from iia_benchmark.models.fm_cfm_ts_original import CFMTSPaperTrajectory, PaperTrajectoryField


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    config = json.loads((ROOT / 'configs/reproducibility/cfm_ts_original_protocol.v1.json').read_text())
    config['evaluation_points'] = 15
    track = copy.deepcopy(config['tracks'][0])
    track['trajectory_counts'] = {'sine': 5, 'lotka_volterra': 5, 'pendulum': 1}
    return config, track


def test_sine_solution_and_complete_grouped_split():
    config, track = protocol()
    dataset = copy.deepcopy(config['datasets'][1]); dataset['observation_points'] = 5
    data = generate_case(config, track, dataset, 1103)
    initial = data['initial_states'][:, 0]
    expected = 2 * np.arctan(np.tan(initial[:, None]/2) * np.exp(data['test_times']))
    np.testing.assert_allclose(data['test_states'][:, :, 0], expected, rtol=1e-8, atol=1e-9)
    assert len(data['train_ids']) == 4 and len(data['test_ids']) == 1
    other = generate_case(config, track, dataset, 1103)
    assert all(np.array_equal(data[k], other[k]) for k in data)


def test_pendulum_position_only_new_times_and_physical_energy():
    config, track = protocol(); dataset = config['datasets'][2]
    data = generate_case(config, track, dataset, 1103)
    assert data['observed_states'].shape == (1, 160, 1)
    assert data['initial_states'].shape == (1, 1)
    assert not np.isin(data['test_times'], data['observed_times']).any()
    theta, omega = data['physical_ground_truth'][0].T
    energy = .5*omega**2+9.81*(1-np.cos(theta))
    np.testing.assert_allclose(energy, 9.81, rtol=1e-8)


def test_lv_equations_and_initial_distribution():
    assert field('lotka_volterra', 0, [2, 3]) == [-2, -6]
    config, track = protocol(); dataset = copy.deepcopy(config['datasets'][0]); dataset['observation_points'] = 5
    data = generate_case(config, track, dataset, 1103)
    assert np.all((data['initial_states'] >= 1) & (data['initial_states'] <= 7))
    assert np.all(data['physical_ground_truth'] > 0)


def test_described_architecture_and_time_conditioning():
    model = PaperTrajectoryField(2)
    linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(3,64),(64,258),(258,258),(258,258),(258,2)]
    assert model(torch.tensor([0., 1.]), torch.ones(2, 2)).shape == (2, 2)


@pytest.mark.parametrize('method,mode', [('bb','paper_literal'), ('bb','gaussian_consistent'), ('gp','paper_literal'), ('gp','gaussian_conditional')])
def test_flow_exact_updates_and_dopri5_prediction(method, mode):
    torch.set_num_threads(1)
    times = np.linspace(0, 1, 5); states = np.sin(times)[:, None]
    model = CFMTSPaperTrajectory(method=method, target_mode=mode, epochs=2, samples=2,
                                 grid_points=5, batch_size=3, gp_max_iterations=4)
    model.fit([(times, states)])
    assert model.optimizer_updates_ == 2 * math.ceil(10 / 3)
    assert len(model.loss_history_) == 2
    output = model.simulate([0.], times)
    assert output.shape == (5, 1) and np.isfinite(output).all()


def test_node_actual_adjoint_backward_and_budget():
    pytest.importorskip('torchdiffeq')
    torch.set_num_threads(1)
    times = np.array([0., .05, .11])
    model = CFMTSPaperTrajectory(method='node', epochs=2, batch_size=2)
    model.fit([(times, np.exp(times)[:, None]), (times, 2*np.exp(times)[:, None])])
    assert model.optimizer_updates_ == 2 and len(model.loss_history_) == 2
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.model.parameters())
    assert np.isfinite(model.simulate([1.], times)).all()
