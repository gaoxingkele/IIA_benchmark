import ast
import inspect
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s


def test_full_rng_roundtrip_restores_python_numpy_and_torch_and_rejects_any_change(tmp_path):
    before = s.capture_RNG(torch, np, cuda=False)
    try:
        random.seed(111)
        np.random.seed(222)
        torch.manual_seed(333)
        snapshot = s.capture_RNG(torch, np, cuda=False)
        path = tmp_path / 'RNG.pt'
        torch.save(snapshot, path)
        expected = (random.random(), np.random.normal(size=101), torch.randn(103))
        state = torch.load(path, weights_only=False)
        s.restore_RNG(state, torch, np, cuda=False)
        actual = (random.random(), np.random.normal(size=101), torch.randn(103))
        assert expected[0] == actual[0]
        assert expected[1].tobytes() == actual[1].tobytes()
        assert torch.equal(expected[2], actual[2])
        assert not s.same_RNG(snapshot, s.capture_RNG(torch, np, cuda=False), torch, np)
    finally:
        s.restore_RNG(before, torch, np, cuda=False)


def test_complete_generated_arrays_require_surplus_rows_all_features_dtype_and_bytes():
    first = np.arange(10005 * 24 * 5, dtype=np.float64).reshape(10005, 24, 5)
    s.assert_same_samples(first, first.copy(), [10005, 24, 5], np)
    for value in [first[:-1], first.astype(np.float32)]:
        with pytest.raises(ValueError):
            s.assert_same_samples(first, value, [10005, 24, 5], np)
    for value in [np.nan, -1.0]:
        other = first.copy()
        other[-1, -1, -1] = value
        with pytest.raises(ValueError):
            s.assert_same_samples(first, other, [10005, 24, 5], np)


def test_model_ema_optimizer_state_compares_every_dtype_value_and_key():
    expected = dict(model=dict(w=torch.tensor([1., 2.])), opt=dict(state={1: dict(step=torch.tensor(12000.))}))
    s.assert_tree_equal(expected, expected, torch)
    for actual in [dict(model=dict(w=torch.tensor([1., 3.])), opt=expected['opt']),
                   dict(model=dict(w=torch.tensor([1., 2.], dtype=torch.float64)), opt=expected['opt']),
                   dict(model=expected['model'], opt=dict(state={})),
                   dict(model=expected['model'], opt=dict(state={1: dict(step=torch.tensor(12001.))}))]:
        with pytest.raises(ValueError):
            s.assert_tree_equal(actual, expected, torch)


def test_original_evaluator_loop_executes_all_four_by_five_on_full_population(monkeypatch, tmp_path):
    # Full shapes, fake evaluator functions: a dispatch test, never benchmark data.
    calls = []
    graph_seeds = []
    fake_tf1 = SimpleNamespace(reset_default_graph=lambda: None, set_random_seed=graph_seeds.append)
    monkeypatch.setitem(globals(), 'tf1', fake_tf1)
    def context(truth, fake):
        calls.append(('context', len(truth), len(fake)))
        return .2
    class Correlation:
        def __init__(self, truth, name): self.truth = truth
        def compute(self, fake):
            calls.append(('correlational', len(self.truth), len(fake)))
            return torch.tensor(.1)
    def discriminative(truth, fake):
        tf1.reset_default_graph()
        calls.append(('discriminative', len(truth), len(fake)))
        return .3, None
    def predictive(truth, fake):
        tf1.reset_default_graph()
        calls.append(('predictive', len(truth), len(fake)))
        return .4
    utils = ModuleType('utils')
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, 'utils', utils)
    monkeypatch.setitem(sys.modules, 'tensorflow', ModuleType('tensorflow'))
    for name, symbol, value in [('context_fid', 'context_fid', context),
        ('cross_correlation', 'CrossCorrelLoss', Correlation),
        ('discriminative_metric', 'discriminative_score_metrics', discriminative),
        ('predictive_metric', 'predictive_score_metrics', predictive)]:
        module = ModuleType('utils.' + name)
        setattr(module, symbol, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    truth = np.zeros((10000, 24, 5), dtype=np.float64)
    generated = np.zeros((10005, 24, 5), dtype=np.float64)
    path = tmp_path / 'reference.npy'
    np.save(path, truth, allow_pickle=False)
    result = s.original_evaluators()(tmp_path, dict(evaluation_seed=0, evaluation_repeats=5), generated, path, torch, np, lambda seed: None)
    assert len(calls) == 20
    assert sum(name == 'context' and a == b == 10000 for name, a, b in calls) == 5
    assert sum(name == 'correlational' and a == b == 2000 for name, a, b in calls) == 5
    for family in ['discriminative', 'predictive']:
        assert sum(name == family and a == b == 10000 for name, a, b in calls) == 5
    assert graph_seeds == [0, 1, 2, 3, 4] * 2
    assert all(len(v['raw_repeats']) == 5 for v in result['metrics'].values())


def test_original_sampling_calls_and_evaluator_budgets_are_not_reduced():
    tree = ast.parse(inspect.getsource(s.worker))
    samples = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'sample']
    assert len(samples) == 2
    for call in samples:
        assert ast.dump(call) == ast.dump(ast.parse("trainer.sample(num=len(data), size_every=job['generation_batch_size'], shape=[data.window, data.var_num])", mode='eval').body)
    c = dict(settings=dict(minimum_commit_headroom_bytes=20 * 2**30, emergency_commit_headroom_bytes=12 * 2**30),
             spectral_flow_minimum_commit_headroom_bytes=48 * 2**30)
    assert s.case_settings(c, dict(model='diffusion_ts')) == c['settings']
    assert s.case_settings(c, dict(model='spectral_flow'))['minimum_commit_headroom_bytes'] == 48 * 2**30
    assert s.case_settings(c, dict(model='spectral_flow'))['emergency_commit_headroom_bytes'] == 12 * 2**30


def test_only_complete_five_training_seed_groups_produce_comparisons():
    cases, verified = {}, {}
    for group in range(14):
        for seed in range(42, 47):
            identity = f'{group}_{seed}'
            cases[identity] = dict(job=dict(model='m', dataset=str(group), data_track='original', seed=seed))
            verified[identity] = dict(original_vs_new_generation={name: dict(difference=.01 * (seed - 41))
                for name in ['context_fid', 'correlational', 'discriminative', 'predictive']})
    verified.pop('0_46')
    summary = s.full_cohort_comparisons(cases, verified)
    assert summary['complete_cohorts'] == 13
    assert 'paired_training_seed_metric_differences' not in summary['cohorts']['m/0/original']
    assert summary['cohorts']['m/1/original']['paired_training_seed_metric_differences']['predictive']['mean'] == pytest.approx(.03)


def test_controller_module_remains_cold_without_numpy_torch_or_tensorflow():
    code = ('import sys;from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1;'
            'assert not any(k in sys.modules for k in ["numpy","torch","tensorflow"])')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=s.ROOT, check=True)
