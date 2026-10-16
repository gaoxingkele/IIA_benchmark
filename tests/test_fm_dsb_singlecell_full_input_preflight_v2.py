import ast
import numpy as np
import torch
from scripts.flow_matching import preflight_dsb_singlecell_full_inputs_v1 as previous
from scripts.flow_matching import preflight_dsb_singlecell_full_inputs_v2 as s


def test_isolated_v2_preserves_every_full_input_budget_and_guard():
    a, b = previous.registration(), s.registration()
    for key in ['cases', 'settings', 'resource_lock', 'environment_lock', 'python',
                'actual_cache_particles', 'actual_cache_steps', 'actual_cache_rows',
                'actual_training_batch', 'gamma', 'parameters_per_network', 'network_count',
                'full_training_models_pending', 'full_updates_per_model_pending', 'original_scope']:
        assert a[key] == b[key]
    assert b['gamma_tensor_dtype'] == 'float64' and b['optimizer_updates'] == 0
    assert b['state_root'] != a['state_root'] and b['artifact_root'] != a['artifact_root']


def test_original_numpy_schedule_double_precision_is_retained_exactly():
    gamma = .001
    author = torch.tensor(np.concatenate([np.linspace(gamma, gamma, 10), np.linspace(gamma, gamma, 10)]))
    repaired = torch.full((20,), gamma, dtype=torch.float64)
    previous_schedule = torch.full((20,), gamma)
    assert author.dtype == repaired.dtype == torch.float64
    assert torch.equal(author, repaired) and author.sum() == repaired.sum()
    assert previous_schedule.dtype == torch.float32 and previous_schedule.sum().double() != author.sum()
    source = (s.ROOT / s.registration()['source_root'] / 'bridge/runners/ipf.py').read_text()
    assert 'gammas = torch.tensor(gammas).to(self.device)' in source


def test_worker_diff_is_exactly_one_dtype_keyword():
    def worker(module):
        tree = ast.parse((s.ROOT / (module.MODULE.replace('.', '/') + '.py')).read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    old, new = worker(previous), worker(s)
    changed = 0
    for n in ast.walk(new):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.full':
            for k in list(n.keywords):
                if k.arg == 'dtype' and ast.unparse(k.value) == 'torch.float64':
                    n.keywords.remove(k)
                    changed += 1
    assert changed == 1 and ast.dump(old) == ast.dump(new)
