import ast
import inspect
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import timegan_joint_checkpoint_observer_v1 as observer
from scripts.flow_matching.timegan_stage_resume_v2 import resumed_ast, compile_observed_resume
from scripts.flow_matching import run_timegan_checkpointed_stage_recovery_v4 as runner


def test_entire_original_native_joint_loop_is_unchanged_after_removing_only_observation():
    module = SimpleNamespace(__file__='data/public_datasets/flow_matching/timegan_native_source_v1/original/timegan.py')
    original = resumed_ast(module)
    observed = observer.append_joint_observation(original)
    loop = next(n for n in observed.body[0].body if isinstance(n, ast.For) and getattr(n.target, 'id', None) == 'itt')
    assert ast.dump(loop.body.pop()) == ast.dump(ast.parse('_save_complete_joint_boundary(sess, itt, locals())').body[0])
    assert ast.dump(observed) == ast.dump(original)


def test_complete_original_observer_compiler_retained_with_only_boundary_and_existing_target_observation():
    original = ast.parse(inspect.getsource(compile_observed_resume))
    observed = observer.compiler_ast(inspect.getsource(compile_observed_resume))
    class Reverse(ast.NodeTransformer):
        def visit_Call(self, n):
            if isinstance(n.func, ast.Name) and n.func.id == '_resumed_function_with_checkpoint':
                n.func.id = 'resumed_function'; n.args = n.args[:2]
            if isinstance(n.func, ast.Name) and n.func.id == '_corrected_observer_ast':
                return self.visit(n.args[0])
            if isinstance(n.func, ast.Name) and n.func.id == 'dict':
                n.keywords = [k for k in n.keywords if k.arg != '_native_predictive_targets']
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(observed)) == ast.dump(original)


def test_original_resource_guards_and_full_worker_budget_are_retained():
    original = ast.parse(inspect.getsource(runner.original.main))
    observed = runner.main_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str):
                n.value = n.value.replace('scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4',
                                           'scripts.flow_matching.run_timegan_stage_resume_v2')
            return n
        def visit_Assign(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == '_cached_resource_api':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api', asname=None)], level=0)
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(observed)) == ast.dump(original)


@pytest.fixture
def checkpoint_mock(tmp_path, monkeypatch):
    # Structural persistence tests only; these mock values are never benchmark
    # or evidence of native full-model training.
    np_before, py_before = np.random.get_state(), random.getstate()
    variables = [SimpleNamespace(op=SimpleNamespace(name='variable_%d' % i),
                                 shape=SimpleNamespace(as_list=lambda: [1])) for i in range(284)]
    values = [np.asarray([float(i)], dtype=np.float32) for i in range(284)]
    session = SimpleNamespace(kind='timegan', counts=dict(E0=50000, GS=50000, G=2000, E=2000, D=998),
        parameters=123, recording=False, saver=None, run=lambda fetches: [v.copy() for v in values])
    saved = {}
    def saver_save(sess, prefix):
        saved.update({v.op.name: a.copy() for v, a in zip(variables, values)})
        for extension in ('.index', '.meta', '.data-00000-of-00001'):
            Path(prefix + extension).write_bytes(b'fixture only')
    reader = SimpleNamespace(get_variable_to_shape_map=lambda: {v.op.name: [1] for v in variables},
                             get_tensor=lambda name: saved[name])
    fake = SimpleNamespace(global_variables=lambda: variables,
        train=SimpleNamespace(Saver=lambda **kw: SimpleNamespace(save=saver_save),
                              NewCheckpointReader=lambda prefix: reader))
    monkeypatch.setitem(sys.modules, 'tensorflow', fake)
    monkeypatch.setattr(__import__('scripts.flow_matching.timegan_stage_resume_v2', fromlist=['stochastic_ancestors']),
                        'stochastic_ancestors', lambda roots: [])
    nodes = {name+'_solver': object() for name in ('E0','E','D','G','GS')}
    nodes['X_hat'] = SimpleNamespace(op=object())
    config = dict(joint_checkpoint_interval=1000, checkpoint_full_variable_count=284,
                  canonical_job_id='timegan_native__gru__sine__seed1103', native_restore_audit_sha256='fixture only')
    try:
        yield session, nodes, config, values, fake, saver_save, tmp_path
    finally:
        np.random.set_state(np_before); random.setstate(py_before)


def test_complete_boundary_saves_all_variables_rng_and_counters_without_changing_state(checkpoint_mock):
    session, nodes, config, values, fake, _, output = checkpoint_mock
    before_np, before_py, counts = np.random.get_state(), random.getstate(), dict(session.counts)
    observer.save_joint_boundary(session, 999, nodes, output, config)
    ledger = observer.read(output/'completed_full_joint_checkpoints.json')
    assert len(ledger) == 1 and ledger[0]['completed_joint_iterations'] == 1000
    assert ledger[0]['native_full_variable_count'] == 284 and ledger[0]['additional_optimizer_updates'] == 0
    assert session.counts == counts and not session.recording
    assert observer.equal_numpy_state(before_np, np.random.get_state(), np) and before_py == random.getstate()
    assert (output/'recovery_joint_checkpoints/iteration_001000/python_random_state.json').is_file()


def test_partial_iteration_has_no_checkpoint(checkpoint_mock):
    session, nodes, config, values, fake, _, output = checkpoint_mock
    observer.save_joint_boundary(session, 998, nodes, output, config)
    assert not list(output.iterdir())


@pytest.mark.parametrize('corruption', ['partial_counter','rng_changed','weights_changed','counters_changed','missing_variable'])
def test_observer_rejects_unrecoverable_or_state_changing_snapshots(checkpoint_mock, corruption):
    session, nodes, config, values, fake, saver_save, output = checkpoint_mock
    if corruption == 'partial_counter': session.counts['G'] -= 1
    if corruption == 'missing_variable': config['checkpoint_full_variable_count'] = 283
    if corruption in ('rng_changed','weights_changed','counters_changed'):
        def save(sess, prefix):
            saver_save(sess, prefix)
            if corruption == 'rng_changed': np.random.uniform()
            if corruption == 'weights_changed': values[0][0] += 1
            if corruption == 'counters_changed': session.counts['D'] += 1
        fake.train.Saver = lambda **kw: SimpleNamespace(save=save)
    with pytest.raises(ValueError): observer.save_joint_boundary(session, 999, nodes, output, config)
    assert not (output/'completed_full_joint_checkpoints.json').exists()
    assert not session.recording


def test_previous_full_boundary_is_never_overwritten(checkpoint_mock):
    session, nodes, config, values, fake, _, output = checkpoint_mock
    observer.save_joint_boundary(session, 999, nodes, output, config)
    with pytest.raises(FileExistsError): observer.save_joint_boundary(session, 999, nodes, output, config)
    assert len(observer.read(output/'completed_full_joint_checkpoints.json')) == 1
