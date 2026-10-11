"""Target observation and controller handoff checks; no benchmark claims."""
import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import run_timegan_native_job_v1 as legacy
from scripts.flow_matching import timegan_stage_resume_v2 as resume
from scripts.flow_matching import timegan_predictive_observer_v2 as corrected
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import (
    allowed_console_child, handoff_phase_allowed, live_identity)


def tree():
    return ast.parse(inspect.getsource(legacy.run))


def test_exactly_one_observation_expression_changes_and_every_other_statement_is_preserved():
    original = tree()
    repaired = corrected.corrected_observer_ast(original)
    expected = ast.dump(ast.parse("_native_predictive_targets(feed, self.native_nodes['Y'], inspect.currentframe().f_back.f_locals, result, np)", mode='eval').body)
    restore = ast.parse("np.asarray(feed[self.native_nodes['Y']])", mode='eval').body
    count = [0]

    class Reverse(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == expected:
                count[0] += 1
                return copy.deepcopy(restore)
            return self.generic_visit(node)

    assert ast.dump(Reverse().visit(repaired)) == ast.dump(original)
    assert count == [1]


@pytest.mark.parametrize('source', ['def run(): pass', "def run():\n a=np.asarray(feed[self.native_nodes['Y']])\n b=np.asarray(feed[self.native_nodes['Y']])"])
def test_missing_or_duplicate_recording_expressions_rejected(source):
    with pytest.raises(ValueError):
        corrected.corrected_observer_ast(ast.parse(source))


def observed_class(module_tree, tmp_path, predictions):
    observed = next(n for n in ast.walk(module_tree) if isinstance(n, ast.ClassDef) and n.name == 'ObservedSession')

    class NativeInference:
        def run(self, *args, **kwargs):
            return predictions

    namespace = dict(legacy.__dict__, original_session=NativeInference, np=np,
        context={'repeat': 0}, output=tmp_path, expected_shape=list(predictions.shape),
        _native_predictive_targets=corrected.native_predictive_targets)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[observed], type_ignores=[])), '<observation-test>', 'exec'), namespace)
    session = namespace['ObservedSession'].__new__(namespace['ObservedSession'])
    session.kind = 'predictive'
    session.recording = False
    session.ops = {}
    session.native_nodes = dict(Y=object(), y_pred=object())
    session.counts = {'evaluator': 0}
    return session


def author_inference(session, targets):
    # Same relevant untouched author caller contract: Y_mb local, only X/T feed.
    Y_mb = targets
    return session.run(session.native_nodes['y_pred'], feed_dict={object(): 'X', object(): 'T'})


def test_actual_legacy_observer_failure_and_repaired_call_path(tmp_path):
    predictions = np.arange(18, dtype=np.float32).reshape(3, 6, 1) / 18
    targets = np.arange(18, dtype=np.float64).reshape(3, 6, 1)[[2, 0, 1]] / 21
    old = observed_class(tree(), tmp_path, predictions)
    with pytest.raises(KeyError):
        author_inference(old, targets)
    new = observed_class(corrected.corrected_observer_ast(tree()), tmp_path, predictions)
    assert np.array_equal(author_inference(new, targets), predictions)
    with np.load(tmp_path / 'predictive_00_full_test.npz', allow_pickle=False) as artifact:
        assert np.array_equal(artifact['predictions'], predictions)
        assert np.array_equal(artifact['targets'], targets)
        assert artifact['targets'].dtype == targets.dtype
    assert new.counts == {'evaluator': 0}


@pytest.mark.parametrize('caller,predictions,feed_target', [
    ({}, np.zeros((2, 3, 1)), False),
    ({'Y_mb': np.zeros((1, 3, 1))}, np.zeros((2, 3, 1)), False),
    ({'Y_mb': np.full((2, 3, 1), np.nan)}, np.zeros((2, 3, 1)), False),
    ({'Y_mb': np.zeros((2, 3, 1))}, np.zeros((2, 3, 1)), True),
])
def test_targets_must_follow_exact_author_contract(caller, predictions, feed_target):
    target = object()
    with pytest.raises(ValueError):
        corrected.native_predictive_targets({target: []} if feed_target else {}, target, caller, predictions, np)


def test_native_full_run_compiles_without_importing_numerical_runtime():
    function = corrected.corrected_native_run()
    assert callable(function) and function.__name__ == 'run'


def test_resume_preserves_stage_restore_full_loop_and_full_result_verifier():
    original_source = Path(resume.__file__).read_bytes()
    original_observer = Path(legacy.__file__).read_bytes()
    assert callable(corrected.corrected_resume_compiler())
    config = corrected.read(corrected.ROOT / 'configs/reproducibility/fm_timegan_stock_stage_recovery.2026-10-11_v3.json')
    function, queue, job = corrected.corrected_resume_compiler()(config)
    assert callable(function)
    assert len(queue['jobs']) == 45
    assert job == dict(config['canonical_job'], output_directory=config['output_directory'])
    assert function.__globals__['_prior'] == dict(E0=50000, GS=50000, G=0, E=0, D=0)
    assert function.__globals__['_provenance']['resume_config'] == config
    assert Path(resume.__file__).read_bytes() == original_source
    assert Path(legacy.__file__).read_bytes() == original_observer


def test_real_author_predictive_call_has_no_Y_feed_and_has_Y_mb_targets():
    model = corrected.read(corrected.ROOT / 'configs/models/fm_timegan_native_original.v1.json')
    source = corrected.ROOT / model['source_root'] / 'metrics/predictive_metrics.py'
    native = ast.parse(source.read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(native) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'run' and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'y_pred']
    assert len(calls) == 1
    feed = next(k.value for k in calls[0].keywords if k.arg == 'feed_dict')
    assert [n.id for n in feed.keys] == ['X', 'T']
    assert any(isinstance(n, ast.Name) and n.id == 'Y_mb' for n in ast.walk(native))


@pytest.mark.parametrize('role', ['original', 'stock', 'energy'])
@pytest.mark.parametrize('phase', ['starting', 'waiting_for_memory', 'between_full_jobs'])
def test_only_idle_verified_handoff_phases_allowed(role, phase):
    assert handoff_phase_allowed(role, {'state': phase, 'active_pid': None}, False, [])


@pytest.mark.parametrize('role', ['original', 'stock', 'energy'])
def test_live_native_child_always_blocks_parent_handoff(role):
    assert not handoff_phase_allowed(role, {'state': 'waiting_for_memory', 'active_pid': 23}, False, [])
    assert not handoff_phase_allowed(role, {'state': 'waiting_for_memory', 'active_pid': None}, False,
        [dict(pid=23, name='python.exe', exe='F:/native/python.exe')])


@pytest.mark.parametrize('role', ['stock', 'energy'])
def test_existing_recovery_attempt_never_overwritten(role):
    assert not handoff_phase_allowed(role, {'state': 'waiting_for_memory', 'active_pid': None}, True, [])


@pytest.mark.parametrize('name,executable', [('python.exe', 'C:/Windows/System32/conhost.exe'),
    ('conhost.exe', 'F:/arbitrary/conhost.exe'), ('conhost.exe', '')])
def test_non_native_console_children_cannot_be_ignored(name, executable):
    assert not allowed_console_child(name, executable)


def test_native_windows_console_child_is_distinguished():
    assert allowed_console_child('conhost.exe', 'C:\\Windows\\System32\\conhost.exe')


@pytest.mark.parametrize('created,command', [(9, ['expected']), (10, ['other'])])
def test_pid_reuse_and_command_changes_are_not_live_owner_identity(created, command):
    process = SimpleNamespace(create_time=lambda: created, cmdline=lambda: command)
    assert not live_identity(process, dict(create_time=10, command=['expected']))


@pytest.mark.parametrize('state', ['running_full_original_timegan', 'unknown', 'completed_with_unresolved_jobs'])
def test_active_or_unknown_phases_not_handoff_candidates(state):
    assert not handoff_phase_allowed('original', {'state': state, 'active_pid': None}, False, [])
