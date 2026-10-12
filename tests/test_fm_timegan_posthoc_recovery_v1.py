"""Recovery attribution, original evaluator preservation and full45 route gates."""
import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_timegan_native_job_v1 as original
from scripts.flow_matching import timegan_posthoc_recovery_v1 as recovery
from scripts.flow_matching import run_timegan_full45_completion_replay_v2 as replay
from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_observer_ast


def config():
    return recovery.read(recovery.ROOT / 'configs/reproducibility/fm_timegan_stock1104_posthoc_recovery.2026-10-12_v1.json')


def evaluator(tree):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'evaluator')


def test_original_predictor_training_checkpoint_and_score_path_is_unchanged():
    before = evaluator(corrected_observer_ast(ast.parse(inspect.getsource(original.run))))
    after = evaluator(recovery.posthoc_ast())
    after.body.pop(0)
    assert ast.dump(before) == ast.dump(after)


def test_only_corrected_observer_is_compiled_author_files_are_not_modified():
    files = [recovery.ROOT / 'scripts/flow_matching/run_timegan_native_job_v1.py',
             recovery.ROOT / 'scripts/flow_matching/run_timegan_full45_completion_replay_v1.py']
    before = [p.read_bytes() for p in files]
    function, queue, job = recovery.compile_recovery(config())
    assert callable(function) and len(queue['jobs']) == 45
    assert job['seed'] == 1104 and job['id'] == config()['canonical_job']['id']
    assert len(function.__globals__['_prior_checkpoints']) == 13
    assert [p.read_bytes() for p in files] == before


@pytest.mark.parametrize('mutation', ['seed', 'guard', 'scope', 'archive', 'checkpoint', 'terminal', 'output'])
def test_mismatched_or_incomplete_predecessor_is_rejected(mutation):
    value = copy.deepcopy(config())
    if mutation == 'seed': value['canonical_job']['seed'] = 1105
    elif mutation == 'guard': value['settings']['minimum_commit_headroom_bytes'] -= 1
    elif mutation == 'scope': value['original_scope']['method_baseline_ablation_records'] = 680
    elif mutation == 'archive': value['archive_receipts'].pop()
    elif mutation == 'checkpoint': value['checkpoint_receipts'].pop()
    elif mutation == 'terminal': value['terminal_receipt']['sha256'] = '0' * 64
    else: value['output_directory'] = value['canonical_job']['output_directory']
    with pytest.raises(ValueError): recovery.binding(value)


def test_native_graph_seeds_attribute_reused_graphs_before_fresh_predictors():
    assert recovery.graph_record(1104, 0) == dict(index=0, kind='timegan', repeat=None, seed=1104)
    assert recovery.graph_record(1104, 10) == dict(index=10, kind='discriminative', repeat=9, seed=11194)


def test_full45_numerical_worker_and_completion_gates_use_identical_code_objects():
    namespace = replay.runtime_namespace()
    for name in ('worker', 'complete_metadata', 'check_report', 'restore_full_checkpoint', 'bitwise_array', 'bitwise_score'):
        assert namespace[name].__code__ is getattr(replay.base, name).__code__
    assert namespace['worker'].__globals__ is namespace
    assert namespace['worker'].__globals__['completed_native_result'] is replay.completed_native_result


@pytest.mark.parametrize('state,children,allowed', [
    (dict(state='waiting_actual_complete_original_full45_native_results', active_pid=None, active_job=None,
          verified_original_ids=[], failed={}), [], True),
    (dict(state='waiting_actual_complete_original_full45_native_results', active_pid=23), [], False),
    (dict(state='waiting_actual_complete_original_full45_native_results', active_pid=None, active_job='a'), [], False),
    (dict(state='running_complete_full45_native_case_replay', active_pid=None), [], False),
    (dict(state='waiting_actual_complete_original_full45_native_results', verified_original_ids=['a']), [], False),
    (dict(state='waiting_actual_complete_original_full45_native_results', failed={'a':'b'}), [], False),
    (dict(state='waiting_actual_complete_original_full45_native_results'),
     [dict(name='python.exe', exe='F:/native/python.exe')], False),
])
def test_handoff_never_stops_native_child_or_discards_completion_history(state, children, allowed):
    assert replay.handoff_phase_allowed('original', state, False, children) is allowed


def test_registered_route_preserves_all45_original_models_and_full_replay_obligations():
    value = recovery.read(recovery.ROOT / 'configs/runtime/fm_timegan_full45_completion_replay.v2.json')
    queue = recovery.read(recovery.ROOT / value['original_queue'])
    routes = replay.check_routes(queue, value)
    assert len(routes) == 45 and sum(r['kind'] == 'posthoc_recovery' for r in routes) == 1
    assert value['required_checkpoints_each'] == 23 and value['required_evaluators_each'] == 20
    assert value['equality_gate'] == 'bitwise' and value['protocol_iterations_each'] == 70000


def test_full_saved_native_arrays_and_discriminator_scores_match():
    import numpy as np
    value = config()
    queue, model, job, records, evaluators = recovery.binding(value)
    real, generated = recovery.verify_saved_arrays(value, model, job, evaluators, np)
    assert real.shape == generated.shape == (3661, 24, 6)


def test_reuse_restores_exact_saved_NumPy_state_without_adding_training_updates(tmp_path):
    import numpy as np
    value = config()
    function, queue, job = recovery.compile_recovery(value)
    namespace = function.__globals__
    canonical = value['canonical_job']
    with np.load(str(recovery.ROOT / canonical['output_directory'] / 'full_native_real_and_generated.npz'), allow_pickle=False) as archive:
        real = archive['real'].copy()
    context = dict(graph_seeds=[], completed_evaluators=[])
    original_state = np.random.get_state()
    try:
        generated = namespace['_reuse_generator'](real, dict(module='gru', hidden_dim=24, num_layer=3,
            iterations=50000, batch_size=128), tmp_path, context)
        for repeat in range(10):
            namespace['_reuse_discriminator'](real, generated, tmp_path, context)
        expected = recovery.numpy_state(recovery.ROOT / canonical['output_directory'] /
            'checkpoints/discriminative_09_complete/numpy_random_state.npz')
        replay.base.numpy_state_matches(np.random.get_state(), expected, np)
        assert len(context['graph_seeds']) == 11 and len(context['completed_evaluators']) == 10
        assert context['graph_seeds'][-1]['seed'] == 11194
        assert len(list(tmp_path.glob('*.npz'))) == 12
        assert len(namespace['_prior_checkpoints']) == 13
        with pytest.raises(ValueError): namespace['_reuse_discriminator'](real, generated, tmp_path, context)
    finally:
        np.random.set_state(original_state)
