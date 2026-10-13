"""All original full replay statements/gates survive saved-graph construction."""
import ast
import copy
import inspect

import pytest

from scripts.flow_matching import timegan_stock1104_saved_graph_full_replay_v1 as current

CONFIG = 'configs/runtime/fm_timegan_stock1104_saved_graph_full_replay.v1.json'


def test_entire_original_worker_identical_except_generator_factory_binding():
    original = ast.parse(inspect.getsource(current.BASE.worker)); actual = current.worker_ast()
    for node in ast.walk(actual):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_saved_generator_constructor':
            node.func.id = 'constructor_prefix'
    assert ast.dump(actual) == ast.dump(original)


def test_all_original_checkpoint_evaluator_numeric_gates_keep_exact_code():
    namespace = current.native_namespace()
    for key in ['bitwise_array', 'bitwise_score', 'check_report', 'complete_metadata',
                'restore_full_checkpoint', 'numpy_state_matches', 'protocol_contract']:
        assert namespace[key].__code__ is getattr(current.BASE, key).__code__


def test_parent_changes_only_actual_native_child_module_route():
    expected = ast.parse(inspect.getsource(current.supplement.main)); actual = current.main_ast()
    for node in ast.walk(actual):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.timegan_stock1104_saved_graph_full_replay_v1',
                                            'scripts.flow_matching.timegan_saved_joint_full_replay_v1')
    assert ast.dump(actual) == ast.dump(expected)


def test_all45_routes_and_full_shape_budgets_remain():
    config = current.BASE.read(current.ROOT / CONFIG);queue, routes = current.verify(config)
    assert len(queue['jobs']) == len(routes) == 45
    assert routes == current.BASE.read(current.ROOT / current.PREVIOUS)['routes']
    assert config['required_checkpoints_each'] == 23 and config['required_evaluators_each'] == 20
    assert config['protocol_iterations_each'] == 70000


@pytest.mark.parametrize('field,value', [('required_checkpoints_each', 22), ('required_evaluators_each', 19),
    ('protocol_iterations_each', 69999), ('equality_gate', 'approximate')])
def test_no_partial_or_relaxed_replay_can_be_registered(field, value):
    config = copy.deepcopy(current.BASE.read(current.ROOT / CONFIG));config[field] = value
    with pytest.raises(ValueError, match='budget/guard/scope'):
        current.verify(config)
