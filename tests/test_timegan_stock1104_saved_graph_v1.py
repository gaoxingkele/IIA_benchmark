"""Graph hypothesis comparator and single full saved-graph execution limits."""
import ast
import copy
import inspect
from types import SimpleNamespace

import pytest

from scripts.flow_matching import audit_timegan_stock1104_saved_graph_v1 as audit

CONFIG = 'configs/runtime/fm_timegan_stock1104_saved_graph_diagnostic.v1.json'


class Attr:
    def __init__(self, value): self.value = value
    def SerializeToString(self, deterministic=False):
        assert deterministic
        return self.value.encode('utf-8')


def graph():
    node = lambda name, op, inputs: SimpleNamespace(name=name, op=op, input=inputs,
        device='', attr={'T': Attr('float32'), '_output_shapes': Attr('shape')})
    return SimpleNamespace(node=[node('input', 'Placeholder', []), node('weights', 'VariableV2', []),
        node('output', 'MatMul', ['input:0', 'weights:0']), node('unused', 'NoOp', [])], library=Attr('empty'))


def test_exact_full_forward_structure_compared_with_control_input_names():
    source, saved = graph(), graph()
    source.node[2].input.append('^unused'); saved.node[2].input.append('^unused')
    result = audit.graph_difference(source, saved, 'output:0')
    assert result['source_forward_nodes'] == 4
    assert result['forward_definitions_equal_excluding_output_shape_metadata']
    assert result['function_library_bitwise_equal']


@pytest.mark.parametrize('change', ['op', 'input', 'dtype', 'device', 'function'])
def test_semantic_forward_changes_are_not_hidden(change):
    source, saved = graph(), graph()
    if change == 'op': saved.node[2].op = 'Add'
    if change == 'input': saved.node[2].input = ['input:0', 'input:0']
    if change == 'dtype': saved.node[2].attr['T'] = Attr('float64')
    if change == 'device': saved.node[2].device = '/device:GPU:0'
    if change == 'function': saved.library = Attr('different_function_definition')
    result = audit.graph_difference(source, saved, 'output:0')
    assert not (result['forward_definitions_equal_excluding_output_shape_metadata'] and result['function_library_bitwise_equal'])


def test_shape_metadata_differences_retained_separately():
    source, saved = graph(), graph(); saved.node[2].attr['_output_shapes'] = Attr('different shape')
    result = audit.graph_difference(source, saved, 'output:0')
    assert result['shape_metadata_changes'] == ['output']
    assert result['forward_definitions_equal_excluding_output_shape_metadata']


def test_exactly_one_full_inference_and_no_solver_fetch():
    tree = ast.parse(inspect.getsource(audit.worker))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'session' and n.func.attr == 'run']
    inference = [n for n in calls if ast.dump(n.args[0]) == ast.dump(ast.parse("tensors['X_hat']", mode='eval').body)]
    assert len(inference) == 1
    assert all('solver' not in ast.dump(n.args[0]) for n in calls)
    assert 'full20_evaluator_replays_completed=False' in inspect.getsource(audit.worker)


def test_registered_original_scope_and_resource_limits_preserved():
    config = audit.read(audit.ROOT / CONFIG)
    queue, route, previous = audit.verify(config)
    assert len(queue['jobs']) == 45 and route['canonical_job']['id'] == config['canonical_job_id']
    assert config['settings'] == previous['settings']
    assert config['maximum_saved_graph_inference_calls'] == 1
    modified = copy.deepcopy(config); modified['maximum_saved_graph_inference_calls'] = 2
    with pytest.raises(ValueError, match='exactly one new graph hypothesis'):
        audit.verify(modified)
