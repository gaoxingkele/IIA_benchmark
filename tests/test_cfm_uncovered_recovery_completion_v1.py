import ast
import copy
import inspect
import pytest
from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as current

CONFIG='configs/runtime/fm_cfm_all8_uncovered_recovery_full285_completion.v1.json'


def test_full285_routes_change_only_eight_new_execution_outputs():
    runtime=current.original.read(current.ROOT/CONFIG)
    previous=current.original.read(current.ROOT/current.PREVIOUS)
    production=current.original.read(current.ROOT/current.PRODUCER)
    selected,recovery=current.producer.verify(production)
    assert runtime['routes']==current.expected_routes(previous,recovery)
    assert len(runtime['routes'])==285 and len(selected)==8
    assert len([1 for a,b in zip(runtime['routes'],previous['routes']) if a!=b])==8
    assert runtime['settings']==previous['settings']


def test_unselected_original_completed_results_cannot_be_duplicated():
    runtime=current.original.read(current.ROOT/CONFIG)
    unselected=next(r for r in runtime['routes'] if r['original_canonical_id'] not in runtime['selected_canonical_ids'])
    assert not current.selected_ready(unselected,runtime)


@pytest.mark.parametrize('field,value',[('cpu_threads',1),('prediction_tolerance',{'rtol':1e-3,'atol':1e-3})])
def test_original_numerical_workers_and_guards_are_not_relaxed(field,value):
    runtime=copy.deepcopy(current.original.read(current.ROOT/CONFIG));runtime[field]=value
    with pytest.raises(ValueError,match='guard changed'):
        current.verify(runtime)


def test_native_and_independent_model_workers_use_original_bytecode():
    namespace=current.namespace()
    for key in ['review_worker','run_review','verify_record','native_registration','check_native_terminal']:
        assert namespace[key].__code__ is getattr(current.original,key).__code__


def test_original_controller_retained_after_reversing_only_host_adoption_and_candidate_gate():
    tree=current.controller_ast()
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While)
        and isinstance(n.test,ast.Constant) and n.test.value is True)
    loop.body=loop.body[3:]
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if getattr(node.func,'id',None)=='_selected_ready':
                return ast.parse("(ROOT/route['result_path']).is_file()",mode='eval').body
            return self.generic_visit(node)
    tree=Restore().visit(tree)
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(current.original.controller)))
