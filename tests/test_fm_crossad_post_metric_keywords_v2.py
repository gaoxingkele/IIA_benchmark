import ast
import inspect
import math
import pytest
from scripts.flow_matching import replay_crossad_post_metric_keywords_v2 as corrected


def test_only_two_math_scalar_keyword_bindings_change_in_the_entire_original_review():
    tree=corrected.post_review_ast();changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='math.isclose':
            for keyword in node.keywords:
                if keyword.arg is None and ast.unparse(keyword.value)=='_math_tolerance()':
                    keyword.value=ast.Name(id='METRIC_TOLERANCE',ctx=ast.Load());changes+=1
    assert changes==2
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(corrected.original.post_review)))


def test_original_API_failure_is_reproduced_and_exact_frozen_values_are_preserved():
    with pytest.raises(TypeError):math.isclose(1.,1.,**corrected.original.METRIC_TOLERANCE)
    assert corrected.math_tolerance()==dict(rel_tol=1e-12,abs_tol=1e-12)


@pytest.mark.parametrize('a,b,expected',[(1.,1.+5e-13,True),(1.,1.+5e-11,False),(0.,5e-13,True),(0.,5e-11,False)])
def test_real_math_comparison_keeps_original_relative_and_absolute_limits(a,b,expected):
    assert math.isclose(a,b,**corrected.math_tolerance())==expected


def test_guarded_controller_only_changes_the_fresh_independent_worker_route():
    tree=corrected.stage_controller_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.replay_crossad_post_metric_keywords_v2':
            node.value='scripts.flow_matching.run_crossad_completion_replay_queue_v1'
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(corrected.original.stage_controller)))
