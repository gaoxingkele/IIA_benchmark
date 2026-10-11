import ast
from pathlib import Path
import pytest

from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v3 as entry


def test_host_controller_differs_only_by_the_compatible_child_entry_module():
    source=Path(entry.previous.original.__file__).read_text(encoding='utf-8')
    expected=entry.previous.dispatch_ast(source)
    matches=[node for node in ast.walk(expected) if isinstance(node,ast.Constant)
        and node.value=='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v2']
    assert len(matches)==1
    matches[0].value='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v3'
    assert ast.dump(entry.host_dispatch_ast(source))==ast.dump(expected)


def test_exact_native_worker_arguments_preserved():
    args=entry.native_worker_arguments(['--config','frozen.json','--worker','--proof-binding','binding.json'],'frozen.json')
    assert args.worker and args.config=='frozen.json' and args.proof_binding=='binding.json'


@pytest.mark.parametrize('change',['other_config','no_worker','no_binding','unexpected_flag'])
def test_native_worker_cannot_enter_a_different_or_partial_invocation(change):
    remaining=['--config','frozen.json','--worker','--proof-binding','binding.json']
    if change=='other_config':remaining[1]='different.json'
    elif change=='no_worker':remaining.remove('--worker')
    elif change=='no_binding':remaining=remaining[:3]
    else:remaining+=['--epochs','1']
    with pytest.raises((ValueError,SystemExit)):entry.native_worker_arguments(remaining,'frozen.json')


def test_native_worker_branch_returns_before_any_host_scheduling_ast():
    tree=ast.parse(Path(entry.__file__).read_text(encoding='utf-8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    position=next(i for i,node in enumerate(function.body) if isinstance(node,ast.If) and ast.unparse(node.test)=="'--worker' in remaining")
    branch=function.body[position]
    calls=[ast.unparse(node.func) for node in ast.walk(branch) if isinstance(node,ast.Call)]
    assert 'runtime.run' in calls and 'host_dispatch_ast' not in calls and 'previous.dispatch_ast' not in calls
    assert isinstance(branch.body[-1],ast.Return)
    assert any(isinstance(node,ast.Call) and ast.unparse(node.func)=='host_dispatch_ast'
        for statement in function.body[position+1:] for node in ast.walk(statement))
