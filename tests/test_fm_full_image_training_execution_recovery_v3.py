import ast
from pathlib import Path
import pytest
from scripts.flow_matching.run_full_image_training_execution_recovery_v3 import training_ast


@pytest.mark.parametrize('method',['flow_mismatching','wt_full_image'])
def test_entire_original_full_training_entry_point_preserved_except_execution_hook(method):
    filename='run_flow_mismatching_full_training_v1.py' if method=='flow_mismatching' else 'run_wt_full_image_training_v1.py'
    path=Path('scripts/flow_matching')/filename
    source=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    tree=training_ast(path);compile(tree,'full_native_recovery','exec')
    recovered=tree.body[0]
    hooks=[n for n in recovered.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_verify_execution_hook']
    assert len(hooks)==1
    recovered.body.remove(hooks[0])
    assert ast.dump(source)==ast.dump(recovered)
    text=ast.unparse(tree)
    assert 'range(start + 1, 2001)' in text
    assert 'batch_size=32' in text
    assert 'global_steps != start * ((n + 31) // 32)' in text


def test_unknown_or_reduced_model_constructor_cannot_receive_full_native_execution_claim(tmp_path):
    path=tmp_path/'source.py'
    path.write_text('def main():\n    model = SmallUNet().to(device)\n',encoding='utf-8')
    with pytest.raises(ValueError):training_ast(path)
