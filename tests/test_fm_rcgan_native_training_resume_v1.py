import ast
from pathlib import Path
import pytest
from scripts.flow_matching.rcgan_native_training_resume_runtime_v1 import observer_ast,resume_experiment_ast
from scripts.flow_matching.rcgan_native_training_runtime_v2 import experiment_ast


def test_exact_full_observer_extensions_preserve_native_training_and_evaluator_budgets():
    tree=observer_ast()
    run=tree.body[0]
    compile(tree,'observer','exec')
    source=Path('scripts/flow_matching/rcgan_native_training_runtime_v2.py').read_text(encoding='utf-8')
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    old={n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in run.body if isinstance(n,ast.FunctionDef)}
    for name in ['full_loader','bandwidth','build_case','mixture']:
        assert ast.dump(old[name])==ast.dump(new[name])
    assert '_prior_epochs' in ast.unparse(tree)
    assert 'resume_segment_full_epochs=262' in ast.unparse(tree)
    assert 'full_original500_epoch_loop_completed=False' in ast.unparse(tree)


def test_source_loop_body_is_unchanged_and_original_end_epoch_is_not_shortened(tmp_path):
    path=tmp_path/'experiment.py'
    path.write_text('import eval\nfor epoch in range(num_epochs):\n    train(epoch)\n    evaluate(epoch)\n',encoding='utf-8')
    before=experiment_ast(path)
    after=resume_experiment_ast(path)
    old=next(n for n in before.body if isinstance(n,ast.For))
    new=next(n for n in after.body if isinstance(n,ast.For))
    assert ast.dump(ast.Module(body=old.body,type_ignores=[]))==ast.dump(ast.Module(body=new.body,type_ignores=[]))
    assert ast.unparse(new.iter)=='range(_start_epoch, num_epochs)'
    assert '_native_restore_before_loop(globals())' in ast.unparse(after)
    path.write_text('import eval\nfor epoch in range(50):\n    train(epoch)\n',encoding='utf-8')
    with pytest.raises(ValueError):resume_experiment_ast(path)
