import ast
import inspect
from pathlib import Path

import pytest

from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import observed_fit_ast,canonical_science
from scripts.flow_matching.run_cfm_checkpointed_original_job_v1 import worker_ast,original


def source_fit():
    tree=ast.parse(Path('src/iia_benchmark/models/fm_cfm_ts_original.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CFMTSPaperTrajectory')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')


def test_full_original_fit_optimizer_sampler_solver_and_gradient_order_are_preserved():
    expected=source_fit()
    observed=observed_fit_ast(ast.unparse(expected))
    class Reverse(ast.NodeTransformer):
        def visit_Expr(self,n):
            if (isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                    and getattr(n.value.func.value,'id',None)=='_epoch_lane'):
                return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if (isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                    and getattr(n.value.func.value,'id',None)=='_epoch_lane'):
                return None
            return self.generic_visit(n)
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='range' and n.args and getattr(n.args[0],'id',None)=='_first_epoch':
                n.args=n.args[1:]
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(observed).body[0])==ast.dump(expected)


def test_full_source_worker_changes_only_complete_epoch_observer_binding():
    observed=worker_ast(inspect.getsource(original.run))
    observed.body[0].body=[n for n in observed.body[0].body if not(
        isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and getattr(n.value.func,'id',None)=='_bind_complete_epoch_fit')]
    assert ast.dump(observed)==ast.dump(ast.parse(inspect.getsource(original.run)))


def test_missing_source_loop_or_fit_is_rejected():
    with pytest.raises(ValueError): observed_fit_ast('def fit(self): pass')
    with pytest.raises(ValueError): worker_ast('def run(queue,job): pass')


def test_execution_paths_never_change_scientific_identity():
    job=dict(id='source',epochs=400,seed=1105,model_overrides=dict(batch_size=258),
        output_directory='old',artifact_directory='old_artifacts')
    retry=dict(job,id='new',output_directory='new',artifact_directory='new_artifacts',recovery_original_job_id='source')
    assert canonical_science(job)==canonical_science(retry)
    retry['epochs']=92
    assert canonical_science(job)!=canonical_science(retry)
