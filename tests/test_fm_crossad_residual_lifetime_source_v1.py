import ast
import inspect
from pathlib import Path

import pytest

from iia_benchmark.models.fm_crossad_reference_lifetime_v1 import residual_lifetime_ast
from scripts.flow_matching import replay_crossad_residual_lifetime_v4 as runner


@pytest.mark.parametrize('kind,uses',[('EncoderLayer',1),('DecoderLayer',2)])
def test_only_final_use_reference_deletions_are_added_to_exact_original_source(kind,uses):
    tree=ast.parse(Path('data/public_datasets/flow_matching/crossad_author_runtime_v2/models/CrossAD/EncDec.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==kind)
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    actual=residual_lifetime_ast(ast.unparse(function),uses)
    assert sum(isinstance(n,ast.Delete) for n in actual.body[0].body)==uses
    actual.body[0].body=[n for n in actual.body[0].body if not isinstance(n,ast.Delete)]
    assert ast.dump(actual.body[0])==ast.dump(function)


def test_unexpected_residual_source_cannot_be_silently_optimized():
    with pytest.raises(ValueError):residual_lifetime_ast('def forward(self,x): return x',1)


def test_original_full_replay_controller_keeps_every_guard_and_gate():
    actual=runner.main_ast();expected=ast.parse(inspect.getsource(runner.original.main))
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='scripts.flow_matching.replay_crossad_residual_lifetime_v4':
                n.value='scripts.flow_matching.replay_crossad_registered_model_v3'
            return n
        def visit_Call(self,n):
            if getattr(n.func,'id',None)=='_cached_resource_api':
                return ast.parse('original.resource_api()',mode='eval').body
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(actual))==ast.dump(expected)
