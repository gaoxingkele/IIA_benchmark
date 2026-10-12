import ast
import inspect
from pathlib import Path
import pytest
from iia_benchmark.models.fm_crossad_attention_lifetime_v2 import attention_lifetime_ast
from scripts.flow_matching import run_crossad_PSM_attention_completion_v2 as controller
from scripts.flow_matching import replay_crossad_attention_lifetime_v5 as replay


def test_all_original_attention_operations_are_identical_after_removing_one_deletion():
    tree=ast.parse(Path('data/public_datasets/flow_matching/crossad_author_runtime_v2/models/CrossAD/Attention_Blocks.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScaledDotProductAttention')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    transformed=attention_lifetime_ast(ast.unparse(forward))
    deletions=[n for n in transformed.body[0].body if isinstance(n,ast.Delete)]
    assert len(deletions)==1 and ast.unparse(deletions[0])=='del attn_scores'
    transformed.body[0].body=[n for n in transformed.body[0].body if not isinstance(n,ast.Delete)]
    assert ast.dump(transformed.body[0])==ast.dump(forward)


@pytest.mark.parametrize('source',[
    'def forward(self,x): return x',
    'def forward(self,x):\n attn_weights = F.softmax(attn_scores, dim=-1)\n return attn_scores'])
def test_changed_or_not_final_score_use_is_rejected(source):
    with pytest.raises(ValueError):attention_lifetime_ast(source)


def test_entire_original_PSM_controller_keeps_all_complete_gates():
    tree=controller.main_ast()
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=='scripts.flow_matching.replay_crossad_attention_lifetime_v5':
            n.value='scripts.flow_matching.replay_crossad_residual_lifetime_v4'
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(controller.previous.main)))


def test_native_main_only_changes_the_fresh_worker_route():
    tree=replay.main_ast()
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=='scripts.flow_matching.replay_crossad_attention_lifetime_v5':
            n.value='scripts.flow_matching.replay_crossad_residual_lifetime_v4'
    assert ast.dump(tree)==ast.dump(replay.previous.main_ast())
