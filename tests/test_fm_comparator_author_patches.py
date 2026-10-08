"""Behavior checks for author-source repairs, conditional on local snapshots."""
from pathlib import Path
import importlib.util
import ast
from types import SimpleNamespace
import pytest
import torch
from iia_benchmark.models.fm_comparator_author_patches import prepare_pi_transformer, prepare_maelnet_reward


ROOT = Path(__file__).resolve().parents[1]
PI = ROOT / "experiments/runs/fm_project_acquisition/sources/pi_transformer_author_history_mirror/original"
MAEL = ROOT / "experiments/runs/fm_code_completion/sources/maelnet/original"


@pytest.mark.skipif(not PI.exists(), reason="local preserved source required")
def test_pi_repair_enforces_identical_causal_support_and_preserves_raw(tmp_path):
    raw = (PI / "model/attn.py").read_bytes()
    target = tmp_path / "corrected"
    manifest = prepare_pi_transformer(PI, target)
    assert (PI / "model/attn.py").read_bytes() == raw
    assert manifest["original_sha256"] != manifest["corrected_sha256"]
    spec = importlib.util.spec_from_file_location("pi_test_attention", target / "model/attn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attention = module.PhaseSyncAttention(8, output_attention=True, device="cpu")
    q = torch.randn(2, 8, 2, 4, requires_grad=True)
    fields = torch.zeros(2, 8, 2)
    result = attention(q, q, q, fields, fields, fields, None)
    series, prior = result[1:3]
    assert torch.count_nonzero(series.triu(1)) == 0
    assert torch.count_nonzero(prior.triu(1)) == 0
    assert torch.isfinite(prior).all()
    result[0].square().mean().backward()
    assert torch.isfinite(q.grad).all()
    with pytest.raises(ValueError):
        prepare_pi_transformer(PI, target)


@pytest.mark.skipif(not MAEL.exists(), reason="local preserved source required")
@pytest.mark.parametrize("fn,fp", [(-1.5, -0.6), (-1.5, -0.8), (-2.0, -0.6), (-2.0, -0.8)])
def test_maelnet_four_paper_rewards_use_patched_author_method(tmp_path, fn, fp):
    target = tmp_path / "reward_variant"
    prepare_maelnet_reward(MAEL, target, fn, fp)
    tree = ast.parse((target / "utils/agentreward.py").read_text(encoding="utf8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_get_reward")
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "author_reward", "exec"), ns)
    # Execute only this dependency-free method; truth is environment state.
    assert ns["_get_reward"](SimpleNamespace(gtruth=[1], time_step=0), [0, 0, 1]) == 1
    assert ns["_get_reward"](SimpleNamespace(gtruth=[1], time_step=0), [0, 0, 0]) == fn
    assert ns["_get_reward"](SimpleNamespace(gtruth=[0], time_step=0), [0, 0, 1]) == fp
    assert ns["_get_reward"](SimpleNamespace(gtruth=[0], time_step=0), [0, 0, 0]) == 0.01


@pytest.mark.skipif(not MAEL.exists(), reason="local preserved source required")
def test_no_slow_learner_filters_author_runner_checkpoint_comprehension(tmp_path):
    target = tmp_path / "no_slow"
    prepare_maelnet_reward(MAEL, target, include_slow_learner=False)
    tree = ast.parse((target / "exp/opt_rl2_anomaly.py").read_text(encoding="utf8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "model_list" for t in node.targets))
    candidates = ["checkpoint_AnomalyTransformer.pth", "checkpoint_slow_learner_MaelNet.pth", "checkpoint_DCDetector.pth"]
    actual = eval(compile(ast.Expression(node.value), "author_model_list", "eval"),
                  {"os": SimpleNamespace(listdir=lambda _: candidates), "model_path": "unused"})
    assert actual == ["checkpoint_AnomalyTransformer.pth", "checkpoint_DCDetector.pth"]
