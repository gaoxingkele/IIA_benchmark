import ast
import copy
from pathlib import Path

import pytest

from scripts.flow_matching.probe_tccm_full_image_execution_recovery_v1 import capacity_ast
from scripts.flow_matching.run_tccm_full_image_capacity_recovery_v1 import controller_ast
from scripts.flow_matching.run_tccm_full_image_recovery_queue_v1 import queue_ast


def main_node(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def test_native_capacity_only_changes_the_declared_TCCM_objective_import():
    source = Path('scripts/flow_matching/probe_full_image_cuda_execution_recovery_v2.py').read_text(encoding='utf-8')
    adapted = capacity_ast(source)
    compile(adapted, '<tccm-capacity>', 'exec')
    imports = [n for n in ast.walk(adapted) if isinstance(n, ast.ImportFrom)
        and n.module == 'iia_benchmark.models.fm_tccm_image']
    assert len(imports) == 1
    assert [(n.name, n.asname) for n in imports[0].names] == [('tccm_uniform_objective', 'image_cfm_objective')]
    imports[0].module = 'iia_benchmark.models.fm_supplemental'
    imports[0].names = [ast.alias(name='image_cfm_objective', asname=None)]
    assert ast.dump(adapted.body[0]) == ast.dump(main_node(source))


def test_controller_preserves_all_GPU_memory_caps_locks_and_checkpoint_checks():
    source = Path('scripts/flow_matching/run_full_image_cuda_capacity_queue_v2.py').read_text(encoding='utf-8')
    adapted = controller_ast(source)
    compile(adapted, '<tccm-capacity-controller>', 'exec')
    counts = dict(cases=0, worker=0, summary=0, state=0)
    for node in ast.walk(adapted):
        if isinstance(node, ast.List) and [getattr(n, 'value', None) for n in node.elts] == ['TCCM']:
            node.elts = [ast.Constant('WT'), ast.Constant('FM')]; counts['cases'] += 1
        if isinstance(node, ast.Constant):
            if node.value == 'scripts.flow_matching.probe_tccm_full_image_execution_recovery_v1':
                node.value = 'scripts.flow_matching.probe_full_image_cuda_execution_recovery_v2'; counts['worker'] += 1
            if node.value == 'full_native_tccm_batch32_capacity_passed':
                node.value = 'both_full_native_batch32_cuda_capacity_cases_passed'; counts['state'] += 1
        if isinstance(node, ast.keyword) and node.arg == 'full_native_capacity_cases':
            assert node.value.value == 1; node.value = ast.Constant(2); counts['summary'] += 1
    assert all(n == 1 for n in counts.values())
    assert ast.dump(adapted.body[0]) == ast.dump(main_node(source))


@pytest.mark.parametrize('adapt', [capacity_ast, controller_ast])
def test_missing_original_structure_is_rejected(adapt):
    with pytest.raises(ValueError):
        adapt('def main(): pass')


def test_full_five_seed_controller_only_changes_worker_execution_adapter():
    source = Path('scripts/flow_matching/run_tccm_full_image_training_queue_v1.py').read_text(encoding='utf-8')
    tree = queue_ast(source); compile(tree, '<full-tccm-queue>', 'exec')
    commands = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'command']
    assert len(commands) == 1
    command = commands[0].value
    assert command.elts[-2].value == '--recovery-config'
    assert ast.unparse(command.elts[-1]) == "runtime['recovery_configuration']"
    command.elts = command.elts[:-2]
    for n in command.elts:
        if isinstance(n, ast.Constant) and n.value == 'scripts.flow_matching.run_tccm_full_image_execution_recovery_v1':
            n.value = 'scripts.flow_matching.run_tccm_full_image_training_v1'
    assert ast.dump(tree.body[0]) == ast.dump(main_node(source))


def test_TCCM_training_entry_keeps_original50epoch_loop_and_loss():
    from scripts.flow_matching.run_full_image_training_execution_recovery_v3 import training_ast
    path = Path('scripts/flow_matching/run_tccm_full_image_training_v1.py')
    tree = training_ast(path)
    original = main_node(path.read_text(encoding='utf-8'))
    hook = [n for n in tree.body[0].body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == '_verify_execution_hook']
    assert len(hook) == 1
    tree.body[0].body.remove(hook[0])
    assert ast.dump(tree.body[0]) == ast.dump(original)


@pytest.mark.parametrize('objective', ['FM', 'WT'])
def test_other_objectives_cannot_certify_TCCM_native_capacity(objective):
    from scripts.flow_matching.run_tccm_full_image_execution_recovery_v1 import verify_capacity
    with pytest.raises(ValueError, match='cannot be replaced'):
        verify_capacity(dict(capacity_cases=[dict(objective=objective)]))


@pytest.mark.parametrize('bf16', [False, True])
def test_TCCM_loss_gradients_RNG_and_native_optimizer_survive_recomputation(bf16):
    import torch
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    from iia_benchmark.models.fm_tccm_image import tccm_uniform_objective
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read
    from scripts.flow_matching.full_image_training_execution_recovery_v2 import enable_native_activation_recomputation
    config = read(ROOT / 'configs/models/fm_wt_mvtec_controlled_image_domain.v1.json')
    parameters = dict(config['parameters']); parameters.update(model_channels=32, num_res_blocks=1,
        channel_mult=[1, 2], attention_resolutions=[1, 2], num_classes=2)
    torch.manual_seed(729); torch.set_num_threads(2)
    model = FlowMismatchUNet(**parameters, asset_root=ROOT)
    with torch.no_grad(): model.model.out[-1].weight.normal_(std=.01)
    recomputed = copy.deepcopy(model); enable_native_activation_recomputation(recomputed)
    opt = read(ROOT / 'configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json')
    optimizers = [build_verified_dinomaly_stableadamw(m.parameters(), **opt['parameters'], asset_root=ROOT)
        for m in (model, recomputed)]
    images = torch.randn(2, 3, 16, 16); classes = torch.tensor([0, 1])
    for _ in range(3):
        rng = torch.get_rng_state(); results = []
        for current, optimizer in zip((model, recomputed), optimizers):
            torch.set_rng_state(rng); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                loss = tccm_uniform_objective(current, images, labels=classes)
            loss.backward()
            gradients = [p.grad.clone() if p.grad is not None else None for p in current.parameters()]
            optimizer.step(); results.append((loss.detach(), gradients, torch.get_rng_state()))
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][2], results[1][2])
        for a, b in zip(results[0][1], results[1][1]):
            if a is None: assert b is None
            else: assert torch.allclose(a, b, rtol=1e-6, atol=1e-7)
        for a, b in zip(model.parameters(), recomputed.parameters()):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-7)
