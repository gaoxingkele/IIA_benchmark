import ast
import copy
from pathlib import Path
import shlex

import numpy as np
import pytest

from scripts.flow_matching.gtgan_energy_native_common_v1 import recipes, release_recipes, full_budget, verify_recipe
from scripts.flow_matching.register_gtgan_native_commands_v1 import native_arguments
from scripts.flow_matching.audit_gtgan_energy_native_v1 import original_loader_module
from scripts.flow_matching.audit_gtgan_native_models_v1 import original_gtgan_branch
from scripts.flow_matching.prepare_gtgan_native_runtime_v1 import drop_unused_imports
from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import recompute_score

SOURCE = Path('data/public_datasets/flow_matching/gtgan_native_source_v1/original')


def case(rate, arguments):
    return dict(missing_rate=rate, original_seed=7777, source_entry=str(SOURCE / 'GTGAN_energy.py'),
        author_arguments=arguments, parsed_original_arguments=native_arguments(SOURCE, 'GTGAN_energy.py', arguments))


def test_both_commands_are_exact_original_readme_lines_with_original_defaults():
    lines = (SOURCE / 'README.md').read_text(encoding='utf-8').splitlines()
    published = [shlex.split(line)[2:] for line in lines if line.startswith('python GTGAN_energy.py ') and '--train' in line]
    assert published == [a for _, a in recipes()]
    for rate, arguments in recipes():
        expected = verify_recipe(case(rate, arguments), SOURCE)
        assert expected['native_embedding_steps'] == 5000
        assert expected['full_generation_windows'] == 19735 - 24 + 1 == 19712
        assert expected['variables'] == 28
        assert expected['evaluator_repeats_per_metric'] == 10
        assert expected['discriminative_evaluator_steps'] == 300
        assert expected['predictive_evaluator_steps'] == 5000


def test_both_release_commands_are_exact_readme_lines_and_have_no_generator_training():
    lines = (SOURCE / 'README.md').read_text(encoding='utf-8').splitlines()
    published = [shlex.split(line)[2:] for line in lines if line.startswith('python GTGAN_energy.py ') and '--train' not in line]
    assert published == [a for _, a in release_recipes()]
    for rate, arguments in release_recipes():
        release = dict(case(rate, arguments), execution_mode='native_release_checkpoint_evaluation')
        expected = verify_recipe(release, SOURCE)
        assert expected['native_joint_steps'] == expected['embedding_optimizer_updates'] == expected['generator_optimizer_updates'] == 0
        assert expected['expected_native_optimizers'] == 0
        assert expected['full_periodic_evaluations'] == 1
        assert expected['full_generation_windows'] == 19712


def test_original_energy_optimizer_budgets_do_not_use_stock_pretrained_shortcuts():
    a, b = [full_budget(case(rate, arguments)) for rate, arguments in recipes()]
    assert (a['embedding_optimizer_updates'], a['generator_optimizer_updates'], a['full_periodic_evaluations']) == (17000, 9000, 12)
    assert (b['embedding_optimizer_updates'], b['generator_optimizer_updates'], b['full_periodic_evaluations']) == (22000, 12750, 17)
    tree = ast.parse((SOURCE / 'GTGAN_energy.py').read_text(encoding='utf-8'))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'load_state_dict' for n in ast.walk(train))


@pytest.mark.parametrize('key,value', [('first_epoch', 5), ('max_steps', 60), ('batch_size', 32),
    ('hidden_dim', 8), ('d_layer', 1), ('log_time', 1), ('max_steps_metric', 1)])
def test_reduced_or_stock_architecture_and_budget_are_rejected(key, value):
    original = case(*recipes()[0])
    changed = copy.deepcopy(original)
    changed['parsed_original_arguments'][key] = value
    with pytest.raises(ValueError, match='budget'):
        full_budget(changed)


def test_entire_original_energy_loader_ast_changes_only_two_device_calls():
    source = (SOURCE / 'time_dataset.py').read_text(encoding='utf-8')
    expected = ast.Module(body=[n for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in
        {'normalize', 'to_tensor', 'load_data', 'save_data', 'TimeDataset'}], type_ignores=[])
    moved = original_loader_module(source)
    count = 0
    for node in ast.walk(moved):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'cpu'):
            node.func.attr = 'cuda'
            node.args = []
            count += 1
    assert count == 2
    assert ast.dump(moved, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_energy_runtime_optional_import_removal_preserves_full_training_and_networks():
    s = (SOURCE / 'GTGAN_energy.py').read_text(encoding='utf-8')
    fixed, changes = drop_unused_imports(s, {'gru_ode_bayes.data_utils'})
    assert len(changes) == 1
    before = [ast.dump(n) for n in ast.parse(s).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    after = [ast.dump(n) for n in ast.parse(fixed).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert before == after
    assigned = {t.id for n in original_gtgan_branch(s).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {'embedder', 'generator', 'supervisor', 'recovery', 'discriminator'} <= assigned


def test_native_energy_discriminator_uses_true_energy_branch_300_updates():
    source = ast.parse((SOURCE / 'GTGAN_energy.py').read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(source) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == 'discriminative_score_metrics']
    assert calls and all(isinstance(n.args[2], ast.Constant) and n.args[2].value is True for n in calls)
    metric = ast.parse((SOURCE / 'metrics/discriminative_metrics.py').read_text(encoding='utf-8'))
    branch = next(n for n in ast.walk(metric) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'energy')
    iterations = next(n.value.value for n in branch.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'iterations' for t in n.targets))
    assert iterations == 300


def test_saved_native_evaluator_predictions_can_recompute_both_source_scores():
    real = np.array([[.8], [.3]])
    fake = np.array([[.1], [.2]])
    assert recompute_score('discriminative', [real, fake]) == .25
    targets = np.array([[[0.], [1.]], [[2.], [3.]]])
    assert recompute_score('predictive', targets + .125, targets) == .125
    with pytest.raises(ValueError, match='targets'):
        recompute_score('predictive', targets)
