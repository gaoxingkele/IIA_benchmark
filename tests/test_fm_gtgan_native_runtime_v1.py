import ast
import json
from pathlib import Path
import pytest
from scripts.flow_matching.prepare_gtgan_native_runtime_v1 import repair_entry, drop_unused_imports
from scripts.flow_matching.audit_gtgan_native_models_v1 import original_gtgan_branch
from scripts.flow_matching.run_gtgan_native_job_v1 import full_budget

SOURCE = Path('data/public_datasets/flow_matching/gtgan_native_source_v1/original')


def test_regular_runtime_repairs_restore_entire_original_ast():
    s = (SOURCE / 'GTGAN_stocks.py').read_text(encoding='utf-8')
    repaired, changes = repair_entry(s, True)
    assert len(changes) == 5
    for change in reversed(changes):
        repaired = repaired.replace(change['after'], change['before'])
    assert ast.dump(ast.parse(repaired)) == ast.dump(ast.parse(s))


def test_irregular_algorithm_needs_no_filesystem_changes():
    s = (SOURCE / 'GTGAN_stocks_irregular.py').read_text(encoding='utf-8')
    repaired, changes = repair_entry(s, False)
    assert repaired == s and changes == []


def test_stock_optional_imports_do_not_modify_any_data_function():
    s = (SOURCE / 'time_dataset.py').read_text(encoding='utf-8')
    repaired, changes = drop_unused_imports(s, {'torchaudio', 'sktime.utils.load_data'})
    assert len(changes) == 2
    a = [ast.dump(n) for n in ast.parse(s).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    b = [ast.dump(n) for n in ast.parse(repaired).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert a == b


def test_used_optional_module_cannot_be_removed():
    with pytest.raises(ValueError, match='used'):
        drop_unused_imports('import torchaudio\nx = torchaudio.load("a")\n', {'torchaudio'})


def test_native_constructor_keeps_all_five_full_networks():
    for filename in ('GTGAN_stocks.py', 'GTGAN_stocks_irregular.py'):
        branch = original_gtgan_branch((SOURCE / filename).read_text(encoding='utf-8'))
        assigned = {t.id for n in branch.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        assert {'embedder', 'recovery', 'generator', 'supervisor', 'discriminator'} <= assigned
        text = ast.unparse(branch)
        assert 'build_model_tabular_nonlinear' in text and 'NeuralCDE' in text
        assert 'Multi_Layer_ODENetwork' in text and 'set_cnf_options' in text


def test_original_full_commands_preserve_pretrain_reload_and_evaluator_budget():
    config = json.loads(Path('configs/models/fm_gtgan_native_original_stock.v1.json').read_text())
    for case in config['cases']:
        budget = full_budget(case)
        assert budget['full_generation_windows'] == 3662
        assert budget['evaluator_repeats_per_metric'] == 10
        if case['missing_rate'] == 0:
            assert budget['embedding_optimizer_updates'] == 17000
            assert budget['generator_optimizer_updates'] == 12750
            assert budget['full_periodic_evaluations'] == 17
        else:
            assert budget['embedding_optimizer_updates'] == 14000
            assert budget['generator_optimizer_updates'] == 4000
            assert budget['full_periodic_evaluations'] == 4
        bad = dict(case, parsed_original_arguments=dict(case['parsed_original_arguments'], max_steps=2))
        with pytest.raises(ValueError, match='budget'):
            full_budget(bad)
