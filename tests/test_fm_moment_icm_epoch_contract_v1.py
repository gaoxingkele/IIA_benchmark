import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching import moment_icm_epoch_contract_v1 as s

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/mtsad_protocol_audit/sources/moment_research/original/moment/tasks/pretrain.py'
HASH = sha256(SOURCE.read_bytes()).hexdigest()


@pytest.mark.parametrize('batches,steps,epochs', [(7, 20, 1), (7, 21, 1), (7, 22, 1), (3, 2, 4), (1, 20, 1)])
def test_original_source_counter_projection_proves_OR_policy_and_no_mid_epoch_stop(batches, steps, epochs):
    task = SimpleNamespace(args=SimpleNamespace(max_opt_steps=steps, max_epoch=epochs))
    result = s.counter_projection(SOURCE, HASH, 'source_two_minima_OR')(task, batches)
    predicted = s.budget_outcome(batches, steps, epochs, 'source_two_minima_OR')
    assert result == (predicted['epochs'], predicted['optimizer_steps'])
    assert result[1] >= steps and result[0] >= epochs


@pytest.mark.parametrize('batches,step_minimum', [(7, 5000000), (2048, 5000000), (1, 2)])
def test_private_paper_loop_completes_exactly_one_full_epoch_despite_source_step_minimum(batches, step_minimum):
    task = SimpleNamespace(args=SimpleNamespace(max_opt_steps=step_minimum, max_epoch=1))
    assert s.counter_projection(SOURCE, HASH, 'paper_complete_epochs')(task, batches) == (1, batches)
    assert s.budget_outcome(batches, step_minimum, 1, 'paper_complete_epochs')['optimizer_steps'] == batches


def test_numerical_train_body_is_unchanged_except_the_explicit_stop_condition():
    original = s.train_node(SOURCE, HASH)
    private = s.paper_train_node(SOURCE, HASH)
    old = next(n for n in ast.walk(original) if isinstance(n, ast.While))
    new = next(n for n in ast.walk(private) if isinstance(n, ast.While))
    assert ast.dump(ast.Module(body=old.body, type_ignores=[])) == ast.dump(ast.Module(body=new.body, type_ignores=[]))
    assert ast.unparse(old.test) == s.SOURCE_CONDITION
    assert ast.unparse(new.test) == s.PAPER_CONDITION
    assert callable(s.build_paper_train(SOURCE, HASH, {}))
    assert sha256(SOURCE.read_bytes()).hexdigest() == HASH


def test_refuse_another_source_version():
    with pytest.raises(ValueError, match='SHA256'):
        s.paper_train_node(SOURCE, '0' * 64)


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_refuse_empty_invalid_or_fractional_epoch_budget(value):
    with pytest.raises(ValueError):
        s.budget_outcome(value, 100, 1, 'paper_complete_epochs')


def test_bind_refuses_a_truncated_final_batch_before_compiling_any_model_code():
    class Loader:
        drop_last = True
        def __len__(self): return 3
    task = SimpleNamespace(args=SimpleNamespace(max_epoch=1), train_dataloader=Loader())
    with pytest.raises(ValueError, match='partial batch'):
        s.bind_one_full_epoch(task, {'paper_epochs': 1}, ROOT)


def test_bind_refuses_disagreement_between_scheduler_and_paper_epoch_contract():
    task = SimpleNamespace(args=SimpleNamespace(max_epoch=50))
    with pytest.raises(ValueError, match='scheduler'):
        s.bind_one_full_epoch(task, {'paper_epochs': 1}, ROOT)


def test_bind_returns_isolated_method_and_keeps_original_task_and_source_unchanged():
    namespace = {}
    exec(compile(ast.Module(body=[s.train_node(SOURCE, HASH)], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    class Loader:
        drop_last = False
        def __len__(self): return 7
    class Task:
        train = namespace['train']
    task = Task()
    task.args = SimpleNamespace(max_epoch=1)
    task.train_dataloader = Loader()
    original_function = task.train.__func__
    method = s.bind_one_full_epoch(task, dict(paper_epochs=1, source_path=SOURCE.relative_to(ROOT).as_posix(), source_sha256=HASH), ROOT)
    assert method.__self__ is task
    assert method.__func__ is not original_function
    assert task.train.__func__ is original_function
    assert method.__func__.__code__.co_filename.startswith('<isolated-paper-one-full-epoch:')
    assert sha256(SOURCE.read_bytes()).hexdigest() == HASH


def test_source_defaults_would_exceed_one_epoch_unless_an_entire_epoch_has_at_least_five_million_batches():
    outcome = s.budget_outcome(2048, 5000000, 1, 'source_two_minima_OR')
    assert outcome['epochs'] == 2442 and outcome['optimizer_steps'] == 5001216
    assert not outcome['actual_neural_training']
