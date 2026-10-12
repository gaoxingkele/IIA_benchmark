"""Preserve actual full replay body; completed-stage audit never certifies completion."""
import ast
import copy
import inspect

import pytest

from scripts.flow_matching import audit_timegan_saved_predecessor_native_v1 as audit


def config():
    return audit.original.read(audit.original.ROOT / 'configs/runtime/fm_timegan_saved_predecessor_native_audit.v1.json')


def test_every_original_generator_restore_and_full_inference_statement_is_identical():
    old = ast.parse(inspect.getsource(audit.original.worker)).body[0]
    new = audit.worker_ast().body[0]

    def span(function):
        start = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'environment' for t in n.targets))
        end = next(i for i,n in enumerate(function.body) if isinstance(n, ast.ClassDef) and n.name == 'ReplayedSession')
        return [ast.dump(n) for n in function.body[start:end]]

    assert span(old) == span(new)


def test_entire_complete_trained_evaluator_session_and_source_loop_are_preserved():
    old = ast.parse(inspect.getsource(audit.original.worker))
    new = audit.worker_ast()
    for kind, name in [(ast.ClassDef,'ReplayedSession'), (ast.FunctionDef,'reset')]:
        before = next(n for n in ast.walk(old) if isinstance(n, kind) and n.name == name)
        after = next(n for n in ast.walk(new) if isinstance(n, kind) and n.name == name)
        assert ast.dump(before) == ast.dump(after)
    before = next(n for n in old.body[0].body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
    after = next(n for n in new.body[0].body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
    assert ast.dump(before) == ast.dump(after)


def test_actual_saved_checkpoint_and_evaluator_ledger_has_no_unsaved_predictors():
    queue, model, job, checkpoints, evaluators = audit.verify(config())
    assert len(queue['jobs']) == 45 and job['seed'] == 1104
    assert len(checkpoints) == 13 and len(evaluators) == 10
    assert all(r['kind'] == 'discriminative' and r['optimizer_updates'] == 2000 for r in evaluators)
    assert callable(audit.compiled_worker())


@pytest.mark.parametrize('mutation', ['guard','scope','mutex','python'])
def test_existing_readonly_audit_guards_and_full_scope_cannot_change(mutation):
    value = copy.deepcopy(config())
    if mutation == 'guard': value['settings']['minimum_commit_headroom_bytes'] -= 1
    elif mutation == 'scope': value['original_scope']['papers'] = 44
    elif mutation == 'mutex': value['resource_lock'] = 'other.lock'
    else: value['python'] = 'other.exe'
    with pytest.raises(ValueError): audit.verify(value)


def test_absent_predictors_never_aggregated_or_reported_as_experiment_completion():
    worker = audit.worker_ast()
    assignment = next(n for n in worker.body[0].body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Tuple) and isinstance(n.targets[0].elts[0], ast.Name)
        and n.targets[0].elts[0].id == 'evaluator_rows')
    assert [k.arg for k in assignment.value.elts[1].keywords] == ['discriminative']
    assert worker.body[0].body[-1].value.func.id == '_finish'
    finish = ast.parse(inspect.getsource(audit.finish))
    report = next(n for n in ast.walk(finish) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == 'dict' and any(k.arg == 'actual_saved_checkpoint_restores' for k in n.keywords))
    values = {k.arg:ast.literal_eval(k.value) for k in report.keywords if isinstance(k.value, ast.Constant)}
    assert values['actual_saved_checkpoint_restores'] == 13
    assert values['actual_saved_trained_evaluator_replays'] == 10
    assert values['missing_full_predictive_evaluator_trainings'] == 10
    assert values['experiment_complete'] is False and values['training_executed'] is False
    assert values['final_full45_checkpoint_restores_still_required'] == 1035
    assert values['final_full45_trained_evaluator_replays_still_required'] == 900
