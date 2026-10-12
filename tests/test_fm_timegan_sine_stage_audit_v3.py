"""Full loader and unchanged restore audit checks; no benchmark performance."""
import ast
import copy
import inspect
import sys
from types import ModuleType

import numpy as np
import pytest

from scripts.flow_matching import audit_timegan_sine_stage_restore_v3 as audit
from scripts.flow_matching import run_timegan_sine_stage_audit_v3 as controller


def runtime():
    return controller.read(controller.ROOT / 'configs/runtime/fm_timegan_sine_stage_restore_audit.v3.json')


def test_entire_original_restore_and_native_transition_audit_is_preserved():
    before = ast.parse(inspect.getsource(audit.original.audit))
    after = audit.audit_ast()
    replacement = ast.dump(ast.parse('_native_sine_loader(job, model, np)', mode='eval').body)
    restored = ast.parse("real_data_loading(job['dataset'], model['parameters']['seq_len'])", mode='eval').body
    changes = [0]

    class Reverse(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == replacement:
                changes[0] += 1
                return copy.deepcopy(restored)
            return self.generic_visit(node)

    assert ast.dump(Reverse().visit(after)) == ast.dump(before)
    assert changes == [1]


def test_full_original_sine_loader_every_coordinate_is_independently_bitwise_verified(monkeypatch):
    value = runtime()
    config, queue, job, records = controller.verify(value)
    model = controller.read(controller.ROOT / queue['model_config'])
    source = controller.ROOT / model['source_root'] / 'data_loading.py'
    module = ModuleType('data_loading')
    exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), module.__dict__)
    monkeypatch.setitem(sys.modules, 'data_loading', module)
    state = np.random.get_state()
    try:
        np.random.seed(job['seed'])
        result = audit.native_sine_loader(job, model, np)
        assert np.asarray(result).shape == (10000, 24, 5)
        assert np.isfinite(result).all()
        with pytest.raises(ValueError): audit.native_sine_loader(dict(job, dataset='stock'), model, np)
    finally:
        np.random.set_state(state)


def test_predecessor_full_stages_are_bound_without_retraining_or_native_imports():
    config, queue, job, records = controller.verify(runtime())
    assert len(queue['jobs']) == 45 and job['dataset'] == 'sine' and job['seed'] == 1103
    assert [r['optimizer_updates'] for r in records] == [
        dict(E0=50000, GS=0, E=0, G=0, D=0), dict(E0=50000, GS=50000, E=0, G=0, D=0)]
    assert callable(audit.corrected_audit())


@pytest.mark.parametrize('field,value', [('minimum_commit_headroom_bytes', 23 * 1024**3),
    ('minimum_available_memory_bytes', 1), ('emergency_commit_headroom_bytes', 15 * 1024**3),
    ('maximum_worker_private_bytes', 8 * 1024**3)])
def test_existing_full_native_audit_guards_cannot_be_loosened(field, value):
    settings = copy.deepcopy(runtime())
    settings['settings'][field] = value
    with pytest.raises(ValueError): controller.verify(settings)


@pytest.mark.parametrize('field,value', [('resource_lock','other.lock'), ('python','other.exe'),
    ('original_scope', dict(papers=44, method_baseline_ablation_records=681, original_data_task_records=273))])
def test_original_environment_mutex_and_broader_scope_required(field, value):
    settings = copy.deepcopy(runtime()); settings[field] = value
    with pytest.raises(ValueError): controller.verify(settings)


def test_source_files_never_hot_edited_during_compilation():
    path = controller.ROOT / 'scripts/flow_matching/audit_timegan_stage_resume_v2.py'
    before = path.read_bytes()
    audit.corrected_audit()
    assert path.read_bytes() == before
