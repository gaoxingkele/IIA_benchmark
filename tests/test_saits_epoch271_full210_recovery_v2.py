import copy
import ast
import inspect
from unittest.mock import patch
import pytest
from scripts.flow_matching import run_saits_epoch271_full210_recovery_v2 as current

CONFIG = 'configs/runtime/fm_saits_saved_epoch271_full210_recovery.v2.json'


def test_full210_original_jobs_and_native_configs_are_unchanged():
    runtime = current.original.read(current.ROOT / CONFIG)
    queue, proof, incident = current.verify(runtime)
    previous = current.original.read(current.ROOT / current.PREVIOUS)
    assert queue == current.original.read(current.ROOT / previous['queue'])
    assert len(queue['jobs']) == 210
    assert proof['full_saved_epoch_boundary'] == 271 and proof['saved_optimizer_updates'] == 1897
    assert not proof['full_training_or_test_complete']
    assert incident['resource_guard_aborted']


@pytest.mark.parametrize('field,value', [('minimum_commit_headroom_bytes', 27*2**30),
    ('emergency_commit_headroom_bytes', 8*2**30), ('minimum_gpu_free_bytes', 8*2**30)])
def test_no_original_resource_guard_or_stronger_admission_can_be_reduced(field, value):
    runtime = copy.deepcopy(current.original.read(current.ROOT / CONFIG))
    runtime['settings'][field] = value
    with pytest.raises(ValueError):
        current.verify(runtime)


def test_live_exact_previous_worker_prevents_duplicate_resume():
    runtime = current.original.read(current.ROOT / CONFIG)
    incident = current.original.read(current.ROOT / runtime['preserved_incident'])
    old = incident['previous_owner']
    with patch('psutil.Process') as process:
        process.return_value.create_time.return_value = old['create_time']
        process.return_value.cmdline.return_value = old['command']
        with pytest.raises(RuntimeError, match='still exists'):
            current.predecessor_absent(incident)


def test_changed_saved_checkpoint_is_not_reused():
    runtime = current.original.read(current.ROOT / CONFIG)
    proof = current.original.read(current.ROOT / runtime['saved_epoch_proof'])
    with patch.object(current.original, 'digest', return_value='different'):
        with pytest.raises(ValueError, match='changed before'):
            current.saved_epoch_still_current(proof)


def test_remaining_healthy_jobs_keep_original_admission_and_only_failed_job_is_stronger():
    runtime = current.original.read(current.ROOT / CONFIG)
    previous = current.original.read(current.ROOT / current.PREVIOUS)
    assert current.job_admission(runtime, current.SELECTED)['minimum_commit_headroom_bytes'] == 40*2**30
    assert current.job_admission(runtime, 'another_original_job') == previous['settings']


def test_full_original_controller_identical_except_selected_admission_argument():
    tree = current.controller_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and getattr(getattr(node.iter,'func',None),'id',None) == '_scheduler_jobs':
            node.iter = ast.parse("queue['jobs']",mode='eval').body
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_job_admission':
            node.func = ast.parse("runtime['settings']", mode='eval').body
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func, ast.Subscript) and not node.keywords and len(node.args)==2:
                if ast.dump(node.func) == ast.dump(ast.parse("runtime['settings']",mode='eval').body):
                    return ast.parse("runtime['settings']",mode='eval').body
            return self.generic_visit(node)
    tree = Restore().visit(tree)
    assert ast.dump(tree) == ast.dump(ast.parse(inspect.getsource(current.original.main)))


def test_full210_stable_order_keeps_every_job_and_does_not_block_healthy_jobs_behind40GiB():
    runtime = current.original.read(current.ROOT / CONFIG)
    queue = current.original.read(current.ROOT / runtime['queue'])
    actual = current.scheduler_jobs(queue,runtime)
    healthy = [j for j in queue['jobs'] if current.original.read(current.ROOT/j['config'])['id'] != current.SELECTED]
    assert len(actual) == 210 and actual[:-1] == healthy
    assert current.original.read(current.ROOT/actual[-1]['config'])['id'] == current.SELECTED
