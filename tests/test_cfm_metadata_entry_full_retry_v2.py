import ast
import inspect
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import run_cfm_metadata_safe_original_job_v2 as worker
from scripts.flow_matching import run_cfm_metadata_entry_full_retry_v2 as producer
from scripts.flow_matching import run_cfm_metadata_entry_full_replay_v2 as collector


def test_original_inline_cli_dispatches_selected_job_without_a_main_function(monkeypatch):
    values = worker.previous.namespace()
    assert 'main' not in values
    job = {'id': 'full_original', 'epochs': 400, 'expected_optimizer_updates': 6400}
    queue = {'jobs': [job]}
    calls = []
    values.update(read=lambda path: queue, run=lambda q, j: calls.append((q, j)))
    monkeypatch.setattr(worker.previous, 'namespace', lambda: values)
    monkeypatch.setattr(sys, 'argv', ['worker', '--queue', 'registered.json', '--job-id', job['id']])
    worker.main()
    assert calls == [(queue, job)]


def test_real_module_help_entry_succeeds_in_an_isolated_process():
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m',
                             'scripts.flow_matching.run_cfm_metadata_safe_original_job_v2', '--help'],
                            cwd=producer.ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--queue' in result.stdout and '--job-id' in result.stdout


def test_cli_ast_and_numeric_run_are_the_original_unchanged_bodies():
    values = worker.previous.namespace()
    original = worker.previous.original
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    guard = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                 and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
    assert ast.dump(worker.cli_ast(original)) == ast.dump(ast.Module(body=guard.body, type_ignores=[]))
    assert values['run'].__code__ is original.run.__code__
    assert values['complete'].__code__ is original.complete.__code__
    assert values['write'] is worker.previous.safe_write


def test_complete_retry_registration_preserves_scope_and_original_budget():
    config = producer.cold.original.read(producer.ROOT / 'configs/runtime/fm_cfm_metadata_entry_controller.v2.json')
    runtime, queue, jobs = producer.verify(config)
    assert len(queue['jobs']) == 285 and len(jobs) == 2
    assert [j['seed'] for j in jobs] == [1104, 1106]
    assert all(j['epochs'] == 400 and j['expected_optimizer_updates'] == 6400 for j in jobs)
    assert all(j['test_trajectories'] == 20 and j['test_points_per_trajectory'] == 500 for j in jobs)
    assert producer.namespace(runtime, queue)['external_complete'].__code__ is producer.previous.original.external_complete.__code__


def test_selected_replay_routes_and_original_numerical_guard_are_retained():
    root = producer.ROOT
    runtime = producer.cold.original.read(root / 'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json')
    primary = producer.cold.original.read(root / collector.previous.PRIMARY)
    queue = producer.cold.original.read(root / 'configs/experiments/fm_cfm_metadata_entry_full_retry_queue.v2.json')
    assert runtime['routes'] == collector.previous.supplement.expected_routes(primary, queue)
    assert len(runtime['routes']) == 285
    for key in ['settings', 'prediction_tolerance', 'original_scope', 'resource_lock', 'environment_lock']:
        assert runtime[key] == primary[key]
    values = collector.namespace()
    assert values['verify_record'].__code__ is collector.previous.cold.original.verify_record.__code__


def test_live_failed_parent_prevents_duplicate_producer_dispatch(monkeypatch):
    import psutil
    config = producer.cold.original.read(producer.ROOT / 'configs/runtime/fm_cfm_metadata_entry_controller.v2.json')
    class LiveOwner:
        def create_time(self): return config['failed_parent']['create_time']
    monkeypatch.setattr(psutil, 'Process', lambda pid: LiveOwner())
    with pytest.raises(ValueError, match='actually be terminal'):
        producer.verify(config)
