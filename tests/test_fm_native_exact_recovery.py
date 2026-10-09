import json
import dis
from types import CodeType
from pathlib import Path
import pytest

from scripts.flow_matching.native_exact_recovery import components, controller_main, verify, worker_main
from scripts.flow_matching.grasp_protocol_runner import sha


def instructions(code):
    # Module symbol-table indices differ when an unchanged function is compiled alone.
    return [(item.opname, instructions(item.argval) if isinstance(item.argval, CodeType)
             else item.argval) for item in dis.get_instructions(code)]


@pytest.mark.parametrize('track', ['statistical', 'grasp'])
def test_native_worker_recovery_preserves_original_train_evaluate_code(track):
    original, _ = components(track)
    bound = worker_main(track)
    assert bound.__globals__['verify'] is verify
    assert bound.__globals__['complete'] is original.complete
    assert instructions(bound.__globals__['run'].__code__) == instructions(original.run.__code__)


@pytest.mark.parametrize('track', ['statistical', 'grasp'])
def test_native_controller_keeps_existing_resource_guards(track):
    _, original = components(track)
    bound = controller_main(track)
    assert bound.__globals__['verify'] is verify
    assert bound.__globals__['complete'] is original.complete
    if track == 'grasp':
        assert bound.__globals__['guarded_gpu_wait'] is original.guarded_gpu_wait
    assert 'scripts.flow_matching.run_native_exact_recovery_job' in bound.__code__.co_consts


def test_semantics_change_is_rejected(tmp_path, monkeypatch):
    import scripts.flow_matching.native_exact_recovery as module
    class Worker:
        @staticmethod
        def verify(queue, root):
            return queue
    monkeypatch.setattr(module, 'components', lambda track: (Worker, None))
    original = {'jobs': [{'id': 'old', 'output_directory': 'oldpath', 'seed': 1, 'epochs': 1500}],
                'source_receipts': [], 'settings': {'cpu_threads': 1}}
    (tmp_path / 'original.json').write_text(json.dumps(original))
    (tmp_path / 'failure.json').write_text('{"exit_code":1}')
    recovery = dict(original, jobs=[{'id': 'retry', 'output_directory': 'newpath', 'seed': 1, 'epochs': 1500,
                                   'recovery_original_job_id': 'old'}],
        recovery_track='grasp', recovery_original_queue='original.json',
        recovery_original_queue_sha256=sha(tmp_path / 'original.json'), recovery_original_job_ids=['old'],
        recovery_failure_receipts=[{'original_job_id': 'old', 'path': 'failure.json', 'sha256': sha(tmp_path / 'failure.json')}])
    verify(recovery, tmp_path)
    recovery['jobs'][0]['epochs'] = 1
    with pytest.raises(ValueError, match='experiment semantics changed'):
        verify(recovery, tmp_path)
