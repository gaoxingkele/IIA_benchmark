from types import SimpleNamespace

from scripts.flow_matching import native_exact_recovery_manifest_v2 as repaired
from scripts.flow_matching import native_exact_recovery as original


def test_worker_receives_manifest_instead_of_registration(monkeypatch):
    registration = {'manifest': 'data.json'}
    manifest = {'datasets': [{'id': 'swan'}]}
    monkeypatch.setattr(original, 'verify', lambda queue, root: registration)
    worker = SimpleNamespace(verify=lambda queue, root: manifest if queue is registration else None)
    monkeypatch.setattr(original, 'components', lambda track: (worker, None))
    assert repaired.verified_manifest({'recovery_track': 'grasp'}) is manifest


def test_repair_preserves_entire_frozen_train_evaluate_function():
    from tests.test_fm_native_exact_recovery import instructions
    worker, _ = original.components('grasp')
    bound = repaired.worker_main()
    assert bound.__globals__['verify'] is repaired.verified_manifest
    assert instructions(bound.__globals__['run'].__code__) == instructions(worker.run.__code__)
    assert bound.__globals__['complete'] is worker.complete


def test_controller_changes_only_worker_dispatch_and_verifier():
    first = original.controller_main('grasp')
    second = repaired.controller_main()
    assert first.__code__.co_code == second.__code__.co_code
    pairs = [(a, b) for a, b in zip(first.__code__.co_consts, second.__code__.co_consts) if a != b]
    assert pairs == [('scripts.flow_matching.run_native_exact_recovery_job',
                      'scripts.flow_matching.run_native_grasp_recovery_v2')]
    assert second.__globals__['guarded_gpu_wait'] is first.__globals__['guarded_gpu_wait']
