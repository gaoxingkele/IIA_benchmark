import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import verify_checkpoint_metadata, compare_full_predictions, capped_wait


def fixture():
    job = dict(id='original_seed1103', seed=1103, epochs=400, expected_optimizer_updates=400)
    digest = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    losses = [1.]*400
    checkpoint = dict(job=job, job_sha256=digest, optimizer_updates=400, training_losses=losses.copy())
    result = dict(job=job, job_sha256=digest, optimizer_updates=400, epochs_completed=400, training_losses=losses.copy())
    return job, checkpoint, result


def test_saved_full_job_seed_budget_and_history_are_required():
    job, checkpoint, result = fixture()
    verify_checkpoint_metadata(checkpoint, result, job)
    for name in ['seed', 'budget', 'history']:
        bad = copy.deepcopy(checkpoint)
        if name=='seed': bad['job']['seed'] += 1
        if name=='budget': bad['optimizer_updates'] -= 1
        if name=='history': bad['training_losses'][-1] += .1
        with pytest.raises(ValueError):
            verify_checkpoint_metadata(bad, result, job)


def test_every_full_prediction_must_match_under_frozen_tolerance():
    saved = np.arange(1000, dtype=float).reshape(2, 500, 1)
    tolerance = dict(rtol=1e-7, atol=1e-8)
    assert compare_full_predictions(saved.copy(), saved, tolerance)['bitwise_identical']
    for changed in [saved[:, :-1], saved + .01, np.full_like(saved, np.nan)]:
        with pytest.raises(ValueError):
            compare_full_predictions(changed, saved, tolerance)


@pytest.mark.parametrize('limit', ['rss', 'private', 'commit'])
def test_read_only_replay_caps_terminate_only_its_own_worker(monkeypatch, limit):
    from types import SimpleNamespace
    from scripts.flow_matching import audit_cfm_checkpoint_replay_v1 as module
    ended = []
    child = SimpleNamespace(pid=12345, poll=lambda: 137 if ended else None, wait=lambda: 137)
    info = SimpleNamespace(rss=3 if limit=='rss' else 1, private=5 if limit=='private' else 1, vms=1)
    process = SimpleNamespace(memory_info=lambda: info, children=lambda recursive: [])
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: process)
    api = dict(resource_snapshot=lambda: dict(commit_available_bytes=15 if limit=='commit' else 50),
        terminate_tree=lambda worker: ended.append(worker.pid))
    config = dict(settings=dict(emergency_commit_headroom_bytes=16, maximum_worker_rss_bytes=2,
        maximum_worker_private_bytes=4))
    code, usage = capped_wait(child, config, api, lambda values: None)
    assert code==137 and usage['resource_guard_aborted'] and ended==[12345]
