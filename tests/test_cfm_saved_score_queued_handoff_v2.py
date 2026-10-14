"""Metadata admission is not permission to weaken full numerical admission."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.flow_matching import run_cfm_saved_score_queued_handoff_v2 as handoff


def config():
    return handoff.previous.cold.original.read(handoff.ROOT/'configs/runtime/fm_cfm_saved_score_queued_handoff.v2.json')


def test_original_full_model_score_and_all_guards_unchanged():
    registered = config()
    primary = handoff.verify(registered)
    assert len(primary['routes']) == 285
    assert registered['settings'] == primary['settings']
    assert registered['job']['epochs'] == 400 and registered['job']['expected_optimizer_updates'] == 62400
    assert registered['job']['test_trajectories'] == 20 and registered['job']['test_points_per_trajectory'] == 500
    for key in ['minimum_available_memory_bytes','emergency_commit_headroom_bytes','maximum_worker_rss_bytes','maximum_worker_private_bytes']:
        assert registered['settings'][key] == primary['settings'][key]


def test_release_before_wait_and_no_scoring_at_metadata_admission():
    registered = config()
    log = []
    snapshots = iter([dict(commit_available_bytes=27*1024**3),dict(commit_available_bytes=31*1024**3)])
    @contextmanager
    def acquire(path, settings, api, update):
        assert settings == registered['family_coordination_settings']
        log.append('lease_granted')
        try: yield {}
        finally: log.append('lease_released')
    api = dict(resource_snapshot=lambda:next(snapshots),
        has_headroom=lambda memory,settings:memory['commit_available_bytes']>=settings['minimum_commit_headroom_bytes'])
    def sleep(seconds):
        assert log[-1] == 'lease_released'
        log.append('unowned_wait')
    with handoff.coordinated_slot(handoff.ROOT/registered['family_lock'],registered,registered['settings'],api,
            acquire_slot=acquire,sleep=sleep,finish=lambda *args:log.append('full_reviews_before_release')) as admitted:
        assert admitted['commit_available_bytes'] == 31*1024**3
        assert log == ['lease_granted','lease_released','unowned_wait','lease_granted']
        log.append('full_scoring_admitted')
    assert log[-1] == 'lease_released'
    assert log[-2] == 'full_reviews_before_release'


def test_exception_releases_only_lease():
    registered = config(); log=[]
    @contextmanager
    def acquire(*args):
        try: yield {}
        finally: log.append('released')
    api=dict(resource_snapshot=lambda:{},has_headroom=lambda *a:True)
    with pytest.raises(RuntimeError):
        with handoff.coordinated_slot(handoff.ROOT/registered['family_lock'],registered,registered['settings'],api,
                acquire_slot=acquire):
            raise RuntimeError('worker failure must preserve scientific evidence')
    assert log==['released']


def test_only_idle_metadata_owner_without_outputs_or_numeric_children_can_handoff():
    state=dict(state='waiting_for_memory',active_pid=None,shared_lock_owned=False)
    assert handoff.idle_allowed('original',state,False,[])
    assert not handoff.idle_allowed('original',dict(state,active_pid=123),False,[])
    assert not handoff.idle_allowed('original',dict(state,shared_lock_owned=True),False,[])
    assert not handoff.idle_allowed('original',state,True,[])
    assert not handoff.idle_allowed('original',state,False,[dict(name='python.exe',exe='python.exe')])
