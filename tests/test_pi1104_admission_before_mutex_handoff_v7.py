from contextlib import contextmanager
import pytest
from scripts.flow_matching import run_pi1104_admission_before_mutex_handoff_v7 as current


def test_original_settings_are_admitted_before_owned_headroom_and_released_on_exit(tmp_path):
    calls=[]
    @contextmanager
    def ordinary(path,update=None,poll_seconds=10):
        calls.append('ordinary');yield 'lane'
    settings={'minimum_commit_headroom_bytes':40*2**30}
    @contextmanager
    def slot(path,supplied,update,poll):
        assert supplied is settings
        calls.append('admitted_then_owned')
        try:yield {'commit_available_bytes':41*2**30}
        finally:calls.append('released')
    values={'waited_lock':ordinary};lock=tmp_path/'cpu.lock'
    state=current.reserve_before_lock(values,{'settings':settings},lock,{'resource_slot':slot})
    with pytest.raises(RuntimeError,match='owned resource slot'):
        values['wait_for_headroom'](settings)
    with values['waited_lock'](lock):
        assert values['wait_for_headroom'](settings)=={'commit_available_bytes':41*2**30}
        with pytest.raises(RuntimeError):values['wait_for_headroom']({'minimum_commit_headroom_bytes':32*2**30})
    assert calls==['admitted_then_owned','released'] and state['snapshot'] is None


def test_lane_lock_is_unchanged_and_exception_releases_admitted_slot(tmp_path):
    calls=[]
    @contextmanager
    def old(path,waiting=None,poll_seconds=10):calls.append('lane');yield 'original'
    @contextmanager
    def slot(*args):
        try:yield {'commit_available_bytes':50}
        finally:calls.append('release')
    values={'waited_lock':old};lock=tmp_path/'cpu.lock'
    state=current.reserve_before_lock(values,{'settings':{}},lock,{'resource_slot':slot})
    with values['waited_lock'](tmp_path/'worker.lock') as actual:assert actual=='original'
    with pytest.raises(ValueError):
        with values['waited_lock'](lock):raise ValueError('scientific failure must remain visible')
    assert calls==['lane','release'] and state['snapshot'] is None
