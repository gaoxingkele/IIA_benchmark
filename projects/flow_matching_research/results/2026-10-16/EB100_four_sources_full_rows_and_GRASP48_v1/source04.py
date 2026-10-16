from pathlib import Path
import pytest
from scripts.flow_matching import reserve_CITE_full_preprocessing_window_v1 as s


class Handle:
    def __init__(self):self.closed=False
    def close(self):self.closed=True


def test_busy_second_lock_rolls_back_first_without_holding_and_waiting():
    first=Handle();calls=[]
    def lock(path):
        calls.append(str(path));return first if len(calls)==1 else None
    assert s.try_reserve(lock,[Path('b'),Path('a')]) is None
    assert calls==['a','b'] and first.closed


def test_exception_second_lock_releases_all_prior_metadata_handles():
    first=Handle();calls=[]
    def lock(path):
        calls.append(path)
        if len(calls)==2:raise RuntimeError('fixture')
        return first
    with pytest.raises(RuntimeError):s.try_reserve(lock,[Path('a'),Path('b')])
    assert first.closed


def test_success_owns_only_both_lock_handles_until_caller_releases():
    handles=[Handle(),Handle()];returned=s.try_reserve(lambda path:handles[0] if str(path)=='a' else handles[1],[Path('b'),Path('a')])
    assert returned==handles and not any(h.closed for h in handles)
    for h in reversed(returned):h.close()
    assert all(h.closed for h in handles)


@pytest.mark.parametrize('args,expected',[
    ((True,True,True,True,True,10,900,120),'release_actual_preparation_terminal_guard'),
    ((False,False,True,True,True,10,900,120),'release_missing_exact_preparation_owner'),
    ((False,True,True,False,False,10,900,120),'release_emergency_headroom_metadata_only'),
    ((False,True,True,False,True,900,900,120),'release_window_deadline_numeric_worker_continues'),
    ((False,True,False,False,True,120,900,120),'release_no_original_headroom_after_grace'),
    ((False,True,False,False,True,10,900,120),'hold_original_family_locks_for_full_preparation'),
    ((False,True,True,False,True,200,900,120),'hold_original_family_locks_for_full_preparation')])
def test_deadlines_and_emergencies_only_release_metadata_window(args,expected):
    assert s.window_action(*args)==expected


def test_all_original_guards_and_own_data_lock_are_preserved():
    c=s.registration();prep=s.read(s.ROOT/s.preparation.CONFIG)
    assert len(c['family_locks'])==2 and prep['resource_lock'] not in c['family_locks']
    assert c['original_preparation_settings']==prep['settings']
    assert c['original_preparation_stable_samples']==prep['stable_samples']==6
    assert c['maximum_owned_window_seconds']==900 and c['maximum_lock_acquisition_seconds']==900
