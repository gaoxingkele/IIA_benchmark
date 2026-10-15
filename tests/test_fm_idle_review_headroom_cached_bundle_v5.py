import copy
import json

import pytest

from scripts.flow_matching import run_idle_review_headroom_cached_bundle_v5 as sut


def registration(root):
    roles, pins = {}, []
    for name, (entry, count) in sut.ENTRIES.items():
        path = root / (name + '.json')
        path.write_text(json.dumps(dict(jobs=[dict(id=str(i)) for i in range(count)])))
        pins.append(dict(path=path.name, sha256=sut.original.digest(path)))
        args = ['--config', name + '.json']
        roles[name] = dict(entry_module=entry, original_arguments=args,
                          previous_owner=dict(command=['python', '-m', entry, *args]),
                          original_queue=path.name, required_original_slots=count)
    return dict(original_scope=copy.deepcopy(sut.original.SCOPE), roles=roles, source_receipts=pins)


def test_full_registries_and_exact_commands_are_preserved(tmp_path):
    config = registration(tmp_path)
    assert [sut.verify(config, name, tmp_path)['required_original_slots']
            for name in sut.ENTRIES] == [140, 70, 285, 10]
    config['roles']['cfm_retry285']['original_arguments'].append('--smaller-budget')
    with pytest.raises(ValueError, match='every original argument'):
        sut.verify(config, 'cfm_retry285', tmp_path)


def test_changed_original_registry_bytes_are_rejected(tmp_path):
    config = registration(tmp_path)
    (tmp_path / 'ls4_physio10.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen handoff source changed'):
        sut.verify(config, 'ls4_physio10', tmp_path)


def test_gpu_idle_phase_does_not_mutate_original_state():
    state = dict(pid=1, create_time=2., state='waiting_full_gpu_resources',
                 active_pid=None, shared_lock_owned=False)
    before = copy.deepcopy(state)
    assert sut.idle_allowed(state, dict(pid=1, create_time=2.), [])
    assert state == before


@pytest.mark.parametrize('updates,children', [
    (dict(active_pid=4), []),
    (dict(shared_lock_owned=True), []),
    (dict(create_time=3.), []),
    ({}, [dict(name='python.exe', exe='C:/Python/python.exe')]),
])
def test_gpu_wait_never_permits_active_child_lock_or_changed_identity(updates, children):
    state = dict(pid=1, create_time=2., state='waiting_full_gpu_resources',
                 active_pid=None, shared_lock_owned=False)
    state.update(updates)
    assert not sut.idle_allowed(state, dict(pid=1, create_time=2.), children)
