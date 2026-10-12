"""Fresh monitor binding preserves original client guards and full native registry."""
import copy

import pytest

from scripts.flow_matching import run_cached_resource_monitor_controller_v1 as monitor


def config():
    return monitor.read(monitor.ROOT/'configs/runtime/fm_rcgan_cached_resource_monitor.v1.json')


def test_full_original_seed_registry_settings_and_source_pins_are_preserved():
    queue=monitor.verify(config())
    assert len(queue['jobs'])==5 and queue['epochs']==500
    assert {j['seed'] for j in queue['jobs']}==set(range(1103,1108))
    assert queue['settings']==config()['settings']


@pytest.mark.parametrize('mutation',['guard','mutex','scope','source','module'])
def test_new_monitor_cannot_weaken_original_caps_or_change_native_scope(mutation):
    value=copy.deepcopy(config())
    if mutation=='guard': value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='mutex': value['resource_lock']='other.lock'
    elif mutation=='scope': value['original_scope']['papers']=44
    elif mutation=='source': value['source_receipts'][0]['sha256']='0'*64
    else: value['entry_module']='scripts.flow_matching.other'
    with pytest.raises(ValueError): monitor.verify(value)


def test_cached_snapshot_is_used_by_the_fresh_compiled_client_and_all_other_guard_code_is_identical(monkeypatch):
    from scripts.flow_matching import run_light_controller as loader
    from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as client
    old_loader=loader.load_functions;old_client=client.load_functions
    monkeypatch.setattr(loader,'load_functions',old_loader)
    monkeypatch.setattr(client,'load_functions',old_client)
    before=client.resource_api()
    calls=[]
    monitor.install_cached_loader(lambda value:calls.append(value))
    after=client.resource_api()
    assert before['resource_snapshot'] is not after['resource_snapshot']
    for name in ('has_headroom','guarded_wait','terminate_tree','resource_slot','exclusive_lock'):
        assert before[name].__code__.co_code==after[name].__code__.co_code
        assert before[name].__code__.co_consts==after[name].__code__.co_consts
    result=after['resource_snapshot']()
    assert calls==[result]
