from copy import deepcopy
import pytest
from scripts.flow_matching import continue_common105_TAB577_fair_entities_v1 as s

def test_entire_scope_and_original_workers_guards_and_lock_unchanged():
    c=s.registration();original=s.verify(c);ns=s.namespace()
    assert c['required_cases']==len(original['entries'])==577
    assert c['required_label_views']==1726 and c['required_entity_label_views']==77490
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']
    assert c['inter_entity_yield_seconds']==20
    assert c['scientific_artifact_root']==original['artifact_root']
    for name in ['stage','fragment_action','handoff','terminal_child']:
        assert ns[name].__code__ is getattr(s.fair,name).__code__
    api=ns['original'];entry=original['entries'][0]
    assert api.CONFIG==s.matched.CONFIG
    assert api.worker_command(original,entry,'e000')==s.matched.worker_command(original,entry,'e000')

def test_only_exact_idle_parent_without_numerical_descendants_can_be_handed_off():
    ns=s.namespace();h=s.EXPECTED
    state=dict(pid=h['pid'],create_time=h['create_time'],required_cases=577,required_label_views=1726,state='between_original_entities')
    assert ns['boundary_allowed'](state,h,[],True)
    assert not ns['boundary_allowed'](state,h,[],False)
    assert not ns['boundary_allowed'](state,h,[dict(name='python.exe',exe='python.exe')],True)
    for key,value in [('pid',1),('create_time',0),('required_cases',878),('required_label_views',2057)]:
        changed=dict(state,**{key:value});assert not ns['boundary_allowed'](changed,h,[],True)

def test_configuration_rejects_scope_or_budget_changes():
    c=s.registration()
    for key,value in [('required_cases',1),('inter_entity_yield_seconds',0),('maximum_new_attempts_per_fragment',2)]:
        changed=deepcopy(c);changed[key]=value
        with pytest.raises(ValueError,match='All577'):s.verify(changed)
