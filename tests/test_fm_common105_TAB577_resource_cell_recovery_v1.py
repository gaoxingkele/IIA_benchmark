from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import recover_common105_TAB577_resource_cells_v1 as s


def test_full577_1726_scope_same_physical_emergency_caps_and_stronger_stable_admission():
    c=s.registration();old=s.read(s.ROOT/s.matched.CONFIG)
    assert c['entries']==old['entries'] and len(c['entries'])==577
    assert sum(len(e['case']['required_label_policies']) for e in c['entries'])==1726
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    for k,v in old['settings'].items():
        if k!='minimum_commit_headroom_bytes':assert c['settings'][k]==v
    assert c['stable_admission_samples']==6 and c['stable_poll_seconds']==5 and c['inter_entity_yield_seconds']==20
    assert c['resource_lock']==old['resource_lock'] and c['maximum_new_attempts_per_entity']==1


def test_actual_known_gap_has_dead_exact_original_guard_and_unchanged_model_lineage():
    c=s.registration();entry=next(e for e in c['entries'] if e['key']=='e25abf920638483064c8')
    actual=s.resource_abort(c,entry)
    assert actual is not None and not actual['failed_guard_is_algorithm_accuracy_failure']
    assert actual['original_failed_guards'][0]['prefix']=='e029'
    assert actual['original_failed_guards'][0]['abort_snapshot']['commit_available_bytes']<16*2**30
    assert actual['source_lineage']==s.matched.ready_case(c,entry)['lineage']


def test_original_complete_fragment_checker_summary_and_journal_math_are_unchanged():
    ns=s.namespace()
    for name in ['checked_fragment','summary_worker']:
        assert ns[name].__code__.co_code==getattr(s.durable,name).__code__.co_code
    assert ns['controller'].__code__.co_consts!=s.durable.controller.__code__.co_consts
    assert ns['original'].ready_case is s.matched.ready_case


def cell_fixture(tmp_path):
    old=dict(artifact_root=str(tmp_path/'old'));entry=dict(key='case',case=dict(id='fixed_model'))
    bank=Path(old['artifact_root'])/'cases/case/e029_cell_bank';bank.mkdir(parents=True)
    identity=dict(case=entry['case'],entity_prefix='e029',source_lineage={'scores':'frozen'},policy='labels',metric='VUS_PR',
        original_configuration_sha256=s.sha(s.ROOT/s.durable.original.CONFIG))
    outcome=dict(status='finite',value=.3)
    item=dict(identity=identity,identity_sha256=s.durable.digest(identity),outcome=outcome,outcome_sha256=s.durable.digest(outcome))
    path=bank/(s.durable.digest(dict(policy='labels',metric='VUS_PR'))+'.json')
    path.write_text(json.dumps(item),encoding='utf-8')
    return old,entry,path,item


def test_partial_cell_bank_is_byte_copied_without_overwriting_failed_original(tmp_path):
    old,entry,path,_=cell_fixture(tmp_path);before=path.read_bytes()
    copied=s.seed_partial_bank(old,entry,'e029',{'scores':'frozen'},tmp_path/'new')
    assert len(copied)==1 and Path(copied[0]['new_path']).read_bytes()==before==path.read_bytes()
    with pytest.raises(FileExistsError):s.seed_partial_bank(old,entry,'e029',{'scores':'frozen'},tmp_path/'new')


@pytest.mark.parametrize('fault',['lineage','outcome','identity'])
def test_corrupted_partial_cells_or_wrong_model_are_rejected(tmp_path,fault):
    old,entry,path,item=cell_fixture(tmp_path)
    if fault=='lineage':item['identity']['source_lineage']={'scores':'different'}
    if fault=='outcome':item['outcome']['value']=.9
    if fault=='identity':item['identity_sha256']='0'*64
    path.write_text(json.dumps(item),encoding='utf-8')
    with pytest.raises(ValueError):s.seed_partial_bank(old,entry,'e029',{'scores':'frozen'},tmp_path/'new')


def test_new_parent_import_remains_cold():
    code='from scripts.flow_matching import recover_common105_TAB577_resource_cells_v1;import sys;assert not any(x in sys.modules for x in ["torch","numpy","scipy"])'
    subprocess.run([sys.executable,'-B','-X','utf8','-c',code],check=True)
