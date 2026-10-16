from copy import deepcopy
import subprocess
import sys
import pytest
from scripts.flow_matching import run_cfm_node1106_epoch280_resume_v1 as s


def fixture():
    terminal=dict(exit_code=1,complete=False,full_original400_checkpointed_result_verified=False,
        resource_observations=dict(resource_guard_aborted=True))
    progress=dict(epoch=284,optimizer_updates=284,required_epochs=400)
    ledger=[dict(completed_original_epochs=e,optimizer_updates=e,required_original_epochs=400,
        complete_original_epoch_boundary=True,all_native_model_optimizer_rng_gradients_history_and_order_values_reread_bitwise=True,
        no_training_state_or_random_state_changed_by_save=True,additional_optimizer_updates=0) for e in range(10,281,10)]
    return terminal,progress,ledger


def test_retains_only_actual_complete280_boundary_not_partial284():
    assert s.check_boundary(*fixture())['completed_original_epochs']==280


@pytest.mark.parametrize('fault',['successful','not_resource','missing_epoch','partial_boundary','mutated_state','extra_update'])
def test_refuses_invalid_or_missing_native_boundary(fault):
    t,p,rows=fixture()
    if fault=='successful':t['complete']=True
    if fault=='not_resource':t['resource_observations']['resource_guard_aborted']=False
    if fault=='missing_epoch':rows.pop(4)
    if fault=='partial_boundary':rows[-1]['completed_original_epochs']=284
    if fault=='mutated_state':rows[-1]['no_training_state_or_random_state_changed_by_save']=False
    if fault=='extra_update':rows[-1]['additional_optimizer_updates']=1
    with pytest.raises(ValueError):s.check_boundary(t,p,rows)


def test_cold_registration_import_does_not_load_numerical_packages():
    code='from scripts.flow_matching import run_cfm_node1106_epoch280_resume_v1;import sys;assert not any(m in sys.modules for m in ["numpy","scipy","torch"])'
    subprocess.run([sys.executable,'-B','-X','utf8','-c',code],check=True)


def test_actual_registration_preserves_all_other284_jobs_and_numerical_policy():
    c,q=s.registration();old=s.read(s.ROOT/s.PREVIOUS);previous=s.read(s.ROOT/old['queue'])
    assert len(q['jobs'])==285 and c['remaining_original_epochs']==120
    assert c['resource_policy']==old['resource_policy'] and q['resource_policy']==previous['resource_policy']
    assert c['resume_full_epoch_checkpoint_sha256']==s.sha(c['resume_full_epoch_checkpoint'])
    before=[j for j in previous['jobs'] if j['id']!=old['job_id']]
    after=[j for j in q['jobs'] if j['id']!=c['job_id']]
    assert before==after and len(after)==284
    assert c['maximum_new_numerical_attempts']==1 and c['discarded_unsaved_epochs']==[281,282,283,284]
