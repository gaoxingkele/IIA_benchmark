"""Contract fixtures check rejection only; they are not performance results."""
from copy import deepcopy
import pytest
from scripts.flow_matching import otcfm_EB100_result_integrity_v1 as s
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as native


def fixture():
    q = native.registration(); job = q['jobs'][0]
    result = dict(status='completed',job=deepcopy(job),job_sha256=native.identity(job),
        maximum_epochs=1000,batch_size=512,parameter_count=9093,
        full_timepoint_cell_counts=job['data']['days'],original_loader_whiten_true=True,
        epoch_history=[dict(epoch=i,optimizer_updates=3*i) for i in range(1,1001)],
        actual_completed_epochs=1000,optimizer_updates=3000,
        posttest_selected_checkpoint_optimizer_updates=1020,early_stopping_epoch=0,
        training_seconds=200.,termination_reason='maximum_epochs',early_stopping_patience_checks=100,
        best_checkpoint='F:/fixture/epoch_0339.ckpt',
        validation_history=[dict(epoch=340,finite=True,validation_loss=.1),dict(epoch=1000,finite=True,validation_loss=.2)],
        test_metrics={name:1. for name in s.full_metric_names(5)},primary_heldout_W1=1.,
        historical_paper_equivalence_certified=False,strict_TAB_result=False)
    guard=dict(passed=True,exit_code=0,resource_guard_aborted=False)
    return q,job,result,guard


def test_full_original_EB100_budget_and_all44_scalar_contract():
    q,j,r,g=fixture()
    assert len(r['test_metrics'])==44
    s.validate_record(q,j,r,g)


@pytest.mark.parametrize('fault',['total_counter','selected_counter','missing_leave3_metric',
    'wrong_heldout_metric','lost_epoch','wrong_population','wrong_PC_dimensions','wrong_cells','failed_guard'])
def test_incomplete_original_EB100_results_are_rejected(fault):
    q,j,r,g=fixture()
    if fault=='total_counter':r['optimizer_updates']=1020
    elif fault=='selected_counter':r['posttest_selected_checkpoint_optimizer_updates']=3000
    elif fault=='missing_leave3_metric':r['test_metrics'].pop('test/t3/1-Wasserstein')
    elif fault=='wrong_heldout_metric':r['test_metrics']['test/t1/1-Wasserstein']=2.
    elif fault=='lost_epoch':r['epoch_history'].pop()
    elif fault=='wrong_population':j['data']['days']['0']=2656
    elif fault=='wrong_PC_dimensions':j['data']['full_original_PC_dimensions']=50
    elif fault=='wrong_cells':j['data']['cells']=18203
    else:g['passed']=False
    with pytest.raises(ValueError):s.validate_record(q,j,r,g)


def test_result_from_other150_queue_is_rejected_even_when_record_contract_passes(monkeypatch):
    q,j,r,g=fixture();r['queue_sha256']='different_original150_queue'
    g['result_sha256']='same_result'
    monkeypatch.setattr(s,'read',lambda path:r if path.name=='result.json' else g)
    monkeypatch.setattr(s,'sha',lambda path:'same_result' if path.name=='result.json' else 'expected_original90_queue')
    with pytest.raises(ValueError,match='original EB100 full90'):
        s.inspect(q,j)
