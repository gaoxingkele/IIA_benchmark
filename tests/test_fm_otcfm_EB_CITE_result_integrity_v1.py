from copy import deepcopy
import pytest
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import full_metric_names,validate_record
from scripts.flow_matching.run_otcfm_EB_CITE_native150_v2 import identity
from scripts.flow_matching.run_otcfm_singlecell_native_v4 import contract


def fixture():
    job=dict(id='fixture_not_performance',dataset='released_author_EB',seed=42,leaveout_timepoint=1,
        scientific_protocol=contract.protocol('released_yaml'),data=dict(days={'0':2656,'1':4374,'2':3650,'3':3873,'4':3650}))
    result=dict(status='completed',job=deepcopy(job),job_sha256=identity(job),maximum_epochs=1000,batch_size=512,
        parameter_count=9093,full_timepoint_cell_counts=job['data']['days'],original_loader_whiten_true=True,
        epoch_history=[dict(epoch=i,optimizer_updates=4*i) for i in range(1,1001)],actual_completed_epochs=1000,
        optimizer_updates=4000,posttest_selected_checkpoint_optimizer_updates=1360,
        early_stopping_epoch=0,training_seconds=200.,termination_reason='maximum_epochs',early_stopping_patience_checks=100,
        best_checkpoint='F:/fixture/epoch_0339.ckpt',validation_history=[dict(epoch=340,finite=True,validation_loss=.1),dict(epoch=1000,finite=True,validation_loss=.2)],
        test_metrics={name:1. for name in full_metric_names(5)},primary_heldout_W1=1.,
        historical_paper_equivalence_certified=False,strict_TAB_result=False)
    return dict(primary_metric='test/t_out/1-Wasserstein'),job,result,dict(passed=True,exit_code=0,resource_guard_aborted=False)


def test_every_reported_scalar_is_present_for_both_complete_data_geometries():
    assert len(full_metric_names(4))==36 and len(full_metric_names(5))==44
    assert 'test/t3/1-Wasserstein' in full_metric_names(5)
    validate_record(*fixture())


@pytest.mark.parametrize('fault',['total_counter','selected_counter','missing_last_timepoint','extra_metric','wrong_heldout','lost_epoch'])
def test_incomplete_or_misattributed_EB_metrics_and_budgets_are_rejected(fault):
    q,j,r,g=fixture()
    if fault=='total_counter':r['optimizer_updates']=1360
    elif fault=='selected_counter':r['posttest_selected_checkpoint_optimizer_updates']=4000
    elif fault=='missing_last_timepoint':r['test_metrics'].pop('test/t3/1-Wasserstein')
    elif fault=='extra_metric':r['test_metrics']['fixture/not_native']=1.
    elif fault=='wrong_heldout':r['test_metrics']['test/t1/1-Wasserstein']=2.
    else:r['epoch_history'].pop()
    with pytest.raises(ValueError):validate_record(q,j,r,g)
