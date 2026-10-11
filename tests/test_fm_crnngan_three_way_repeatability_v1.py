import copy
import pytest

from scripts.flow_matching.audit_crnngan_three_way_repeatability_v1 import require_complete_original_diagnostic


def proof():
    return dict(full_complete_epochs_executed=4, original_executor_unchanged=True,
        all_original_data_curriculum_and_optimizer_budgets_identical=True,
        all_complete_graph_definitions_bitwise_identical=True,
        records=[dict(epoch=i, full_training_batches=130,
            optimizer_updates=dict(pretraining=130*(i+1), G=0, D=0), songlength=4*(i+1),
            full_native_variables=19+i, full_trainable_parameters=4924855,
            all_graph_nodes_and_library_bitwise_identical=True) for i in range(4)])


def test_completed_original_diagnostic_is_not_a_bitwise_equivalence_claim():
    value = proof()
    value['all_trained_states_bitwise_identical'] = False
    require_complete_original_diagnostic(value, dict(exit_code=0, passed=True, resource_guard_aborted=False))


@pytest.mark.parametrize('defect', ['partial', 'missing_record', 'duplicate_epoch', 'replaced_executor',
                                   'short_batch', 'short_updates', 'smaller_network', 'resource_exit'])
def test_partial_or_modified_original_execution_cannot_support_three_way_audit(defect):
    value = copy.deepcopy(proof())
    guard = dict(exit_code=0, passed=True, resource_guard_aborted=False)
    if defect == 'partial':
        value['full_complete_epochs_executed'] = 3
    elif defect == 'missing_record':
        value['records'].pop()
    elif defect == 'duplicate_epoch':
        value['records'][3]['epoch'] = 2
    elif defect == 'replaced_executor':
        value['original_executor_unchanged'] = False
    elif defect == 'short_batch':
        value['records'][3]['full_training_batches'] = 129
    elif defect == 'short_updates':
        value['records'][3]['optimizer_updates']['pretraining'] = 519
    elif defect == 'smaller_network':
        value['records'][3]['full_trainable_parameters'] = 100
    else:
        guard.update(exit_code=1, passed=False, resource_guard_aborted=True)
    with pytest.raises(ValueError, match='original'):
        require_complete_original_diagnostic(value, guard)
