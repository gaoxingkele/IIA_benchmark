import copy
import pytest

from scripts.flow_matching.register_crnngan_executor_cache_v4 import registration


def records():
    return [dict(epoch=i, all_native_variable_hashes_bitwise_identical=True,
        whole_graph_geometry_and_rng_seeds_identical=True, trainable_parameters=4924855,
        full_native_variables=19+i, full_training_batches=130, songlength=4*(i+1),
        optimizer_updates=dict(pretraining=130*(i+1), G=0, D=0)) for i in range(4)]


@pytest.mark.parametrize('defect', ['missing_epoch', 'empty', 'variables', 'graph', 'budget', 'network', 'curriculum'])
def test_incomplete_or_different_actual_full_epoch_proof_cannot_register_full_training(defect):
    audit = dict(full_complete_epochs_verified=4, all_full_native_train_states_bitwise_identical=True,
                 all_original_graph_geometry_and_random_seeds_identical=True, records=records())
    if defect == 'missing_epoch':
        audit['records'].pop()
    elif defect == 'empty':
        audit['records'] = []
    else:
        key, value = dict(variables=('all_native_variable_hashes_bitwise_identical', False),
            graph=('whole_graph_geometry_and_rng_seeds_identical', False), budget=('full_training_batches', 129),
            network=('trainable_parameters', 100), curriculum=('songlength', 4))[defect]
        audit['records'][3][key] = value
    with pytest.raises(ValueError, match='four real complete'):
        registration({}, {}, copy.deepcopy(audit))
