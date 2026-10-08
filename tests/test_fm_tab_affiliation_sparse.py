from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.tab_affiliation_sparse import load_sparse_reference, sparse_outer_partition
from scripts.flow_matching.tab_range_metrics import load_reference

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/mtsad_protocol_audit/sources/tab/original/ts_benchmark/evaluation/metrics'


def same_outputs(expected, actual):
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize('seed', range(20))
def test_random_all_reference_outputs_bit_exact(seed):
    reference, sparse = load_reference(SOURCE), load_sparse_reference(SOURCE)
    rng = np.random.default_rng(seed)
    truth = (rng.random(180) < .3).astype(int)
    truth[0] = truth[-1] = 1
    prediction = (rng.random(180) < .45).astype(int)
    args = (reference['events'](prediction), reference['events'](truth), (0, 180))
    same_outputs(reference['pr_from_events'](*args), sparse['pr_from_events'](*args))
    # Source namespace and singleton recall partition contract remain intact.
    assert reference['pr_from_events'].__globals__['affiliation_partition'] is not sparse_outer_partition


@pytest.mark.parametrize('predicted', [[], [(0, 180)], [(1, 2), (19, 20), (41, 50)], [(9.5, 10.5)]])
def test_empty_broad_boundary_fractional_and_none_semantics(predicted):
    reference, sparse = load_reference(SOURCE), load_sparse_reference(SOURCE)
    args = (predicted, [(10, 11), (30, 31), (50, 51)], (0, 180))
    same_outputs(reference['pr_from_events'](*args), sparse['pr_from_events'](*args))


def test_invalid_events_retain_original_exceptions():
    reference, sparse = load_reference(SOURCE), load_sparse_reference(SOURCE)
    for args in [([], [], (0, 5)), ([(2, 2)], [(1, 3)], (0, 5)),
                 ([(3, 4), (1, 2)], [(1, 3)], (0, 5)), ([(1, 3)], [(1, 3)], (2, 4))]:
        with pytest.raises(Exception) as original:
            reference['pr_from_events'](*args)
        with pytest.raises(type(original.value)) as actual:
            sparse['pr_from_events'](*args)
        assert str(actual.value) == str(original.value)


def test_partition_keeps_all_nonzero_intersections_without_gt_by_pred_matrix():
    zones = [(0, 2), (2, 4), (4, 6), (6, 8)]
    predictions = [(0, 1), (1.5, 6.5), (7, 8)]
    assert sparse_outer_partition(predictions, zones) == [
        [(0, 1), (1.5, 2)], [(2, 4)], [(4, 6)], [(6, 6.5), (7, 8)]]
    # A large disjoint case retains O(G+P) actual fragments, not G*P Nones.
    size = 10000
    partition = sparse_outer_partition([(3*i, 3*i+1) for i in range(size)],
                                      [(3*i, 3*i+3) for i in range(size)])
    assert sum(len(row) for row in partition) == size
