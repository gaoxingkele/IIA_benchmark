"""Executable differential checks against pinned TAB, including threshold ties."""
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.tab_range_metrics import accelerated_vus, load_reference


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/mtsad_protocol_audit/sources/tab/original/ts_benchmark/evaluation/metrics'


@pytest.mark.parametrize('seed', range(6))
def test_rank_prefix_matches_all_original_thresholds_and_buffers(seed):
    reference = load_reference(SOURCE)
    rng = np.random.default_rng(seed)
    labels = (rng.random(70) < .25).astype(int)
    labels[0], labels[-1] = 1, 1
    scores = np.round(rng.normal(size=len(labels)), 1)  # Deliberate ties.
    max_buffer = 8
    original = reference['vus'].generate_curve(labels, scores, max_buffer)
    accelerated = accelerated_vus(labels, scores, max_buffer)
    assert accelerated['VUS_ROC'] == pytest.approx(original[-2], abs=1e-12)
    assert accelerated['VUS_PR'] == pytest.approx(original[-1], abs=1e-12)
    assert accelerated['buffers_evaluated'] == 9 and accelerated['threshold_samples'] == 250


def test_no_label_or_constant_score_shortcut_is_invented():
    reference = load_reference(SOURCE)
    labels = np.array([0, 1, 1, 0, 0, 1, 0])
    scores = np.ones(7)
    original = reference['vus'].generate_curve(labels, scores, 4)
    actual = accelerated_vus(labels, scores, 4)
    assert actual['VUS_ROC'] == pytest.approx(original[-2], abs=1e-12)
    assert actual['VUS_PR'] == pytest.approx(original[-1], abs=1e-12)
    with pytest.raises(ValueError, match='single-class'):
        accelerated_vus(np.zeros(7), scores, 4)


def test_affiliation_uses_original_functions_and_event_length_helper():
    reference = load_reference(SOURCE)
    truth = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(reference['get_list_anomaly'](truth), [2, 3])
    assert reference['affiliation_f'](truth, truth) == pytest.approx(1.)
