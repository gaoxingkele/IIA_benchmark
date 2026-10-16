import io
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as s


def test_tied_rank_integrals_match_external_library_for_full_vectors():
    rng = np.random.default_rng(17319)
    for size in [2, 17, 1007, 6931]:
        scores = rng.integers(-5, 7, size).astype(float)
        labels = rng.integers(0, 2, size)
        labels[0], labels[-1] = 0, 1
        result = s.ranking(scores.tolist(), labels.tolist())
        assert result['auroc'] == pytest.approx(roc_auc_score(labels, scores), abs=1e-14)
        assert result['average_precision'] == pytest.approx(average_precision_score(labels, scores), abs=1e-14)
    assert s.ranking([3, 3], [0, 1]) == dict(auroc=.5, average_precision=.5)
    assert s.ranking([3, 4], [0, 0]) == dict(auroc=None, average_precision=None)


def test_validation_quantiles_ties_and_threshold_equality():
    values = sorted([0., .01, .1, 3., 5., 7., 9., 20.])
    for ratio in [.1, 1., 5., 25., 50., 99.]:
        assert s.percentile(values, ratio) == pytest.approx(np.percentile(values, 100 - ratio), abs=1e-14)
    r = s.at_threshold([0., 1., 1., 3., 0., 4.], [0, 1, 1, 1, 0, 1], 1.)
    assert r['micro']['true_positives'] == 2
    assert r['micro']['false_negatives'] == 2
    assert r['event_metrics']['events'] == r['event_metrics']['detected_events'] == 2
    assert r['event_metrics']['mean_detection_delay_points'] == 1.


def test_every_payload_coordinate_dtype_finiteness_and_tail_required():
    def saved(value):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        stream.seek(0)
        return stream
    a = np.arange(1009, dtype=np.float32)
    assert list(s.vector(saved(a), 1009, '<f4')[0]) == a.tolist()
    for value in [a[:-1], a.astype(np.float64), np.append(a[:-1], np.float32(np.nan))]:
        with pytest.raises(ValueError):
            s.vector(saved(value), 1009, '<f4')
    stream = saved(a)
    stream.seek(0, 2)
    stream.write(b'x')
    stream.seek(0)
    with pytest.raises(ValueError):
        s.vector(stream, 1009, '<f4')
    with pytest.raises(ValueError):
        s.vector(saved(np.array([0, 2], dtype=np.uint8)), 2, '|u1')


def test_exact_counts_and_independent_threshold_mismatch_are_rejected():
    with pytest.raises(ValueError):
        s.equal_fields(dict(true_positives=1000000), dict(true_positives=1000001), dict(rtol=2e-6, atol=2e-6))
    with pytest.raises(ValueError):
        s.equal_fields(dict(threshold=.5), dict(threshold=.51), dict(rtol=2e-6, atol=2e-6))


def test_live_audit_imports_no_numerical_or_author_implementation():
    script = ('import sys;from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1;'
              'assert not any(x in sys.modules for x in ["numpy","torch","sklearn","scripts.flow_matching.run_tsad_experiment"])')
    subprocess.run([sys.executable, '-B', '-c', script], cwd=s.ROOT, check=True)
