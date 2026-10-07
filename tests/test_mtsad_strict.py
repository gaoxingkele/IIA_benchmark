import numpy as np
import pytest

from iia_benchmark.evaluation.mtsad_strict import calibrate_validation, evaluate_entities


def frozen(scores=None):
    return calibrate_validation(
        np.array([0., 0., 1., 1.]) if scores is None else scores,
        false_alarm_percent=50, calibration_split="validation",
        calibration_dataset_sha256="a" * 64, model_artifact_sha256="b" * 64,
    )


def test_calibration_excludes_test_split_and_requires_provenance():
    with pytest.raises(ValueError, match="validation split"):
        calibrate_validation(np.ones(3), false_alarm_percent=1,
                             calibration_split="test", calibration_dataset_sha256="a" * 64,
                             model_artifact_sha256="b" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        calibrate_validation(np.ones(3), false_alarm_percent=1,
                             calibration_split="validation", calibration_dataset_sha256="unknown",
                             model_artifact_sha256="b" * 64)


def test_one_detection_does_not_fill_the_event():
    result = evaluate_entities({"machine": (np.array([0, 1, 1, 1, 0]), np.array([0., 0., 1., 0., 0.]))},
                               expected_lengths={"machine": 5}, calibration=frozen())
    assert result["micro"]["recall"] == pytest.approx(1 / 3)
    assert result["micro"]["f1"] == pytest.approx(.5)
    assert result["point_adjustment"] is False


@pytest.mark.parametrize("scores", [np.array([0., np.nan, 1.]), np.array([0., 1.]), np.array([[0., 1., 2.]])])
def test_missing_or_unscored_timestamps_are_rejected(scores):
    with pytest.raises(ValueError):
        evaluate_entities({"a": (np.array([0, 1, 0]), scores)}, expected_lengths={"a": 3}, calibration=frozen())


def test_entity_coverage_cannot_be_silently_reduced():
    with pytest.raises(ValueError, match="Entity IDs"):
        evaluate_entities({"a": (np.array([0, 1]), np.array([0., 1.]))},
                          expected_lengths={"a": 2, "b": 3}, calibration=frozen())


def test_test_changes_do_not_change_frozen_threshold_and_auc_uses_raw_scores():
    calibration = frozen()
    before = calibration.to_dict()
    result = evaluate_entities({"a": (np.array([0, 0, 1, 1]), np.array([.1, .2, .3, .4]))},
                               expected_lengths={"a": 4}, calibration=calibration)
    assert result["micro"]["f1"] == 0
    assert result["micro"]["auroc"] == 1
    evaluate_entities({"a": (np.array([1, 1, 0, 0]), np.array([10., 2., 30., 4.]))},
                      expected_lengths={"a": 4}, calibration=calibration)
    assert calibration.to_dict() == before
