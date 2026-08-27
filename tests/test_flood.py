from iia_benchmark.data import AlarmEpisode, AlarmEvent, make_synthetic_floods
from iia_benchmark.models import (
    EmpiricalNextAlarmPredictor,
    detect_alarm_floods,
    perturb_alarm_episode,
    smith_waterman_similarity,
)


def test_alignment_and_next_alarm() -> None:
    assert smith_waterman_similarity(("A", "B", "C"), ("A", "B", "C")) == 1.0
    predictor = EmpiricalNextAlarmPredictor().fit((("A", "B", "C"), ("A", "B", "D")))
    assert predictor.predict(("A",)) in {"B", "C", "D"}


def test_flood_detection_and_perturbation_are_reproducible() -> None:
    events = tuple(AlarmEvent(float(index), f"A{index}") for index in range(6))
    floods = detect_alarm_floods(events, window_seconds=10, threshold=5)
    assert len(floods) == 1
    source = AlarmEpisode("x", make_synthetic_floods()[0].events)
    assert perturb_alarm_episode(source, missing_probability=0.2, spurious_count=2, seed=5) == perturb_alarm_episode(
        source, missing_probability=0.2, spurious_count=2, seed=5
    )
