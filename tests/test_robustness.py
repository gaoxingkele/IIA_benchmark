import json

from iia_benchmark.data import AlarmEpisode, AlarmEvent
from iia_benchmark.evaluation import (
    PerturbationScenario,
    apply_robustness_scenario,
    default_perturbation_grid,
    run_afc_robustness_benchmark,
)
from iia_benchmark.models import perturb_alarm_episode


def _episode(name: str, label: str, tag: str) -> AlarmEpisode:
    return AlarmEpisode(
        name,
        tuple(AlarmEvent(float(index), tag if index < 3 else f"COMMON_{index}") for index in range(6)),
        label=label,
    )


def test_detector_delay_drops_pre_detection_events() -> None:
    source = _episode("a", "A", "A")
    delayed = perturb_alarm_episode(source, detector_delay=2.5, seed=3)
    assert [event.timestamp for event in delayed.events] == [3.0, 4.0, 5.0]


def test_individual_and_mixed_scenarios_are_reproducible() -> None:
    source = _episode("a", "A", "A")
    mixed = PerturbationScenario("mixed", 0.25)
    first = apply_robustness_scenario(source, mixed, seed=11)
    second = apply_robustness_scenario(source, mixed, seed=11)
    assert first == second
    assert any(event.tag.startswith("SPURIOUS") for event in first.events)
    assert all(event.timestamp >= 1.25 for event in first.events if not event.tag.startswith("SPURIOUS"))


def test_default_grid_covers_five_corruption_families() -> None:
    grid = default_perturbation_grid((0.1,))
    assert {item.kind for item in grid} == {
        "missing",
        "spurious",
        "timing",
        "detector_delay",
        "mixed",
    }


def test_robustness_report_contains_progress_profiles_auc_and_audit_seeds() -> None:
    episodes = (_episode("a1", "A", "A"), _episode("b1", "B", "B"))

    def predictor(samples):
        return ["A" if any(event.tag == "A" for event in item.events) else "B" for item in samples]

    report = run_afc_robustness_benchmark(
        episodes,
        predictor,
        scenarios=(PerturbationScenario("missing", 0.2), PerturbationScenario("mixed", 0.2)),
        observation_progress=(0.5, 1.0),
        seeds=(2, 3, 5),
    )
    assert len(report.points) == 4
    assert report.seeds == (2, 3, 5)
    assert set(report.normalized_robustness_auc) == {
        "missing@0.5",
        "missing@1",
        "mixed@0.5",
        "mixed@1",
    }
    assert all(0 <= value <= 1 for value in report.normalized_robustness_auc.values())
    json.dumps(report.as_dict())
