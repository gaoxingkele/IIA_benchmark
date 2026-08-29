import numpy as np
import pytest
from itertools import combinations

from iia_benchmark.models import (
    AlarmToken,
    ClosedAlarmPattern,
    MaximumEntropyNextAlarmPredictor,
    accelerated_alarm_alignment,
    charm_closed_alarm_patterns,
    criterion_c_alarm_flood_detection,
    maximum_entropy_single_constraint,
    priority_match_score,
    representative_alarm_patterns,
)


def test_criterion_c_inherits_recent_and_excludes_long_standing_tags() -> None:
    states = np.zeros((30, 4), dtype=int)
    states[2:, 0] = 1
    states[12:18, 1] = 1
    states[20:, 2] = 1
    result = criterion_c_alarm_flood_detection(
        states,
        tag_names=["standing", "recent", "new", "idle"],
        attention_window=10,
        long_standing_window=20,
        update_step=10,
        threshold=1,
    )
    assert "standing" in result.attention_sets[0]
    assert "standing" not in result.attention_sets[-1]
    assert "new" in result.attention_sets[-1]


def test_criterion_c_delay_rejects_one_sample_chatter() -> None:
    states = np.zeros((20, 2), dtype=int)
    states[5, 0] = 1
    result = criterion_c_alarm_flood_detection(
        states,
        attention_window=1,
        long_standing_window=5,
        update_step=1,
        threshold=1,
        delay_samples=2,
    )
    assert np.max(result.raw_detection) == 1
    assert np.max(result.delayed_detection) == 0


def test_priority_seeded_alignment_recovers_common_subsequence() -> None:
    first = [
        AlarmToken("X", 0, 3),
        AlarmToken("A", 1, 1),
        AlarmToken("B", 2, 2),
        AlarmToken("C", 3, 3),
    ]
    second = [
        AlarmToken("Q", 0, 3),
        AlarmToken("A", 1.1, 1),
        AlarmToken("B", 2.1, 2),
        AlarmToken("R", 4, 3),
    ]
    result = accelerated_alarm_alignment(
        first, second, seed_length=2, time_tolerance=1.0, extension_band=2
    )
    assert result.seeds
    assert result.similarity > 0.4
    assert result.aligned_pairs == ((1, 1), (2, 2))
    assert result.cells_evaluated < len(first) * len(second)


def test_charm_closed_patterns_and_representative_clustering() -> None:
    transactions = [
        {"A", "B", "C"},
        {"A", "B", "C"},
        {"A", "B", "D"},
        {"A", "B", "D"},
    ]
    patterns = charm_closed_alarm_patterns(transactions, minimum_support=0.5)
    itemsets = {pattern.items for pattern in patterns}
    assert frozenset({"A", "B"}) in itemsets
    assert frozenset({"A", "B", "C"}) in itemsets
    representatives = representative_alarm_patterns(patterns, similarity_threshold=0.5)
    assert len(representatives) < len(patterns)
    assert set().union(*(item.items for item in representatives)) == {"A", "B", "C", "D"}


def test_charm_direct_closure_matches_brute_force() -> None:
    transactions = [
        {"A", "B", "D"},
        {"A", "C", "D"},
        {"A", "B", "C", "D"},
        {"B", "C", "E"},
        {"A", "B", "C", "E"},
        {"A", "B", "C", "D", "E"},
    ]
    minimum_count = 2
    items = sorted(set().union(*transactions))
    frequent = {}
    for size in range(2, len(items) + 1):
        for candidate in combinations(items, size):
            itemset = frozenset(candidate)
            tids = frozenset(
                index for index, transaction in enumerate(transactions) if itemset <= transaction
            )
            if len(tids) >= minimum_count:
                frequent[itemset] = tids
    expected = {
        itemset
        for itemset, tids in frequent.items()
        if not any(itemset < other and tids == other_tids for other, other_tids in frequent.items())
    }
    observed = {
        pattern.items
        for pattern in charm_closed_alarm_patterns(
            transactions, minimum_support=minimum_count
        )
    }
    assert observed == expected


def test_maximum_entropy_predictor_learns_dominant_next_alarm() -> None:
    sequences = [("A", "B", "D")] * 16 + [("A", "C")] * 3 + [("B", "E")]
    model = MaximumEntropyNextAlarmPredictor().fit(sequences)
    probabilities = model.predict_proba(("A", "B"))
    assert abs(sum(probabilities.values()) - 1.0) < 1e-10
    assert model.predict(("A", "B")) == "D"
    assert probabilities["D"] > probabilities["C"]


def test_book_table_5_5_and_equation_5_16_alignment() -> None:
    assert [priority_match_score(level, 3) for level in (1, 2, 3)] == [6.0, 4.5, 3.0]
    first = [
        AlarmToken(str(tag), index)
        for index, tag in enumerate((3, 2, 1, 4, 3, 2, 2))
    ]
    second = [
        AlarmToken(str(tag), index)
        for index, tag in enumerate((3, 4, 2, 1, 4, 2))
    ]
    result = accelerated_alarm_alignment(
        first,
        second,
        seed_length=3,
        max_seeds=7,
        extension_band=10,
    )
    assert [(first[i].tag, second[j].tag) for i, j in result.aligned_pairs] == [
        ("3", "3"),
        ("2", "2"),
        ("1", "1"),
        ("4", "4"),
        ("2", "2"),
    ]


def test_book_section_5_3_representative_pattern_example() -> None:
    patterns = tuple(
        ClosedAlarmPattern(
            frozenset(map(str, items)), frozenset({index}), 0.2
        )
        for index, items in enumerate(
            (
                {2, 3, 4, 5},
                {1, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 2, 3, 5},
                {1, 2, 3, 4},
            )
        )
    )
    representatives = representative_alarm_patterns(
        patterns, similarity_threshold=1 / 3
    )
    assert len(representatives) == 1
    assert representatives[0].items == frozenset({"1", "2", "3", "4", "5"})
    assert len(representatives[0].descendants) == 5


def test_book_table_5_15_maximum_entropy_constraints() -> None:
    candidates = ("x3", "x4", "x5")
    x1 = maximum_entropy_single_constraint(
        candidates,
        constrained_candidate="x3",
        constrained_probability=3 / 20,
    )
    pair = maximum_entropy_single_constraint(
        candidates,
        constrained_candidate="x4",
        constrained_probability=4 / 5,
    )
    x2 = maximum_entropy_single_constraint(
        candidates,
        constrained_candidate="x5",
        constrained_probability=1 / 20,
    )
    assert np.allclose(
        [x1.lagrange_multiplier, pair.lagrange_multiplier, x2.lagrange_multiplier],
        [-1.0414, 2.0794, -2.2513],
        atol=5e-5,
    )
    assert pair.probabilities["x4"] == pytest.approx(0.8)
    assert max(pair.probabilities, key=pair.probabilities.get) == "x4"
