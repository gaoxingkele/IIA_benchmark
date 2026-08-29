import numpy as np
from itertools import combinations

from iia_benchmark.models import (
    AlarmToken,
    MaximumEntropyNextAlarmPredictor,
    accelerated_alarm_alignment,
    charm_closed_alarm_patterns,
    criterion_c_alarm_flood_detection,
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
