from iia_benchmark.data import make_synthetic_causal_alarm_series
from iia_benchmark.models import TransferEntropyRanker, transfer_entropy


def test_transfer_entropy_finds_delayed_root() -> None:
    series = make_synthetic_causal_alarm_series(length=1800, lag=3)
    ranking = TransferEntropyRanker(
        max_lag=5, permutations=19, significance=0.1, seed=2
    ).rank(series, target="TARGET")
    assert ranking[0][0] == "ROOT"
    assert ranking[0][2] == 3
    assert transfer_entropy(series["ROOT"], series["TARGET"], lag=3) > 0
