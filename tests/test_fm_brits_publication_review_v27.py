import copy
import statistics

import pytest
from scipy.stats import t

from scripts.flow_matching.review_brits_two_seed_publication_v27 import check_statistics


def test_independent_seed_interval_must_not_be_narrowed_or_count_replays_as_seeds():
    values = [.8, .9]
    mean, std = statistics.mean(values), statistics.stdev(values)
    se = std / 2**.5
    margin = float(t.ppf(.975, 1)) * se
    row = dict(n='2', required_repeats='5', mean=mean, std=std, se=se,
        ci95_low=mean-margin, ci95_high=mean+margin)
    assert check_statistics(row, values)['ci95_high'] > 1
    for key, value in [('n', '3'), ('required_repeats', '2'), ('ci95_high', .91),
                       ('mean', .89), ('std', 0), ('se', float('nan'))]:
        bad = copy.deepcopy(row)
        bad[key] = value
        with pytest.raises(ValueError):
            check_statistics(bad, values)
