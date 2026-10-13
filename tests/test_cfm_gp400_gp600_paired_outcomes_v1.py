import pytest
from scripts.flow_matching import audit_cfm_gp400_gp600_paired_outcomes_v1 as current


def test_two_failed_seed_slots_never_become_a_three_seed_full_cohort_mean():
    rows=[dict(seed=s,GP600_original_scoring_complete=s not in (1103,1105),GP600_MSE=None if s in (1103,1105) else .1) for s in range(1103,1108)]
    summary=current.complete_pair_summary(rows)
    assert summary['GP600_complete_original_scores']==3 and summary['GP600_failed_original_scores']==2
    assert summary['GP600_full_five_seed_MSE_mean'] is None
    assert not summary['failure_free_five_seed_GP600_performance_complete']
    assert not summary['complete_success_subset_promoted_as_full_cohort']


def test_missing_or_duplicate_original_seed_cannot_pass_paired_coverage():
    rows=[dict(seed=s,GP600_original_scoring_complete=True,GP600_MSE=.1) for s in range(1103,1108)]
    for invalid in [rows[:-1],rows[:-1]+[rows[0]]]:
        with pytest.raises(ValueError,match='five distinct'):current.complete_pair_summary(invalid)
