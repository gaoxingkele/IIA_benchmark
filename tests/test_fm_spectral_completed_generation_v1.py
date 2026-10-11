import copy
import math
import statistics

import pytest
from scipy.stats import t

from scripts.flow_matching.audit_spectral_completed_generation_v1 import evaluator_statistics,METRICS


def original():
    raw=[.1,.2,.3,.4,.5]
    mean=statistics.mean(raw)
    std=statistics.stdev(raw)
    se=std/math.sqrt(5)
    margin=float(t.ppf(.975,4))*se
    return {key:dict(raw_repeats=list(raw),mean=mean,std=std,se=se,
        ci95_low=mean-margin,ci95_high=mean+margin,author_display_halfwidth=margin) for key in METRICS}


@pytest.mark.parametrize('fault',['lost_metric','partial','nonfinite','negative','mean','std','se','ci95_low','ci95_high','author_display_halfwidth'])
def test_all_original_evaluators_and_statistics_required_without_faking_five_training_seeds(fault):
    metrics=original()
    reviewed=evaluator_statistics(metrics)
    assert all(r['independent_training_models']==1 and r['planned_original_training_models']==5
               and r['between_training_seed_STD'] is None and r['between_training_seed_CI95'] is None
               for r in reviewed.values())
    bad=copy.deepcopy(metrics)
    if fault=='lost_metric': del bad['context_fid']
    elif fault=='partial': bad['context_fid']['raw_repeats'].pop()
    elif fault=='nonfinite': bad['context_fid']['raw_repeats'][0]=float('nan')
    elif fault=='negative': bad['context_fid']['raw_repeats'][0]=-.1
    else: bad['context_fid'][fault]+=.01
    with pytest.raises(ValueError): evaluator_statistics(bad)
