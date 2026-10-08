import copy
import math

import pytest

from scripts.flow_matching.build_result_table import summarize, validate_result, display_missing_ratio


def test_missing_result_is_null_and_real_zero_is_preserved():
    assert summarize([])['local_mean'] is None
    one=summarize([0.0])
    assert one['local_mean']==0.0 and one['local_se'] is None
    assert summarize([1.,2.,3.])['local_se']==pytest.approx(1/math.sqrt(3))


@pytest.mark.parametrize('value',[float('nan'),float('inf'),None,'0.2'])
def test_invalid_metrics_cannot_become_numeric_scores(value):
    with pytest.raises(ValueError):summarize([value])


@pytest.mark.parametrize('change',[{'config_sha256':'other'}, {'diagnostic':True},
    {'config':{'id':'run','seed':2}}, {'config':{'id':'run','seed':1,'limit_test_cases':10}}])
def test_config_identity_and_diagnostic_limits_are_required(change):
    config={'id':'run','seed':1}
    result={'status':'completed','config_sha256':'frozen','config':copy.deepcopy(config),**change}
    with pytest.raises(ValueError):validate_result(config,'frozen',result)


def test_valid_config_identity_survives_without_mutation():
    cfg={'id':'run','fold':0,'reference_id':'paper','num_samples':100}
    result={'config_sha256':'frozen','config':copy.deepcopy(cfg)}
    validate_result(cfg,'frozen',result)
    assert result['config']==cfg


def test_historical_air_zero_config_is_not_reported_as_zero_missingness():
    cfg={'missing_ratio':0.0}
    assert display_missing_ratio(cfg,'csdi_cfmi','air36') is None
    assert cfg['missing_ratio']==0.0


def test_literal_zero_or_positive_masks_outside_historical_air_are_preserved():
    assert display_missing_ratio({'missing_ratio':0.0},'transfer','tep')==0.0
    assert display_missing_ratio({'missing_ratio':0.1},'csdi_cfmi','physionet2012')==0.1
