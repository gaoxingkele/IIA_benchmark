import pytest

from scripts.flow_matching.publish_complete_text_metadata_v29 import complete_text
from scripts.flow_matching.publish_result_listing_audited_v9 import KEY_FIELDS


def test_text_preserves_all_scientific_identity_and_nested_uncertainty_fields():
    fields=list(KEY_FIELDS)+['mean','evaluator_std','evaluator_ci95','uncertainty_scope','source']
    first={k:k for k in fields}
    first.update(epochs='400',pattern='point',missing_ratio='.2',score_recipe='s5t10',
        evaluator_std='.00002',evaluator_ci95='.1 / .2',uncertainty_scope='one native generator')
    second=dict(first,epochs='600',pattern='block',missing_ratio='.5',score_recipe='s10t20')
    text=complete_text(fields,[first,second])
    for value in ['400','600','point','block','.2','.5','s5t10','s10t20','.00002','.1 / .2','one native generator']:
        assert value in text
    with pytest.raises(ValueError): complete_text([k for k in fields if k!='epochs'],[first])
