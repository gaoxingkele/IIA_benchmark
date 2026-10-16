from types import FunctionType
import pytest
from scripts.flow_matching import audit_otcfm_CITE13176_GEO_full60_replay_v1 as s


def test_entire_native_checkpoint_and_full_cost_LP_worker_bytecode_is_unchanged():
    fn=FunctionType(s.base.worker.__code__,s.namespace())
    assert fn.__code__ is s.base.worker.__code__
    assert fn.__globals__['native'] is s.native and fn.__globals__['inspect'] is s.inspect
    assert s.base.interior_geometry([6071,7643,8485,7195])==[(6071,7643),(7643,8485)]
    assert len(s.integrity.full_metric_names(4))==36


def test_cold_inspector_rejects_other_cohort_before_loading_learned_artifacts():
    with pytest.raises(ValueError):s.validate_record({},dict(dataset='released_author_CITE',data={}),{},{} )
