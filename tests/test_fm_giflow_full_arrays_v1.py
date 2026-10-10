import numpy as np
import pytest

from scripts.flow_matching.audit_giflow_full_arrays_v1 import batch_errors


def test_native_mape_uses_nonzero_target_cutoff():
    errors=batch_errors(np.array([1.,4.,7.]),np.array([0.,2.,5.]))
    assert errors['native_mean_batch_mae']==pytest.approx(5/3)
    assert errors['native_mean_batch_mse']==3
    assert errors['native_mean_batch_mape_percent']==pytest.approx(70)


@pytest.mark.parametrize('p,t', [([],[]),([1],[1,2]),([float('nan')],[1]),([1],[0])])
def test_reject_incomplete_or_undefined_native_arrays(p,t):
    with pytest.raises(ValueError):
        batch_errors(np.array(p),np.array(t))
