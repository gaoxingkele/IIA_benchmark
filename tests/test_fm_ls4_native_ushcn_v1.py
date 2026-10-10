import numpy as np
import pandas as pd
import pytest

from scripts.flow_matching.prepare_ls4_native_ushcn_v1 import audit_frame, audit_partition


def frame():
    rows = []
    for i in range(10):
        for t in [0, 150, 151, 152, 153, 154]:
            rows.append([i, t] + [1.]*5 + [1.]*5)
    return pd.DataFrame(rows, columns=['ID', 'Time']+[f'Value_{i}' for i in range(5)]+[f'Mask_{i}' for i in range(5)])


def test_complete_raw_frame_geometry():
    assert audit_frame(frame())['observed_values'] == 300
    f = frame(); f.loc[0, 'Mask_0'] = .5
    with pytest.raises(ValueError, match='binary'):
        audit_frame(f)
    f = frame(); f.loc[0, 'Mask_0'] = 0
    with pytest.raises(ValueError, match='placeholders'):
        audit_frame(f)


def test_ids_and_duplicates_cannot_be_hidden():
    f = frame(); f.ID += 1
    with pytest.raises(ValueError, match='contiguous'):
        audit_frame(f)
    f = frame()
    with pytest.raises(ValueError, match='row identity'):
        audit_frame(pd.concat([f, f.iloc[:1]]))


def test_missing_future_is_not_a_complete_original_series():
    f = frame(); f = f[~((f.ID == 0) & (f.Time > 150))]
    with pytest.raises(ValueError, match='validation-eligible'):
        audit_frame(f)


def test_whole_series_disjoint_partition():
    parts = dict(train=np.arange(7), val=np.arange(7, 9), test=np.array([9]))
    assert audit_partition(parts, 10) == dict(train=7, val=2, test=1)
    parts['test'] = np.array([0])
    with pytest.raises(ValueError, match='exactly once'):
        audit_partition(parts, 10)
