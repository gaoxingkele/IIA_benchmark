import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_PILE_classification_v1 as s


def fixture(tmp_path):
    values = np.ones((3, 2, 256), dtype=np.float64)
    masks = np.ones((3, 256), dtype=np.float64)
    rows = np.arange(3, dtype=np.int64)
    for name, array in [('values.npy', values), ('input_mask.npy', masks), ('original_instance_rows.npy', rows)]:
        np.save(tmp_path / name, array)
    record = dict(instances=3, channels=2, sequence_length=256, values_shape=[3, 2, 256], values_dtype='float64',
        source_coordinates_copied_and_checked=1536, label_values_exported=False,
        original_instance_order_and_all_channels_retained=True,
        files_sha256={p.name: s.sha(p) for p in tmp_path.glob('*.npy')})
    return record


def test_complete_label_free_cache_and_independent_raw_case_channel_counts(tmp_path):
    record = fixture(tmp_path)
    raw = tmp_path / 'TRAIN.ts'
    raw.write_text('@timestamps false\n@classLabel true a b\n@data\n1,2:3,4:a\n5,6:7,8:b\n9,0:1,2:a\n')
    assert s.raw_case_shape(raw) == (3, 2)
    proof = s.verify_arrays(tmp_path, record, *s.raw_case_shape(raw))
    assert proof['coordinates'] == 1536 and proof['label_arrays_absent']


@pytest.mark.parametrize('kind', ['row_order', 'nonfinite', 'mask_nonbinary', 'dtype'])
def test_reject_semantic_corruption_even_when_the_file_hash_is_refreshed(tmp_path, kind):
    record = fixture(tmp_path)
    if kind == 'row_order': name, value = 'original_instance_rows.npy', np.array([1, 0, 2], dtype=np.int64)
    elif kind == 'nonfinite':
        name, value = 'values.npy', np.ones((3, 2, 256)); value[2, 1, 255] = np.inf
    elif kind == 'mask_nonbinary':
        name, value = 'input_mask.npy', np.ones((3, 256)); value[2, 255] = .5
    else: name, value = 'values.npy', np.ones((3, 2, 256), dtype=np.float32)
    np.save(tmp_path / name, value); record['files_sha256'][name] = s.sha(tmp_path / name)
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 3, 2)


def test_reject_changed_bytes_and_extra_label_array(tmp_path):
    record = fixture(tmp_path)
    with (tmp_path / 'values.npy').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'): s.verify_arrays(tmp_path, record, 3, 2)
    record = fixture(tmp_path); np.save(tmp_path / 'labels.npy', np.zeros(3))
    with pytest.raises(ValueError, match='label-free'): s.verify_arrays(tmp_path, record, 3, 2)


def test_reject_dropped_raw_cases_and_wrong_channel_cardinality(tmp_path):
    record = fixture(tmp_path)
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 4, 2)
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 3, 1)


def test_raw_shape_rejects_channel_mismatch_in_independent_stream(tmp_path):
    path = tmp_path / 'TRAIN.ts'
    path.write_text('@timestamps false\n@data\n1:2:a\n1:a\n')
    with pytest.raises(ValueError): s.raw_case_shape(path)
