from copy import deepcopy
import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_PILE_classification_v2 as s


def fixture(tmp_path):
    for name, array in [('values.npy', np.ones((9, 2, 256), dtype=np.float64)),
                        ('input_mask.npy', np.ones((9, 256))), ('original_instance_rows.npy', np.arange(9, dtype=np.int64))]:
        np.save(tmp_path / name, array)
    return dict(instances=9, channels=2, sequence_length=256, values_shape=[9, 2, 256], values_dtype='float64',
        source_coordinates_copied_and_checked=4608, label_values_exported=False,
        original_instance_order_and_all_channels_retained=True,
        files_sha256={p.name: s.sha(p) for p in tmp_path.glob('*.npy')})


def test_complete_bounded_read_scan_releases_each_mapping_and_never_calls_old_wholemap_reader(tmp_path, monkeypatch):
    record = fixture(tmp_path)
    monkeypatch.setattr(s.original, 'verify_arrays', lambda *args: (_ for _ in ()).throw(AssertionError('old wholemap')))
    proof = s.verify_arrays(tmp_path, record, 9, 2, block_bytes=1024)
    signal = proof['complete_read_mapping_scans']['signal']
    assert signal['blocks'] == 9 and signal['checked_coordinates'] == 4608
    assert signal['maximum_active_mapping_bytes'] == 4096
    assert signal['mapping_released_after_every_block'] and proof['whole_signal_array_never_mapped']


@pytest.mark.parametrize('kind', ['row_order', 'nonfinite', 'mask_nonbinary', 'dtype', 'extra_bytes'])
def test_reject_corruption_after_updating_hash_including_last_block(tmp_path, kind):
    record = fixture(tmp_path)
    if kind == 'row_order': name, value = 'original_instance_rows.npy', np.arange(9, dtype=np.int64)[::-1]
    elif kind == 'nonfinite':
        name, value = 'values.npy', np.ones((9, 2, 256)); value[-1, -1, -1] = np.inf
    elif kind == 'mask_nonbinary':
        name, value = 'input_mask.npy', np.ones((9, 256)); value[-1, -1] = .5
    elif kind == 'dtype': name, value = 'values.npy', np.ones((9, 2, 256), dtype=np.float32)
    else: name, value = 'values.npy', None
    if value is None:
        with (tmp_path / name).open('ab') as f: f.write(b'extra')
    else: np.save(tmp_path / name, value)
    record['files_sha256'][name] = s.sha(tmp_path / name)
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 9, 2, block_bytes=1024)


def test_refuse_extra_label_array_and_wrong_raw_cardinality(tmp_path):
    record = fixture(tmp_path)
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 8, 2)
    np.save(tmp_path / 'labels.npy', np.zeros(9))
    with pytest.raises(ValueError): s.verify_arrays(tmp_path, record, 9, 2)


def recovery_fixture():
    expected = dict(proof_sha256='proof', worker_command=['python', '--worker'], configuration_sha256='recovery',
        numerical_source_configuration_sha256='original', preserved_failed_guard_sha256='failed')
    guard = dict(passed=True, exit_code=0, resource_guard_aborted=False, proof_sha256='proof',
                 exact_child_handle=dict(command=['python', '--worker']))
    proof = dict(configuration_sha256='recovery', numerical_source_configuration_sha256='original',
        preserved_failed_guard_sha256='failed', diagnosed_chunked_output_mapping_recovery=True,
        signal_mapping_chunk_bytes=64 * 2**20, original_complete_prefix_cases_compared=5414,
        original_failed_outcome_not_overwritten=True, full_original_failed_export_equivalence_certified=False)
    return proof, guard, expected


def test_explicit_recovered_source_chain_accepts_only_its_own_guard_and_configuration():
    proof, guard, expected = recovery_fixture(); s.validate_recovery_metadata(proof, guard, expected)
    for key in ['configuration_sha256', 'numerical_source_configuration_sha256', 'preserved_failed_guard_sha256']:
        broken = deepcopy(proof); broken[key] = 'different'
        with pytest.raises(ValueError): s.validate_recovery_metadata(broken, guard, expected)
    broken = deepcopy(guard); broken['exact_child_handle']['command'] = ['original', '--dataset', 'InsectWingbeat']
    with pytest.raises(ValueError): s.validate_recovery_metadata(proof, broken, expected)
