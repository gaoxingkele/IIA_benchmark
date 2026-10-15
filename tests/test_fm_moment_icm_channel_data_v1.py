from collections import Counter
import numpy as np
import pytest

from scripts.flow_matching import moment_icm_channel_data_v1 as s

POLICY = 'appendix_D_partition8_remainder_replacement'


@pytest.mark.parametrize('channels', [1, 2, 7, 8, 9, 16, 17, 963, 1345])
def test_appendix_groups_keep_every_channel_and_oversample_only_the_final_remainder(channels):
    groups = s.channel_groups(channels, 13, 'example', POLICY)
    assert len(groups) == (1 if channels == 1 else (channels + 7) // 8)
    assert {c for group in groups for c in group} == set(range(channels))
    assert all(len(group) == (1 if channels == 1 else 8) for group in groups)
    for i, group in enumerate(groups):
        original = tuple(range(i * 8, min(i * 8 + 8, channels)))
        assert group[:len(original)] == original
        assert set(group[len(original):]).issubset(original)


def test_plan_is_stable_by_dataset_and_seed_without_touching_external_numpy_rng():
    np.random.seed(10)
    state = np.random.get_state()
    first = s.channel_groups(19, 13, 'A', POLICY)
    s.channel_groups(19, 13, 'B', POLICY)
    assert first == s.channel_groups(19, 13, 'A', POLICY)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_do_not_silently_substitute_main_text_subsampling():
    with pytest.raises(ValueError, match='main-text'):
        s.channel_groups(19, 13, 'A', 'subsample8')


def fixture(tmp_path, name, channels, cases=3, split='TRAIN'):
    path = tmp_path / name / split
    path.mkdir(parents=True)
    values = np.arange(cases * channels * 256, dtype=np.float64).reshape(cases, channels, 256)
    mask = np.ones((cases, 256), dtype=np.int64)
    mask[:, :4] = 0
    np.save(path / 'values.npy', values)
    np.save(path / 'input_mask.npy', mask)
    np.save(path / 'original_instance_rows.npy', np.arange(cases, dtype=np.int64) + 100)
    # A label-free pretraining reader must not attempt to open this invalid file.
    (path / 'labels.npy').write_bytes(b'must not be opened')
    return s.GroupedCases(path, dataset_key=name, split=split, seed=13, policy=POLICY), values, mask


def test_real_array_reader_preserves_case_identity_all_coordinates_dtype_and_masks(tmp_path):
    data, values, masks = fixture(tmp_path, 'A', 10)
    try:
        seen = set()
        for i in range(len(data)):
            item = data.item(i)
            case = item['case_index']
            seen.add((case, item['group_index']))
            assert item['original_instance_row'] == case + 100
            assert np.array_equal(item['values'], values[case, item['channel_indices'], :])
            assert np.array_equal(item['input_mask'], masks[case])
            assert item['values'].dtype == values.dtype
        assert len(seen) == 6
        item['values'][:] = -100
        assert np.array_equal(data.values, values)
        with pytest.raises(IndexError): data.item(-1)
        with pytest.raises(IndexError): data.item(len(data))
    finally:
        data.close()


def test_paired_complete_epoch_covers_each_item_once_and_keeps_both_partial_bucket_batches(tmp_path):
    a, _, _ = fixture(tmp_path, 'A', 1, cases=5)
    b, _, _ = fixture(tmp_path, 'B', 10, cases=3)
    try:
        plan = s.paired_epoch_batches([a, b], 4, 13)
        assert plan == s.paired_epoch_batches([a, b], 4, 13)
        actual = Counter(item for batch in plan for item in batch.items)
        expected = Counter((source, item) for source, data in enumerate([a, b]) for item in range(len(data)))
        assert actual == expected
        assert sorted(len(batch.items) for batch in plan) == [1, 2, 4, 4]
        for batch in plan:
            independent = s.materialize_batch([a, b], batch)
            icm = s.materialize_batch([a, b], batch)
            assert np.array_equal(independent['values'], icm['values'])
            assert independent['provenance'] == icm['provenance']
            assert independent['values'].shape == (len(batch.items), batch.width, 256)
    finally:
        a.close(); b.close()


def test_refuse_training_and_test_mix_or_duplicate_dataset_identity(tmp_path):
    a, _, _ = fixture(tmp_path, 'A', 1)
    b, _, _ = fixture(tmp_path, 'B', 2, split='TEST')
    try:
        with pytest.raises(ValueError, match='official split'): s.paired_epoch_batches([a, b], 4, 13)
        with pytest.raises(ValueError, match='unique'): s.paired_epoch_batches([a, a], 4, 13)
    finally:
        a.close(); b.close()


@pytest.mark.parametrize('channels', [0, -1, True, 2.5])
def test_invalid_channel_cardinality_is_not_reinterpreted(channels):
    with pytest.raises(ValueError): s.channel_groups(channels, 13, 'A', POLICY)
