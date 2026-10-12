import csv
import json

import pytest

from scripts.flow_matching import retry_shcl_shared_data_v1 as retry


def fixture_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(rows)


def test_error_page_can_never_be_credited_as_dataset(tmp_path):
    path = tmp_path / 'train.csv'; path.write_text('Internal Server Error', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid CSV header'):
        retry.inspect_csv(path, dict(allowed_column_counts=[25,26], expected_rows=132481))


def test_complete_csv_audit_rejects_truncated_download_and_malformed_middle_row(tmp_path):
    path = tmp_path / 'test.csv'; specification = dict(allowed_column_counts=[2], expected_rows=3)
    fixture_csv(path, [['timestamp','label'],[0,0],[1,1]])
    with pytest.raises(ValueError, match='Full original expected row count'):
        retry.inspect_csv(path, specification)
    fixture_csv(path, [['timestamp','label'],[0,0],[1],[2,1]])
    with pytest.raises(ValueError, match='Malformed or incomplete'):
        retry.inspect_csv(path, specification)
    fixture_csv(path, [['timestamp','label'],[0,0],[1,0],[2,1]])
    result = retry.inspect_csv(path, specification)
    assert result['full_rows']==3 and result['complete_csv_structure_verified']
    assert result['numerical_loader_and_label_equivalence_pending']


def test_every_identified_bad_raw_payload_is_preserved_and_new_download_root_is_external():
    config = retry.read(retry.ROOT / 'configs/acquisition/fm_shcl_shared_data_retry.v1.json')
    assert len(config['sources'])==5
    assert config['download_root'].startswith('F:/aicoding/IIA_Data/')
    assert config['proxy']=='http://127.0.0.1:17890'
    for row in config['sources']:
        path = retry.ROOT / row['preserved_previous_file']
        assert path.read_bytes()==b'Internal Server Error'
        assert retry.sha(path)==row['preserved_previous_sha256']
        assert row['drive_file_id'] in row['url']
        assert row['publisher_checksum'] is None


def test_actual_full_operator_run_covers_every_registered_profile_without_training_credit():
    config = retry.read(retry.ROOT / 'configs/runtime/fm_shcl_released_operator_audit.v1.json')
    report = retry.read(retry.ROOT / config['output_root'] / 'full_released_operator_audit.json')
    reference = retry.read(retry.ROOT / config['reference_resource_runtime'])
    assert config['settings']==reference['settings'] and config['resource_lock']==reference['resource_lock']
    assert report['configuration_sha256']==retry.sha(retry.ROOT / 'configs/runtime/fm_shcl_released_operator_audit.v1.json')
    assert report['passed'] and len(report['records'])==len(config['profiles'])==13
    for row, profile in zip(report['records'], config['profiles']):
        assert row['full_input_shape']==[64,profile['parameters']['win_size'],profile['parameters']['c_in']]
        assert row['optimizer_updates_executed']==0
        assert row['inference_remains_stochastic'] and row['all12_native_output_tensors_RNG_replayed_bitwise']
        assert row['independent_float64_loss_passed'] and row['all_parameter_gradients_finite']
    assert report['synthetic_operator_only_not_benchmark_performance']
    assert not report['original_environment_equivalence'] and not report['paper_performance_reproduced']
