import csv

import pytest

from scripts.flow_matching import prepare_shcl_released_data_v1 as preparation


def test_timestamp_removal_preserves_all_feature_tokens_and_rows_without_overwriting_raw(tmp_path):
    source=tmp_path/'raw.csv'; destination=tmp_path/'features.csv'
    header=['timestamp_(min)']+['feature_'+str(i) for i in range(25)]
    rows=[['0001']+['0.000000000000000123']*25,['0002']+['NaN']*25]
    with source.open('w',encoding='utf-8',newline='') as stream: csv.writer(stream).writerows([header,*rows])
    original=source.read_bytes(); preparation.drop_timestamp(source,destination)
    with destination.open(encoding='utf-8',newline='') as stream: actual=list(csv.reader(stream))
    assert actual==[header[1:],*[row[1:] for row in rows]]
    assert source.read_bytes()==original
    with pytest.raises(FileExistsError): preparation.drop_timestamp(source,destination)


def test_unregistered_timestamp_schema_is_rejected(tmp_path):
    source=tmp_path/'raw.csv'; source.write_text('time,value\n0,1\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Declared PSM timestamp'):
        preparation.drop_timestamp(source,tmp_path/'features.csv')


def test_actual_full_native_data_preparation_keeps_original_rows_and_all52_loader_modes():
    config=preparation.native.read(preparation.native.ROOT/'configs/datasets/fm_shcl_released_data_views.v3.json')
    report=preparation.native.read(config['view_root']+'/full_data_preparation_audit.json')
    assert report['passed'] and report['full_datasets']==5
    assert report['full_native_loader_modes_audited']==52
    assert report['preserved_raw_files_unchanged'] and not report['original_author_loader_source_changed']
    assert not report['paper_performance_reproduced'] and not report['original_training_executed']
    for row,binding in zip(report['records'],config['datasets']):
        if binding.get('expected_shapes'):
            assert row['full_raw_train_shape']==binding['expected_shapes']['train']
            assert row['full_raw_test_shape']==binding['expected_shapes']['test']
        assert row['source_train_validation_split_row']==int(row['full_raw_train_shape'][0]*.8)
        assert row['scaler_fit_includes_original_validation_rows'] and not row['test_rows_used_for_scaler_fit']
        assert row['raw_and_feature_row_order_unchanged'] and not row['strict_grouped_TAB_protocol']
        assert {r['mode'] for r in row['native_loaders']}=={'train','val','test','thre'}
        for path in row['output_files']:
            assert preparation.native.sha(path['path'])==path['sha256']
