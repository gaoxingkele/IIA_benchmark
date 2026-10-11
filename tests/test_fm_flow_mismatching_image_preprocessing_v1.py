"""Small image fixtures check leakage/integrity, not model performance."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts.flow_matching.prepare_flow_mismatching_original_images_v1 import (
    prepare,resized_rgb,training_statistics,unified_rows,
)


def make_image(root, name, values, split='train', label=0, mask=''):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.asarray(values,dtype=np.uint8)).save(path)
    return dict(image=name,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                split=split,label=label,category='test_category',mask=mask,mask_sha256=None)


def fixtures(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    training=np.array([[[20,40,60],[80,100,120]],[[120,140,160],[180,200,220]]],dtype=np.uint8)
    train=make_image(raw,'train.png',training)
    test=make_image(raw,'test.png',np.ones((2,2,3))*255,split='test')
    return raw,train,test


def test_statistics_never_decode_or_use_test_values(tmp_path):
    raw,train,test=fixtures(tmp_path)
    original=training_statistics(raw,[train,test],2)
    # No test image is even opened during fitting; an absent test path cannot affect it.
    test['image']='absent_test.png';test['image_sha256']='not_a_real_checksum'
    assert training_statistics(raw,[train,test],2)==original
    assert original['test_images_used']==0 and original['normal_training_images']==1


def test_raw_changed_bytes_are_rejected_before_use(tmp_path):
    raw,train,test=fixtures(tmp_path)
    (raw/train['image']).write_bytes(b'preserved simulation of corruption')
    with pytest.raises(ValueError,match='SHA256'):resized_rgb(raw,train,2)


def test_complete_native_fixture_arrays_preserve_raw_and_refuse_overwrite(tmp_path):
    raw,train,test=fixtures(tmp_path)
    audit=dict(passed=True,all_original_images_and_masks_decoded_and_verified=True,
        maintained_anomalib_full_file_checksum_verified=True,native_image_manifest=[train,test],raw_root=str(raw))
    audit_path=tmp_path/'full_raw.json';audit_path.write_text(json.dumps(audit),encoding='utf-8')
    config=dict(dataset='MVTec-AD',full_raw_audit='full_raw.json',expected_images=2,expected_categories=1,
                expected_native_splits={'train':1,'test':1},raw_root_field='raw_root',
                derived_directory='derived',asset_root='raw',minimum_free_disk_bytes=0,resolution=2)
    original={p.name:p.read_bytes() for p in raw.iterdir()}
    report=prepare(tmp_path,config,lambda _:None)
    result=json.loads(report.read_text(encoding='utf-8'))
    assert result['complete_original_images']==2 and result['raw_files_written_or_overwritten']==0
    assert len(result['artifacts'])==8 and not result['paper_preprocessing_equivalence_certified']
    data=np.load(tmp_path/'derived/train_images.npy')
    np.testing.assert_allclose(data.mean((0,2,3)),np.zeros(3),atol=1e-7)
    np.testing.assert_allclose(data.std((0,2,3)),np.ones(3),atol=1e-7)
    assert {p.name:p.read_bytes() for p in raw.iterdir()}==original
    with pytest.raises(FileExistsError):prepare(tmp_path,config,lambda _:None)


def test_native_train_anomaly_and_incomplete_raw_audit_are_rejected(tmp_path):
    raw,train,test=fixtures(tmp_path)
    audit=dict(passed=True,all_original_images_and_masks_decoded_and_verified=True,
        maintained_anomalib_full_file_checksum_verified=True,native_image_manifest=[dict(train,label=1),test])
    with pytest.raises(ValueError,match='normal-only'):unified_rows(audit,'MVTec-AD')
    audit['native_image_manifest']=[train,test];audit['maintained_anomalib_full_file_checksum_verified']=False
    with pytest.raises(ValueError,match='archive checksum'):unified_rows(audit,'MVTec-AD')
