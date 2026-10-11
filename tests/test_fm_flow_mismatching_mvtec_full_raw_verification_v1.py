import copy

import pytest

from scripts.flow_matching.verify_full_mvtec_raw_binding_v1 import identity_checks


def rows_and_files():
    rows=[dict(category='bottle',split='train',defect_type='good',label=0,image='bottle/train/good/001.png',mask='',image_sha256='a'),
          dict(category='bottle',split='test',defect_type='broken',label=1,image='bottle/test/broken/001.png',mask='bottle/ground_truth/broken/001_mask.png',image_sha256='b')]
    files={r['image'] for r in rows}|{rows[-1]['mask']}
    return rows,files


def test_extra_raw_anomaly_mask_cannot_be_omitted_from_complete_audit():
    rows,files=rows_and_files()
    identity_checks(rows,files)
    with pytest.raises(ValueError,match='coverage differs'):identity_checks(rows,files|{'bottle/ground_truth/broken/002_mask.png'})


@pytest.mark.parametrize('field,value',[('label',0),('split','train'),('category','cable'),('mask','bottle/ground_truth/other/001_mask.png')])
def test_native_directory_cannot_be_relabelled_as_a_different_identity(field,value):
    rows,files=rows_and_files();rows=copy.deepcopy(rows);rows[-1][field]=value
    with pytest.raises(ValueError,match='binding differs'):identity_checks(rows,files)


def test_encoded_content_overlap_is_disclosed_without_changing_native_split():
    rows,files=rows_and_files();rows[-1]['image_sha256']='a'
    result=identity_checks(rows,files)
    assert result['cross_split_identical_encoded_file_content_images']==2
    assert rows[0]['split']=='train' and rows[-1]['split']=='test'
