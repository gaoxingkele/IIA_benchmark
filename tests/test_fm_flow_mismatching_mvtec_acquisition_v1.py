import pytest

from scripts.flow_matching.acquire_flow_mismatching_mvtec_v1 import native_identity


def test_original_mvtec_training_and_mask_identities():
    train=native_identity('bottle/train/good/000.png',set())
    assert train['label']==0 and train['mask']=='' and train['split']=='train'
    mask='bottle/ground_truth/broken_large/000_mask.png'
    test=native_identity('bottle/test/broken_large/000.png',{mask})
    assert test['label']==1 and test['mask']==mask and test['split']=='test'
    with pytest.raises(ValueError,match='test mask'):
        native_identity('bottle/test/broken_large/000.png',set())
    with pytest.raises(ValueError,match='defect-free'):
        native_identity('bottle/train/broken_large/000.png',{mask})


def test_original_mvtec_image_depth_and_archive_identity_required():
    with pytest.raises(ValueError,match='identity'):
        native_identity('bottle/test/000.png',set())
    with pytest.raises(ValueError,match='Unsafe'):
        native_identity('../bottle/test/good/000.png',set())
