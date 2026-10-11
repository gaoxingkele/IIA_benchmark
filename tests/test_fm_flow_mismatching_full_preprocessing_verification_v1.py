import numpy as np
import pytest

from scripts.flow_matching.verify_full_visa_preprocessing_v1 import check_row,check_normalized_moments


@pytest.mark.parametrize('field',['image','mask','label','category'])
def test_single_corrupted_image_coordinate_mask_or_identity_is_rejected(field):
    image=np.zeros((3,2,2),dtype=np.float32);mask=np.zeros((2,2),dtype=np.uint8)
    actual_image=image.copy();actual_mask=mask.copy();label=0;category=2
    if field=='image':actual_image[0,0,0]=.001
    if field=='mask':actual_mask[0,0]=1
    if field=='label':label=1
    if field=='category':category=3
    with pytest.raises(ValueError):check_row(actual_image,actual_mask,label,category,image,mask,0,2)


def test_nonfinite_matching_transformation_cannot_pass():
    image=np.full((3,2,2),np.inf,dtype=np.float32);mask=np.zeros((2,2),dtype=np.uint8)
    with pytest.raises(ValueError,match='image differs'):check_row(image,mask,0,0,image,mask,0,0)


def test_statistics_inconsistent_with_complete_training_population_are_rejected():
    tolerance=dict(absolute_mean=1e-6,absolute_variance=2e-6)
    check_normalized_moments(np.zeros(3),np.ones(3)*100,100,tolerance)
    with pytest.raises(ValueError,match='moments differ'):
        check_normalized_moments(np.ones(3),np.ones(3)*100,100,tolerance)
