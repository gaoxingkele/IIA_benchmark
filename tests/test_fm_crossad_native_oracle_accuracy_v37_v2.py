import numpy as np
from scripts.flow_matching.review_crossad_complete_native_inventory_v37_v2 import original_author_accuracy


def test_original_author_accuracy_preserves_strict_threshold_ties():
    scores=np.array([.25,.5,.75]);labels=np.array([0,1,1])
    assert original_author_accuracy(scores,.5,labels)==2/3
    assert np.mean((scores>=.5)==labels)==1.
