import copy
import pytest

from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import safe_relative,validate_split_rows


@pytest.mark.parametrize('name',['../raw.png','/absolute/raw.png','C:/raw.png','category\\raw.png','candle/NUL.png','candle/name.'])
def test_original_archive_paths_cannot_escape_or_alias_windows_files(name):
    with pytest.raises(ValueError):safe_relative(name)


def test_all_original_images_and_masks_are_bound_to_disjoint_native_splits():
    rows=[dict(object='candle',split='train',label='normal',image='candle/Data/Images/Normal/a.JPG',mask=''),
          dict(object='candle',split='test',label='anomaly',image='candle/Data/Images/Anomaly/b.JPG',mask='candle/Data/Masks/Anomaly/b.png')]
    images={r['image'] for r in rows};files=images|{rows[-1]['mask']}
    counts,categories=validate_split_rows(rows,images,files)
    assert sum(counts.values())==2 and categories=={'candle'}
    with pytest.raises(ValueError,match='disjoint'):validate_split_rows(rows+[rows[0]],images,files)
    with pytest.raises(ValueError,match='Every original'):validate_split_rows(rows,images|{'candle/extra.JPG'},files)
    with pytest.raises(ValueError,match='masks required'):validate_split_rows(rows,images,images)
    changed=copy.deepcopy(rows);changed[1]['split']='train'
    with pytest.raises(ValueError,match='Anomalies'):validate_split_rows(changed,images,files)
