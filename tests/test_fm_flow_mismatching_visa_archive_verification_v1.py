import hashlib
import io
import tarfile

import pytest

from scripts.flow_matching.verify_visa_archive_and_split_v1 import split_audit, verify_tar


def test_complete_archive_rejects_same_length_extracted_corruption(tmp_path):
    raw=tmp_path/'raw';raw.mkdir(); (raw/'image.dat').write_bytes(b'original')
    archive=tmp_path/'original.tar'
    with tarfile.open(archive,'w') as source:
        item=tarfile.TarInfo('image.dat');item.size=8;source.addfile(item,io.BytesIO(b'original'))
    records=[dict(path='image.dat',bytes=8,sha256=hashlib.sha256(b'original').hexdigest())]
    assert verify_tar(archive,raw,records)==1
    (raw/'image.dat').write_bytes(b'corrupt!')
    with pytest.raises(ValueError,match='content mismatch'):verify_tar(archive,raw,records)


def test_complete_archive_rejects_omitted_file_even_when_other_files_match(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'a').write_bytes(b'a')
    archive=tmp_path/'source.tar'
    with tarfile.open(archive,'w') as source:
        item=tarfile.TarInfo('a');item.size=1;source.addfile(item,io.BytesIO(b'a'))
    records=[dict(path=name,bytes=1,sha256=hashlib.sha256(name.encode()).hexdigest()) for name in ['a','b']]
    with pytest.raises(ValueError,match='Incomplete'):verify_tar(archive,raw,records)


def test_distinct_paths_with_same_content_across_splits_are_disclosed():
    rows=[dict(object='candle',split=split,label='normal',image=split+'.png',mask='',image_sha256='identical')
        for split in ['train','test']]
    result=split_audit(rows)
    assert len(result['cross_split_identical_file_content_groups'])==1
    assert result['cross_split_identical_file_content_images']==2
    assert [r['split'] for r in result['cross_split_identical_file_content_groups'][0]['images']]==['train','test']


def test_original_training_anomaly_cannot_pass_split_audit():
    rows=[dict(object='candle',split='train',label='anomaly',image='a.png',mask='mask.png',image_sha256='a')]
    with pytest.raises(ValueError,match='contains anomaly'):split_audit(rows)
