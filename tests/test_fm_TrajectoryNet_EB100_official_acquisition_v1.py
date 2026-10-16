import hashlib
import subprocess
import pytest
from scripts.flow_matching.acquire_TrajectoryNet_EB100_official_v1 import verify_blob


def test_git_blob_integrity_matches_git_hash_object_without_line_ending_filters(tmp_path):
    path=tmp_path/'binary.npz';payload=b'PK\x03\x04\x00data\r\n\xff'*1000;path.write_bytes(payload)
    official=subprocess.check_output(['git','hash-object','--no-filters',str(path)],text=True).strip()
    proof=verify_blob(path,len(payload),official)
    assert proof['git_blob_sha1']==official and proof['sha256']==hashlib.sha256(payload).hexdigest()


def test_raw_sha1_must_not_be_mistaken_for_official_git_blob_identity(tmp_path):
    path=tmp_path/'artifact';payload=b'PKdataset';path.write_bytes(payload)
    with pytest.raises(ValueError):verify_blob(path,len(payload),hashlib.sha1(payload).hexdigest())
    assert path.read_bytes()==payload


def test_wrong_size_and_same_length_corruption_are_rejected_without_raw_overwrite(tmp_path):
    path=tmp_path/'artifact';path.write_bytes(b'PKofficial')
    original=subprocess.check_output(['git','hash-object','--no-filters',str(path)],text=True).strip()
    with pytest.raises(ValueError):verify_blob(path,999,original)
    path.write_bytes(b'PKcorrupted'[:10])
    with pytest.raises(ValueError):verify_blob(path,10,original)
