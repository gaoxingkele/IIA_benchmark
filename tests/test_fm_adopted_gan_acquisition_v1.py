import hashlib
import pytest

from scripts.flow_matching.acquire_adopted_gan_sources_v1 import safe_member, verify_blob


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'repo/../../escape', 'repo/C:/escape', 'repo\\escape'])
def test_rejects_windows_and_posix_archive_escape(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_git_blob_verification_covers_binary_bytes_and_length():
    payload = b'\0binary\xff\r\n'
    entry = dict(size=len(payload), sha=hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest())
    assert verify_blob(payload, entry) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError):
        verify_blob(payload[:-1], entry)
    with pytest.raises(ValueError):
        verify_blob(b'\1' + payload[1:], entry)
