import hashlib
import zipfile
import pytest
from scripts.flow_matching.acquire_RegCNF_RNODE_author_source_v1 import verified_archive


def node(path, data):
    return dict(path=path, size=len(data), type='blob',
                sha=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())


def test_complete_real_git_blob_verification_and_fingerprint(tmp_path):
    p = tmp_path / 'source.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('source/a.py', b'original\n')
        z.writestr('source/lib/b.py', b'second\n')
    tree = dict(truncated=False, tree=[node('a.py', b'original\n'), node('lib/b.py', b'second\n')])
    rows = verified_archive(p, tree, 'source')
    assert len(rows) == 2 and rows[0]['sha256'] == hashlib.sha256(b'original\n').hexdigest()
    tree['tree'][0]['sha'] = '0' * 40
    with pytest.raises(ValueError, match='blob mismatch'):
        verified_archive(p, tree, 'source')


@pytest.mark.parametrize('bad', ['../outside', '/absolute', 'source/../outside', 'source/C:/outside', 'source/a\\b'])
def test_no_unsafe_archive_member_can_be_extracted(tmp_path, bad):
    p = tmp_path / 'bad.zip'
    with zipfile.ZipFile(p, 'w') as z:
        # ZipInfo normalizes backslashes on Windows during construction; assign
        # afterward to actually exercise a malicious raw archive member.
        info = zipfile.ZipInfo('fixture')
        info.filename = bad
        z.writestr(info, b'x')
    with pytest.raises(ValueError, match='Unsafe'):
        verified_archive(p, dict(truncated=False, tree=[]), 'source')


def test_missing_original_file_or_truncated_metadata_is_fatal(tmp_path):
    p = tmp_path / 'empty.zip'
    with zipfile.ZipFile(p, 'w'):
        pass
    with pytest.raises(ValueError, match='Incomplete'):
        verified_archive(p, dict(truncated=False, tree=[node('a', b'x')]), 'source')
    with pytest.raises(ValueError, match='complete recursive'):
        verified_archive(p, dict(truncated=True, tree=[]), 'source')
