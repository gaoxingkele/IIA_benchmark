import io
import tarfile

import pytest

from scripts.flow_matching.prepare_ls4_native_physionet_v1 import safe_archive_members


@pytest.mark.parametrize('name', ['../outside.txt', 'set-a/../../outside.txt', '/outside.txt', 'set-b/1.txt'])
def test_native_archive_cannot_escape_or_replace_another_patient_set(tmp_path, name):
    archive = tmp_path / 'patients.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        member = tarfile.TarInfo(name)
        member.size = 1
        output.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        safe_archive_members(archive, tmp_path / 'derived', 'set-a')


def test_native_archive_requires_all_4000_patients(tmp_path):
    archive = tmp_path / 'patients.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        member = tarfile.TarInfo('set-a/1.txt')
        member.size = 1
        output.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='4000'):
        safe_archive_members(archive, tmp_path / 'derived', 'set-a')
