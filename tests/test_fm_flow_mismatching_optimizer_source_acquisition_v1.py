import subprocess
import zipfile

import pytest

from scripts.flow_matching.acquire_flow_mismatching_optimizer_sources_v1 import verified_extract


def original_fixture(tmp_path):
    root=tmp_path/'git';subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'optimizer.py').write_bytes(b'original optimizer bytes\n')
    (root/'LICENSE').write_bytes(b'original license\n')
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','source'],check=True)
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in ['optimizer.py','LICENSE']:z.writestr('candidate-'+commit+'/'+name,(root/name).read_bytes())
    return root,commit,archive


def test_every_source_and_license_blob_matches_immutable_git_and_is_preserved(tmp_path):
    root,commit,archive=original_fixture(tmp_path);target=tmp_path/'extracted'
    records=verified_extract(archive,target,root,commit,'candidate')
    assert {r['path'] for r in records}=={'optimizer.py','LICENSE'}
    assert (target/'optimizer.py').read_bytes()==(root/'optimizer.py').read_bytes()
    with pytest.raises(FileExistsError):verified_extract(archive,target,root,commit,'candidate')


def test_valid_crc_cannot_hide_changed_native_optimizer(tmp_path):
    root,commit,_=original_fixture(tmp_path);archive=tmp_path/'forged.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('candidate-'+commit+'/optimizer.py',b'changed optimizer')
    with pytest.raises(ValueError,match='bytes differ'):verified_extract(archive,tmp_path/'forged',root,commit,'candidate')


def test_missing_license_or_source_cannot_be_called_complete(tmp_path):
    root,commit,_=original_fixture(tmp_path);archive=tmp_path/'missing.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('candidate-'+commit+'/optimizer.py',(root/'optimizer.py').read_bytes())
    with pytest.raises(ValueError,match='Incomplete original'):verified_extract(archive,tmp_path/'missing',root,commit,'candidate')


def test_wrong_repository_wrapper_cannot_relabel_source(tmp_path):
    root,commit,archive=original_fixture(tmp_path)
    with pytest.raises(ValueError,match='wrapper differs'):verified_extract(archive,tmp_path/'other',root,commit,'other')
