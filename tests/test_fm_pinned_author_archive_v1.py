import hashlib
import zipfile

import pytest

from scripts.flow_matching.fetch_pinned_github_archive_v1 import blob_identity,verified_archive,preserve_assets,safe_relative


def fixture(tmp_path,contents):
    archive=tmp_path/'source.zip';assets=[]
    with zipfile.ZipFile(archive,'w') as z:
        for name,value in contents.items():
            z.writestr('repo-deadbeef/'+name,value)
            assets.append(dict(upstream_path=name,expected_size=len(value),git_blob_sha1=blob_identity(value),upstream_mode='100644'))
    return archive,dict(archive_prefix='repo-deadbeef',assets=assets,asset_root='data/public_datasets/flow_matching/original')


def test_archive_requires_every_registered_blob_and_exact_git_identity(tmp_path):
    archive,c=fixture(tmp_path,{'main.py':b'pass\n','datasets/raw.csv':b'a,b\n1,2\n'})
    records=verified_archive(archive,c)
    assert len(records)==2 and records[0]['sha256']==hashlib.sha256(b'pass\n').hexdigest()
    c['assets'][0]['git_blob_sha1']='0'*40
    with pytest.raises(ValueError,match='blob differs'):verified_archive(archive,c)


def test_archive_cannot_hide_extra_missing_or_duplicate_files(tmp_path):
    archive,c=fixture(tmp_path,{'main.py':b'pass\n'})
    c['assets'].append(dict(c['assets'][0],upstream_path='missing.py'))
    with pytest.raises(ValueError,match='tree differs'):verified_archive(archive,c)
    c['assets'].pop()
    with zipfile.ZipFile(archive,'a') as z:z.writestr('repo-deadbeef/extra.py',b'pass')
    with pytest.raises(ValueError,match='tree differs'):verified_archive(archive,c)


def test_raw_asset_conflict_is_preserved_and_upstream_symlink_is_inert_text(tmp_path):
    archive,c=fixture(tmp_path,{'raw.csv':b'original','fold':b'/author/private/path'})
    c['assets'][1]['upstream_mode']='120000'
    records=verified_archive(archive,c);preserve_assets(archive,c,records,tmp_path)
    dest=tmp_path/c['asset_root']
    assert (dest/'fold.upstream_link.txt').read_bytes()==b'/author/private/path'
    assert not (dest/'fold').exists()
    (dest/'raw.csv').write_bytes(b'user data')
    with pytest.raises(FileExistsError):preserve_assets(archive,c,records,tmp_path)
    assert (dest/'raw.csv').read_bytes()==b'user data'


@pytest.mark.parametrize('name',['../outside','C:/outside','/outside','nested/../../outside','a\\b','a//b'])
def test_archive_path_cannot_escape_registered_raw_storage(name):
    with pytest.raises(ValueError):safe_relative(name)


def test_registered_source_destination_cannot_escape_dataset_storage(tmp_path):
    archive,c=fixture(tmp_path,{'main.py':b'pass\n'});records=verified_archive(archive,c)
    c['asset_root']='outside'
    with pytest.raises(ValueError,match='escaped'):preserve_assets(archive,c,records,tmp_path)
