import hashlib

import pytest

from scripts.flow_matching.fetch_ls4_ushcn_author_assets_v1 import validate_asset, verify_blob


def asset(content=b'abc'):
    return dict(path='data/public_datasets/flow_matching/ls4_native_ushcn_v1/raw/a.csv',
                upstream_path='a.csv', upstream_mode='100644', expected_size=len(content),
                git_blob_sha1=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest(),
                url='https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/commit/a.csv')


def test_git_blob_identity_and_tampering(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_bytes(b'abc')
    assert verify_blob(p, asset())['sha256'] == hashlib.sha256(b'abc').hexdigest()
    p.write_bytes(b'abd')
    with pytest.raises(ValueError, match='identity'):
        verify_blob(p, asset())
    p.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='size'):
        verify_blob(p, asset())


def test_source_url_cannot_drift(tmp_path):
    a = asset()
    a['url'] = a['url'].replace('/commit/', '/master/')
    with pytest.raises(ValueError, match='pinned'):
        validate_asset(a, 'commit', tmp_path)


def test_symlinks_are_only_inert_text(tmp_path):
    a = asset()
    a['upstream_mode'] = '120000'
    with pytest.raises(ValueError, match='inert'):
        validate_asset(a, 'commit', tmp_path)
    a['path'] += '.upstream_link.txt'
    assert validate_asset(a, 'commit', tmp_path).name.endswith('.upstream_link.txt')


def test_destination_cannot_escape(tmp_path):
    a = asset()
    a['path'] = 'data/public_datasets/flow_matching/ls4_native_ushcn_v1/../../escape.csv'
    with pytest.raises(ValueError, match='escaped'):
        validate_asset(a, 'commit', tmp_path)
