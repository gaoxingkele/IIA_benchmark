import hashlib
from pathlib import Path
import pytest

from scripts.flow_matching import acquire_otcfm_author_torch_wheels_v1 as s


@pytest.mark.parametrize('fault', [None, 'partial', 'changed'])
def test_complete_official_payload_requires_both_full_size_and_sha256(tmp_path, fault):
    content = b'full reproducible wheel fixture'
    path = tmp_path / 'asset.whl'; path.write_bytes(content)
    asset = dict(path=str(path), expected_bytes=len(content), checksum='sha256:' + hashlib.sha256(content).hexdigest())
    if fault == 'partial': path.write_bytes(content[:-1])
    elif fault == 'changed': path.write_bytes(b'X' + content[1:])
    if fault:
        with pytest.raises(ValueError): s.verify_asset(asset)
    else: s.verify_asset(asset)


def test_registered_binaries_match_exact_author_instructions_and_official_hashes():
    c = s.read(s.ROOT / s.CONFIG)
    assert c['proxy'] == 'http://127.0.0.1:17890'
    instructions = (s.ROOT / c['author_requirements']['path']).read_text()
    assert 'torch==1.12.1+cu113 torchvision==0.13.1+cu113' in instructions
    assert s.sha(s.ROOT / c['author_requirements']['path']) == c['author_requirements']['sha256']
    assert [x['filename'] for x in c['sources']] == ['torch-1.12.1+cu113-cp310-cp310-win_amd64.whl',
        'torchvision-0.13.1+cu113-cp310-cp310-win_amd64.whl']
    assert all(x['expected_bytes'] > 0 and x['url'].startswith('https://download-r2.pytorch.org/')
               and x['checksum'].startswith('sha256:') and len(x['checksum']) == 71 for x in c['sources'])
    assert Path(c['environment']).name == 'otcfm_author_py310_v1'
