import json
import sys
from pathlib import Path

import pytest

from iia_benchmark.config.storage import project_relative_path, resolve_project_path


def mount(root, external):
    config = root / 'configs/storage/data_storage.v1.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'mounts': [{'logical_path': 'data/public_datasets',
                                            'physical_path': str(external)}]}))
    logical = root / 'data/public_datasets'
    logical.parent.mkdir(parents=True)
    try:
        logical.symlink_to(external, target_is_directory=True)
    except OSError:
        # Windows directory junctions do not need symbolic-link privilege.
        import subprocess
        result = subprocess.run(['powershell', '-NoProfile', '-Command',
                                 f"New-Item -ItemType Junction -Path '{logical}' -Target '{external}' | Out-Null"],
                                capture_output=True)
        if result.returncode:
            pytest.skip('Test host cannot create a directory mount')
    return logical


def test_configured_mount_preserves_logical_registration(tmp_path):
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    logical = mount(root, external)
    (external / 'data.bin').write_bytes(b'raw')
    result = resolve_project_path(root, 'data/public_datasets/data.bin')
    assert result == logical / 'data.bin'
    assert result.resolve() == external / 'data.bin'
    assert project_relative_path(root, external / 'data.bin') == 'data/public_datasets/data.bin'


def test_unconfigured_or_misdirected_mount_is_rejected(tmp_path):
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    mount(root, external)
    config = root / 'configs/storage/data_storage.v1.json'
    config.unlink()
    with pytest.raises(ValueError, match='escapes'):
        resolve_project_path(root, 'data/public_datasets/data.bin')
    mount_entry = {'mounts': [{'logical_path': 'data/public_datasets', 'physical_path': str(tmp_path / 'other')}]}
    config.write_text(json.dumps(mount_entry))
    with pytest.raises(ValueError, match='escapes'):
        resolve_project_path(root, 'data/public_datasets/data.bin')


def test_escape_and_absolute_registration_rejected(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        resolve_project_path(tmp_path, '../outside')
    with pytest.raises(ValueError, match='relative'):
        resolve_project_path(tmp_path, tmp_path / 'data', relative_only=True)


def test_nested_unauthorized_link_rejected(tmp_path):
    root, external, other = tmp_path / 'repo', tmp_path / 'storage', tmp_path / 'outside'
    root.mkdir()
    external.mkdir()
    other.mkdir()
    mount(root, external)
    try:
        (external / 'escape').symlink_to(other, target_is_directory=True)
    except OSError:
        import subprocess
        import sys
        if sys.platform != 'win32':
            pytest.skip('Test host cannot create nested directory link')
        result = subprocess.run(['powershell', '-NoProfile', '-Command',
                                 f"New-Item -ItemType Junction -Path '{external / 'escape'}' -Target '{other}' | Out-Null"],
                                capture_output=True)
        if result.returncode:
            pytest.skip('Test host cannot create nested junction')
    with pytest.raises(ValueError, match='escapes'):
        resolve_project_path(root, 'data/public_datasets/escape/data.bin')


def test_acquisition_and_integrity_audit_follow_configured_mount(tmp_path, monkeypatch):
    import hashlib
    from scripts.data_acquisition import download_flow_matching_bundle as bundle
    from scripts.flow_matching.verify_registered_snapshot import verify_source
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    mount(root, external)
    payload = b'channel,label\n1,0\n2,1\n'
    (external / 'raw.csv').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    item = {'id': 'raw', 'path': 'data/public_datasets/raw.csv', 'format': 'csv',
            'size_bytes': len(payload), 'checksum': {'algorithm': 'sha256', 'value': digest}}
    monkeypatch.setattr(bundle, 'ROOT', root)
    assert bundle.acquire(item, None, audit_only=True)['status'] == 'available'
    record = verify_source(item, root)
    assert record['status'] == 'verified'
    assert record['path'] == item['path']
    assert record['sha256'] == digest


def test_runner_evidence_keeps_same_registered_path_after_relocation(tmp_path):
    from iia_benchmark.runner import _data_evidence, _resolve
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    mount(root, external)
    (external / 'raw.csv').write_bytes(b'1,2\n')
    logical = 'data/public_datasets/raw.csv'
    assert _resolve(root, logical).read_bytes() == b'1,2\n'
    assert _data_evidence([external / 'raw.csv'], root)['files'][0]['path'] == logical


def test_mega_existing_file_verification_uses_logical_provenance(tmp_path, monkeypatch):
    from scripts.data_acquisition import download_mega_public_bundle as mega
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    mount(root, external)
    (external / 'raw.bin').write_bytes(b'raw')
    monkeypatch.chdir(root)
    monkeypatch.setattr(mega, 'verify_plain_file', lambda *_: 'verified_digest')
    result = mega.PublicMega(None).download(
        {'id': 'source', 'path': 'data/public_datasets', 'url': 'public',
         'evidence_url': 'author', 'paper_ids': ['paper']},
        {'name': 'raw.bin', 'size': 3, 'key': '00' * 32})
    assert result['status'] == 'existing_not_overwritten'
    assert result['relative_path'] == 'data/public_datasets/raw.bin'
    assert Path(result['path']) == external / 'raw.bin'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows extended-length path spelling')
def test_mounted_path_accepts_equivalent_extended_length_spelling(tmp_path, monkeypatch):
    root, external = tmp_path / 'repo', tmp_path / 'storage'
    root.mkdir()
    external.mkdir()
    logical = mount(root, external)
    (external / 'data.bin').write_bytes(b'raw')
    resolve = Path.resolve

    def extended(self, *args, **kwargs):
        value = resolve(self, *args, **kwargs)
        return Path('\\\\?\\' + str(value)) if self.name == 'data.bin' else value

    monkeypatch.setattr(Path, 'resolve', extended)
    assert resolve_project_path(root, 'data/public_datasets/data.bin') == logical / 'data.bin'
    assert project_relative_path(root, external / 'data.bin') == 'data/public_datasets/data.bin'
