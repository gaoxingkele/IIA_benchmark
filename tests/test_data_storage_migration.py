import hashlib
import os
from pathlib import Path

import pytest

from scripts.data_acquisition.migrate_data_storage import audit_stable_tree, copy_verified, inventory


def roots(tmp_path):
    source, destination = tmp_path / 'old', tmp_path / 'new'
    source.mkdir()
    destination.mkdir()
    return source, destination


def test_verified_copy_preserves_source_and_metadata(tmp_path):
    source, destination = roots(tmp_path)
    original = source / 'data.bin'
    payload = b'raw data\x00' * 200
    original.write_bytes(payload)
    record = copy_verified(source, destination, 'data.bin')
    assert original.read_bytes() == (destination / 'data.bin').read_bytes() == payload
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['source_signature'] == record['destination_signature']
    assert audit_stable_tree(source, destination, {'data.bin': record})['complete']


def test_existing_unrelated_destination_is_never_overwritten(tmp_path):
    source, destination = roots(tmp_path)
    (source / 'data.bin').write_bytes(b'original')
    (destination / 'data.bin').write_bytes(b'other raw data')
    with pytest.raises(FileExistsError):
        copy_verified(source, destination, 'data.bin')
    assert (source / 'data.bin').read_bytes() == b'original'
    assert (destination / 'data.bin').read_bytes() == b'other raw data'


def test_changed_or_new_source_blocks_cutover(tmp_path):
    source, destination = roots(tmp_path)
    (source / 'data.bin').write_bytes(b'original')
    record = copy_verified(source, destination, 'data.bin')
    (source / 'data.bin').write_bytes(b'changed')
    (source / 'new.bin').write_bytes(b'new')
    audit = audit_stable_tree(source, destination, {'data.bin': record})
    assert not audit['complete']
    assert audit['changed'] == ['data.bin']
    assert audit['missing'] == ['new.bin']


def test_copy_rejects_path_escape(tmp_path):
    source, destination = roots(tmp_path)
    with pytest.raises(ValueError, match='escapes'):
        copy_verified(source, destination, '../other.bin')


def test_uncommitted_copy_remnant_blocks_cutover(tmp_path):
    source, destination = roots(tmp_path)
    (source / 'data.bin').write_bytes(b'original')
    record = copy_verified(source, destination, 'data.bin')
    (destination / 'data.bin.iia_copying_interrupted').write_bytes(b'partial')
    audit = audit_stable_tree(source, destination, {'data.bin': record})
    assert not audit['complete']
    assert audit['uncommitted_copy_files'] == ['data.bin.iia_copying_interrupted']
    assert (source / 'data.bin').read_bytes() == b'original'


@pytest.mark.skipif(os.name != 'nt', reason='Windows extended-length path spelling')
def test_copy_accepts_equivalent_extended_length_path(tmp_path, monkeypatch):
    source, destination = roots(tmp_path)
    (source / 'data.bin').write_bytes(b'original')
    resolve = Path.resolve

    def extended(self, *args, **kwargs):
        value = resolve(self, *args, **kwargs)
        return Path('\\\\?\\' + str(value)) if self.name == 'data.bin' else value

    monkeypatch.setattr(Path, 'resolve', extended)
    record = copy_verified(source, destination, 'data.bin')
    assert record['sha256'] == hashlib.sha256(b'original').hexdigest()
    assert (destination / 'data.bin').read_bytes() == b'original'


def test_unreadable_directory_cannot_silently_pass_inventory(tmp_path, monkeypatch):
    def denied(*_, onerror, **__):
        onerror(PermissionError('unreadable data folder'))
        return iter(())

    monkeypatch.setattr(os, 'walk', denied)
    with pytest.raises(PermissionError, match='unreadable'):
        inventory(tmp_path)
