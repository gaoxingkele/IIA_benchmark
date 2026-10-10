import hashlib
import sys
from types import SimpleNamespace

import pytest

from scripts.flow_matching.audit_adopted_gan_native_v1 import install_midi_alias, verify_source


def test_verify_source_detects_changed_original_binary(tmp_path):
    payload = b'\0MIDI\xff\r\n'
    file = tmp_path / 'original.mid'
    file.write_bytes(payload)
    asset = dict(original_directory=str(tmp_path), expected_git_entries=[dict(path='original.mid', size=len(payload),
        sha=hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest())])
    assert len(verify_source(asset)) == 1
    file.write_bytes(payload.replace(b'MIDI', b'FAKE'))
    with pytest.raises(ValueError, match='Pinned original Git file differs'):
        verify_source(asset)


def test_midi_alias_preserves_same_actual_module(monkeypatch):
    actual = SimpleNamespace(**{key: object() for key in
        ('read_midifile', 'write_midifile', 'NoteOnEvent', 'NoteOffEvent', 'Pattern', 'Track', 'events')})
    monkeypatch.setitem(sys.modules, 'python3_midi', actual)
    monkeypatch.delitem(sys.modules, 'midi', raising=False)
    assert install_midi_alias() is actual
    assert sys.modules['midi'] is actual
    monkeypatch.delitem(sys.modules, 'midi')


def test_midi_alias_rejects_an_existing_different_implementation(monkeypatch):
    monkeypatch.setitem(sys.modules, 'python3_midi', object())
    monkeypatch.setitem(sys.modules, 'midi', object())
    with pytest.raises(ValueError, match='different loaded MIDI implementation'):
        install_midi_alias()
