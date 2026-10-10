import os
from types import SimpleNamespace

import pytest

from scripts.flow_matching import audit_adopted_gan_native_v2 as midi_bridge
from scripts.flow_matching import audit_rcgan_mnist_native_v2 as cwd_bridge


def test_velocity_bridge_reads_and_writes_only_existing_event_velocity_byte(monkeypatch):
    class NoteEvent:
        def __init__(self, data):
            self.data = data

        def get_velocity(self):
            return self.data[1]

        def set_velocity(self, value):
            self.data[1] = value

    midi = SimpleNamespace(NoteEvent=NoteEvent, NoteOnEvent=NoteEvent)
    monkeypatch.setattr(midi_bridge, 'original_install_alias', lambda: midi)
    assert midi_bridge.install_legacy_velocity_property() is midi
    event = NoteEvent([84, 57])
    assert event.velocity == event.get_velocity() == 57
    event.velocity = 0
    assert event.data == [84, 0]


@pytest.mark.parametrize('fails', [False, True])
def test_native_audit_restores_cwd_for_report_writing_even_on_error(tmp_path, monkeypatch, fails):
    initial = os.getcwd()

    def original(*args):
        os.chdir(tmp_path)
        if fails:
            raise ValueError('Preserved native failure')
        return {'source_unchanged': True}

    monkeypatch.setattr(cwd_bridge.native, 'graph_audit', original)
    if fails:
        with pytest.raises(ValueError, match='Preserved native failure'):
            cwd_bridge.graph_audit(None, None, None)
    else:
        assert cwd_bridge.graph_audit(None, None, None) == {'source_unchanged': True}
    assert os.getcwd() == initial
