"""Legacy MIDI velocity property bridge; preserve the complete v1 failed audit."""
from scripts.flow_matching import audit_adopted_gan_native_v1 as base

original_install_alias = base.install_midi_alias


def install_legacy_velocity_property():
    midi = original_install_alias()
    # The Python 3 port retains the exact data[1] getter/setter, but removed
    # the Python 2 library's velocity property used by the author loader.
    if not hasattr(midi.NoteEvent, 'velocity'):
        midi.NoteEvent.velocity = property(midi.NoteEvent.get_velocity, midi.NoteEvent.set_velocity)
    event = midi.NoteOnEvent(data=[60, 91])
    if event.velocity != event.get_velocity() or event.velocity != event.data[1]:
        raise ValueError('Legacy MIDI attribute must preserve the exact original event byte')
    event.velocity = 37
    if event.get_velocity() != 37 or event.data != [60, 37]:
        raise ValueError('Legacy MIDI setter must change only the original velocity byte')
    return midi


def main():
    base.install_midi_alias = install_legacy_velocity_property
    base.main()


if __name__ == '__main__':
    main()
