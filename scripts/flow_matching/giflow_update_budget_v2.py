"""Validate native updates against the observed loader, retaining its drop_last."""
import math


def validate_observed_updates(epochs, maximum_epochs, updates, loader, per_epoch):
    samples, batch = loader['samples'], loader['batch_size']
    expected = samples // batch if loader['drop_last'] else math.ceil(samples / batch)
    if not 0 < epochs <= maximum_epochs or expected <= 0 or loader['batches'] != expected:
        raise ValueError('Original full GiFlow loader or epoch count differs')
    if updates != epochs * expected or per_epoch != {str(e):expected for e in range(epochs)}:
        raise ValueError('Observed full native GiFlow update count differs')
    return dict(updates_per_epoch=expected, dropped_tail_per_epoch=samples % batch if loader['drop_last'] else 0)
