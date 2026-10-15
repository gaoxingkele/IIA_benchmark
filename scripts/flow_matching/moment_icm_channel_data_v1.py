"""Appendix-D channel groups and paired complete-epoch batch references.

No model dependency. Fixed ordered groups and seeded remainder replacement are
disclosed reconstruction choices; the paper does not publish those details.
The main-text subsampling description is a separate, unimplemented control.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def channel_groups(channels, seed, dataset_key, policy):
    import numpy as np
    integer(channels, 'Channel count')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Explicit uint32 seed required')
    if not isinstance(dataset_key, str) or not dataset_key:
        raise ValueError('Stable dataset identity required')
    if policy != 'appendix_D_partition8_remainder_replacement':
        raise ValueError('Explicit Appendix-D policy required; do not silently substitute main-text subsampling')
    if channels == 1:
        return ((0,),)
    key = hashlib.sha256((str(seed) + '\0' + dataset_key).encode('utf-8')).digest()
    rng = np.random.RandomState(int.from_bytes(key[:4], 'little'))
    groups = []
    for start in range(0, channels, 8):
        original = list(range(start, min(start + 8, channels)))
        group = original + rng.choice(original, 8 - len(original), replace=True).tolist()
        groups.append(tuple(int(v) for v in group))
    return tuple(groups)


class GroupedCases:
    """Read every case/group without flattening channels into unrelated cases.

    Values are the source-preprocessed arrays supplied by configuration. This
    reader does not fit a scaler, change dtype or read labels. Source normalized
    TEST remains a diagnostic source track, never a strict-train-only claim.
    """
    def __init__(self, split_directory, *, dataset_key, split, seed, policy):
        import numpy as np
        if split not in ('TRAIN', 'TEST'):
            raise ValueError('Explicit official split required')
        self.directory = Path(split_directory)
        self.dataset_key, self.split = dataset_key, split
        self.values = np.load(self.directory / 'values.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(self.directory / 'input_mask.npy', mmap_mode='r', allow_pickle=False)
        self.instance_rows = np.load(self.directory / 'original_instance_rows.npy', mmap_mode='r', allow_pickle=False)
        if self.values.ndim != 3 or self.values.shape[-1] != 256:
            self.close()
            raise ValueError('All original [case,channel,256] coordinates required')
        cases, channels, length = self.values.shape
        if self.mask.shape != (cases, length) or self.instance_rows.shape != (cases,):
            self.close()
            raise ValueError('Full case masks and source instance identities required')
        self.groups = channel_groups(channels, seed, dataset_key, policy)
        self.width = len(self.groups[0])
        self.cases, self.channels = cases, channels

    def __len__(self):
        return self.cases * len(self.groups)

    def item(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError('Complete case-group item index out of range')
        case, group = divmod(index, len(self.groups))
        indices = self.groups[group]
        return dict(values=self.values[case, list(indices), :].copy(), input_mask=self.mask[case].copy(),
            dataset_key=self.dataset_key, split=self.split, case_index=case,
            original_instance_row=int(self.instance_rows[case]), group_index=group, channel_indices=indices)

    def close(self):
        for name in ('values', 'mask', 'instance_rows'):
            value = getattr(self, name, None)
            mapping = getattr(value, '_mmap', None)
            if mapping is not None:
                mapping.close()


@dataclass(frozen=True)
class BatchReference:
    width: int
    items: tuple[tuple[int, int], ...]


def paired_epoch_batches(datasets, batch_size, seed):
    """One manifest shared by independent/ICM models, including partial batches.

    Only shape-compatible cases share a batch. Official splits cannot mix.
    RNG is independent of model initialization, masking and optimizer state.
    """
    import numpy as np
    integer(batch_size, 'Batch size')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Explicit uint32 epoch seed required')
    if not datasets or len({d.split for d in datasets}) != 1:
        raise ValueError('One nonempty collection of a single official split required')
    if len({d.dataset_key for d in datasets}) != len(datasets):
        raise ValueError('Dataset identities must be unique within the split')
    buckets = {}
    for source_index, dataset in enumerate(datasets):
        if dataset.width not in (1, 8):
            raise ValueError('Appendix-D batches have exactly1or8channels')
        buckets.setdefault(dataset.width, []).extend((source_index, i) for i in range(len(dataset)))
    rng = np.random.RandomState(seed)
    batches = []
    for width in sorted(buckets):
        items = buckets[width]
        order = rng.permutation(len(items))
        for start in range(0, len(order), batch_size):
            batches.append(BatchReference(width, tuple(items[int(i)] for i in order[start:start + batch_size])))
    rng.shuffle(batches)
    return tuple(batches)


def materialize_batch(datasets, reference):
    import numpy as np
    examples = [datasets[source].item(item) for source, item in reference.items]
    if not examples or any(example['values'].shape != (reference.width, 256) for example in examples):
        raise ValueError('Every paired item must share the declared channel width')
    return dict(values=np.stack([e['values'] for e in examples]),
        input_mask=np.stack([e['input_mask'] for e in examples]),
        provenance=[{k: e[k] for k in ['dataset_key', 'split', 'case_index', 'original_instance_row',
                                      'group_index', 'channel_indices']} for e in examples])
