"""Author CFMI dataset adapter for config-prepared independent benchmarks."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class CampaignWindowDataset(Dataset):
    def __init__(self, root: str):
        self.root = root

    def setup(self, split, *, rng=None):
        with np.load(Path(self.root), allow_pickle=False) as data:
            selected = data[split]
            self.data = data['values'][selected]
            self.observed = data['observed'][selected]
            self.conditioning = data['conditioning'][selected]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.observed[index], self.conditioning[index]

    def __getitems__(self, index):
        return self[index]
