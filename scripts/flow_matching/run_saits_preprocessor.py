"""Run an author preprocessor with an explicit, separately labelled window fix."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def corrected_window_truncate(features, seq_len, sliding_len=None):
    import numpy as np
    stride = seq_len if sliding_len is None else sliding_len
    if seq_len <= 0 or stride <= 0:
        raise ValueError('Window and stride must be positive')
    starts = range(0, len(features) - seq_len + 1, stride)
    windows = [features[start:start + seq_len] for start in starts]
    if not windows:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', required=True, choices=['gene_UCI_BeijingAirQuality_dataset.py', 'gene_UCI_electricity_dataset.py', 'gene_ETTm1_dataset.py'])
    parser.add_argument('--exact_masks', action='store_true')
    known, rest = parser.parse_known_args()
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/corrected'
    sys.path.insert(0, str(source))
    from dataset_generating_scripts import data_processing_utils
    data_processing_utils.window_truncate = corrected_window_truncate
    if known.exact_masks:
        def exact_mask(vector, rate):
            import numpy as np
            indices = np.flatnonzero(~np.isnan(vector))
            return np.random.choice(indices, int(len(indices) * rate), replace=False)
        data_processing_utils.random_mask = exact_mask
    script = source / 'dataset_generating_scripts' / known.script
    sys.argv = [str(script)] + rest
    runpy.run_path(str(script), run_name='__main__')
