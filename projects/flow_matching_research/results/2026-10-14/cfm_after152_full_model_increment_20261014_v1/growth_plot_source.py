"""Plot complete saved grids; no model inference or filtering of benchmark errors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT = cold.ROOT
folder = ROOT / 'projects/flow_matching_research/results/2026-10-14/cfm_after152_full_model_increment_20261014_v1'
snapshot = cold.original.read(folder / 'actual_current_deduplicated_complete_original_models.json')
records = {r['original_canonical_id']: r for r in snapshot['deduplicated_full_records']}
curves = []
reference = None
for epochs in (400, 600):
    identity = 'cfm_ts_original__main_text__gp_gaussian_conditional_e%d__sine__seed1106' % epochs
    record = records[identity]
    assert cold.streaming_sha(record['replay_arrays']) == record['replay_arrays_sha256']
    with np.load(record['replay_arrays'], allow_pickle=False) as arrays:
        position = int(np.flatnonzero(arrays['test_ids'] == 58)[0])
        times = arrays['times'][position].copy()
        truth = arrays['truth'][position, :, 0].copy()
        prediction = arrays['predictions'][position, :, 0].astype(np.float64)
    assert len(times) == len(truth) == len(prediction) == 500 and np.isfinite(prediction).all()
    if reference is None:
        reference = times, truth
    else:
        assert np.array_equal(reference[0], times) and np.array_equal(reference[1], truth)
    curves.append((epochs, prediction))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
colors = {400: '#2563eb', 600: '#d97706'}
axes[0].plot(times, truth, color='#111827', lw=1.7, label='Original truth')
for epochs, prediction in curves:
    axes[0].plot(times, prediction, color=colors[epochs], lw=1.4, label='%d epochs' % epochs)
    error = np.abs(prediction - truth)
    assert (error > 0).all()
    axes[1].plot(times, error, color=colors[epochs], lw=1.4, label='%d epochs' % epochs)
axes[0].set_ylim(-4.2, -2.5)
axes[0].set_title('State view: axis zoom only')
axes[0].set_ylabel('State')
axes[1].set_yscale('log')
axes[1].set_title('Absolute error: all 500 saved points')
axes[1].set_ylabel('Absolute error (log scale)')
for ax in axes:
    ax.set_xlabel('Original time')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
axes[1].axvline(1.8874931267797401, color='#9ca3af', lw=.9, ls='--')
fig.suptitle('Main-text Sine / seed 1106 / trajectory 58\nIdentical full test grids; benchmark MSE retains this trajectory')
for extension in ('png', 'svg'):
    destination = folder / ('main_GP400_GP600_seed1106_saved_growth.' + extension)
    assert not destination.exists()
    fig.savefig(destination, dpi=180)
    if extension == 'svg':
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
plt.close(fig)
(folder / 'growth_plot_source.py').write_bytes(Path(__file__).read_bytes())
print('Saved complete-grid diagnostic PNG and SVG; no benchmark points removed.')
