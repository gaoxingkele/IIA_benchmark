# Method

## Harness formalisation

For a dataset `D` with train values `X_train` and test values `X_test`:

1. **Standardisation**: `mu, sigma = mean_std(X_train)`; every split is mapped by
   the same statistics. Test statistics never enter the scaler.
2. **Window grid**: `starts = arange(0, len(X) - w + 1, s)` with `(w, s)` from the
   dataset config; `s = 100` for SMD and `s = 1` for the other four, matching the
   reference loaders.
3. **Detector**: either instantaneous (`f: R^d -> R`) or windowed
   (`f: R^{w x d} -> R` or `-> R^w`).
4. **Layout**: `window_flatten` maps a window score to `w` consecutive entries and
   a per-position score to its position; `last_point` maps a window score to its
   final timestamp and averages overlapping votes.
5. **Threshold**: `percentile_combined` uses the `(100 - anomaly_ratio)` percentile
   of the pooled train and test energies; `best_f1` searches the threshold that
   maximises point-wise F1 on the scored test series.
6. **Metrics**: point-wise P/R/F1; point-adjusted P/R/F1; range-based
   (`alpha = 0`, flat bias) P/R/F1. All three are reported for every run so that
   no single family can be quoted in isolation.

## Protocol matrix

| protocol | layout | threshold | point adjustment |
|---|---|---|---|
| `tslib_reference` | window_flatten | percentile_combined | yes |
| `best_f1_adjusted` | window_flatten | best_f1 (oracle) | yes |
| `best_f1_pointwise` | window_flatten | best_f1 (oracle) | no |
| `per_point_adjusted` | last_point | percentile_combined | yes |

The matrix is the experiment: the same detector object is scored once and then
evaluated four ways, so differences between rows are attributable to the protocol
alone.

## Detectors

- **pca** - reconstruction error of the leading principal subspace.
- **mahalanobis** - squared Mahalanobis distance with ridge-regularised covariance.
- **knn** - distance to the k-th nearest training neighbour over a subsampled memory.
- **isolation_forest** - negated isolation-forest score.
- **ocsvm** - negated one-class SVM decision function.
- **usad** - shared encoder, two decoders, two-phase adversarial training (reconstructed).
- **anomaly_transformer** - association discrepancy with a Gaussian prior (transcribed).

## Reproducibility contract

Each run record carries the dataset shapes, the anomaly rate, repaired-value
counts, the window grid, the detector parameters, the protocol, the seed, the fit
and wall time, the git revision and the Python version. Payload bytes are pinned
by SHA-256 in `configs/datasets/public_sources.json`.
