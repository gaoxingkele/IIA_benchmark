# Experiments

No paper-score reproduction is asserted by collection generation alone.

## Official-v3 author and split-control grids (P2)

- Both official datasets, four classifiers, five folds, 30 jackknife+ alpha values, and all 10–60 minute prefixes were executed.
- The original v3 overlap lane completes 40/40 tasks but closes only 6/16 paper rows under the frozen joint tolerances.
- The paper-text disjoint lane closes 7/16; adding a global NumPy seed gives 7/16 overlap and 8/16 disjoint.
- Across the Python-seeded pair, AFC/train/calibration/test indices are identical and CP-calibration/RF overlap changes from 250 or 500 samples to zero. Test bifurcations are identical for 30/40 paired tasks; all 10 remaining mismatches are CASIM, whose vendored MultiRocket transformer calls Numba `np.random` without receiving the declared model random state.
- A CASIM-only lane resets both Python-level NumPy and Numba RNG state immediately before each fit. All 10 TEP/synthetic same-fold pairs then have identical test bifurcations and maximum absolute pair delta zero, so the remaining metric changes can be attributed to the RF training subset under the single-process `n_jobs=1` condition.
- Evidence: `experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json` and `docs/reports/p0_bip_paper_grid_complete_2026-08-31.md`.
- Boundary: the split correction improves some rows but does not explain the full paper gap; Docker file-order/data parity and independent same-fold execution remain open.

## SOTA Wave 2 next-reduction forecasting (E2/P1)

- ConE set trajectories on grouped TEP/NPP/FCC splits generate observable next-contraction targets for all three seeds.
- Mean jackknife+ RF MAE versus the median-time parent is 12.5743/16.0377, 3.7065/4.0506, and 0.8197/2.3580 minutes.
- Interval coverage is 0.9032/0.8666/0.9230; NPP passes only 2/3 nominal-coverage gates. Paired MAE credit passes 7/9 dataset-seeds.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: 30 trees and at most 40 jackknife rows are a bounded independent transfer validation and are not interchangeable with the now-complete original-domain v3 grids.
