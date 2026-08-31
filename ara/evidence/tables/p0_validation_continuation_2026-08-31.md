# P0 Validation Continuation Evidence - 2026-08-31

| Evidence | Observed | Source |
|---|---:|---|
| BiP total fold tasks after RNG control | 180 / 180 | `experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json` |
| CASIM Numba-controlled equal/unequal test-bifurcation pairs | 10 / 0 | `experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json` |
| CASIM maximum paired test-bifurcation delta | 0 | `experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json` |
| Independent ConE MBW-LR folds | 50 / 50 | `experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/summary.json` |
| Independent ConE paired metric rows | 1,800 / 1,800 exact | `experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/summary.json` |
| Independent ConE maximum absolute delta | 0.0 | `experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/summary.json` |
| Full test suite | 133 passed | `docs/reports/p0_paper_exact_checkpoint_2026-08-31.md` |

The host has no Docker CLI. The independent ConE result covers the conformal layer with frozen author MBW-LR scores; it does not close independent base-classifier, end-to-end wrapper, or archived-container gates.
