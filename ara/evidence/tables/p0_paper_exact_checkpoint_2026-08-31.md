# P0 Paper-Exact Evidence — 2026-08-31

| Evidence | Observed | Source |
|---|---:|---|
| CASIM completed model fits | 700 / 700 | `paper_harness/paper_exact/status.v1.json` |
| CASIM frozen numeric gates passed | 1 / 3 | `docs/reports/p0_casim_paper_grid_complete_2026-08-30.md` |
| CASIM full-threshold mean BA delta | +0.024267 | `docs/reports/p0_casim_paper_grid_complete_2026-08-30.md` |
| ConE completed model-split tasks | 250 / 250 | `paper_harness/paper_exact/status.v1.json` |
| ConE named paper means within tolerance | 95 / 95 | `experiments/paper_harness/p0_paper_exact/run_2/paper_grid/summary.json` |
| ConE maximum absolute mean delta | 0.002271006535947473 | `paper_harness/paper_exact/status.v1.json` |
| BiP completed lane/dataset/model/fold tasks | 160 / 160 | `paper_harness/paper_exact/status.v1.json` |
| BiP paper rows passed by lane | 6 / 16; 7 / 16; 7 / 16; 8 / 16 | `experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json` |
| BiP controlled equal/unequal test-bifurcation pairs | 30 / 10 | `paper_harness/paper_exact/status.v1.json` |
| Full test suite | 131 passed | `docs/reports/p0_paper_exact_checkpoint_2026-08-31.md` |

All three Capsule archive/code/data manifests passed full SHA-256 verification. The three completed author-code grids remain P2 because the Docker and independent same-fold gates are still open; CASIM and BiP also retain frozen numeric discrepancies.
