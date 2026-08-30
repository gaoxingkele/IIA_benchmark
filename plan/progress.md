# Progress

## 2026-08-30 — P0 Paper-Exact closure

- Acquired and SHA-256 verified the complete published CASIM v1, ConE-AFC v2,
  and BiP-AFC v3 Code Ocean Capsules, including code, data, environment, metadata,
  and licenses.
- Built isolated Python 3.9.25 environments with each Dockerfile's exact package
  versions under ignored `tmp/codeocean_envs/`.
- Docker is not installed on this host. Native Windows compatibility execution is
  therefore tracked separately from authoritative container execution.
- Completed unchanged CASIM capsule-default five-fold closed-set execution:
  mean balanced accuracy 0.993819, median 0.991071, range 0.989011–1.000000.
- Prior profiling found 308 unique trajectories among 310 CASIM samples. Two
  same-label duplicate pairs cross train/test in official folds 3–5 (four fold-group
  instances); retain author scores and add a separate grouped sensitivity result.
- Started unchanged ConE-AFC v2 author-default execution; BiP-AFC v3 is queued.
- Froze three Paper-Exact protocol cards, paper target tables, tolerances, known
  Capsule/paper mismatches, and separate transfer/paper-exact result paths.
- Identified a BiP v3 artifact defect: calibration and random-forest training use
  the same leading per-class indices. Preserve it in author-exact pass one and test
  a disjoint correction only in the independent implementation pass.
