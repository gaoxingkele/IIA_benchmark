# Repository working agreement

- Treat `configs/` as the source of truth; avoid hard-coded dataset paths in models.
- Preserve user-provided books and downloaded raw data; never overwrite them during preprocessing.
- Prefer `aria2c` through `http://127.0.0.1:17890` for registered downloads and verify checksums.
- New methods need a model config, a callable implementation or explicit reproduction status, tests, and a literature/data citation.
- New leaderboard results need grouped splits, frozen hyperparameters, data audit, random seeds, and uncertainty/robustness reporting.
- Synthetic smoke results validate plumbing only and must not be described as benchmark performance.
- Canonical remote: `https://github.com/gaoxingkele/IIA_benchmark.git`.
- After each coherent iteration, run the relevant validation suite, create a descriptive Git commit, and push `main` when the remote is reachable. Never force-push or overwrite remote history.
