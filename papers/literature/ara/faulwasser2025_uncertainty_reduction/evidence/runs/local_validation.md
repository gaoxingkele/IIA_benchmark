# Local validation

Status: **passed**.

- Command: `python -m pytest -q tests/test_uncertainty_reduction.py`
- Scope: bifurcation targets, explicit leave-one-out jackknife+ intervals and delay-timer invariants.
- Author-grid command: `.\tmp\codeocean_envs\bip\Scripts\python.exe experiments\paper_harness\p0_paper_exact\bip_grid.py run-bip --workers 8 --lanes author_overlap paper_disjoint seeded_author_overlap seeded_paper_disjoint`.
- Author-grid evidence: 160/160 fold tasks complete; Tables 1-4 pass 6/16 in the original v3 lane and 8/16 in the seeded disjoint lane. Numeric and CASIM Numba-RNG gaps remain open rather than being relabeled as success.
