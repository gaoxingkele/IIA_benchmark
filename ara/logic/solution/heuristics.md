# Heuristics

## H01: Persist long paper reproductions at the finest reliable completion boundary
- **Rationale**: Task- or split-level checkpoints can be resumed without relabeling partial outputs as complete results; outputs from a monolithic author script stay in a separate checkpoint lane until the full run succeeds.
- **Sources**: [experiments/paper_harness/p0_paper_exact/checkpoint_2026-08-30.json]
- **Status**: active
- **Provenance**: ai-suggested
- **Sensitivity**: unknown
- **Code ref**: [experiments/paper_harness/p0_paper_exact/paper_grid.py, scripts/resume_p0_checkpoint.ps1]

## H02: Reconstruct the omitted CASIM outer loop while reusing the published inner splitter
- **Rationale**: Relabel the held-out known class as `-1`, then invoke the author open-set splitter so the class is absent from training and each novel sample is assigned to one test fold; keep author confirmation as an unresolved protocol boundary.
- **Sources**: [`-1` ← paper_harness/paper_exact/faulwasser2024_casim.v1.json:49 «"paper_grid_reconstruction": "for each known class, relabel it -1 and invoke the published get_train_test(..., open_set=True); the author utility removes all -1 rows from training and assigns each novel row to exactly one test fold"» [input]]
- **Status**: active
- **Provenance**: ai-suggested
- **Sensitivity**: unknown
- **Code ref**: [experiments/paper_harness/p0_paper_exact/paper_grid.py:94]
