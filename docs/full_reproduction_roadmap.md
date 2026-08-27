# Full reproduction roadmap

The target is the complete algorithm inventory in `configs/algorithms/book_algorithms.json` plus a versioned SOTA layer. “Complete” has four independently auditable gates:

1. **Equation closure**: every book equation, pseudocode branch, input assumption, and tuning rule used by an algorithm is mapped to code and tests.
2. **Engineering closure**: model config, dataset adapter, grouped split, deterministic seeds, dependency lock, and runnable CLI exist.
3. **Evidence closure**: the literature ARA records source hash, extracted claim, implementation mapping, validation command, and observed result.
4. **Score closure**: the paper's representative result is reproduced on the same data/protocol or explicitly marked blocked with the missing artifact.

## Current baseline after inventory

- 20 book deliverables are registered: 19 algorithmic methods plus the Chapter 6 visual-analytics verification suite.
- Existing callable code now covers 13 entries, but all 13 remain `partial` until their remaining paper-data/score evidence is closed.
- The other seven entries are `missing`; therefore the repository does **not** yet claim that all book algorithms have landed.
- SOTA sources are tracked separately from book methods. A DOI or downloaded PDF is evidence acquisition, not an implementation or score reproduction.

## Planned implementation batches

| Batch | Scope | Exit condition |
|---|---|---|
| B1 | Chapter 2 exact IID metrics, non-IID segmentation/PMFs/Bayesian intervals, deadband, APP | equation tests and synthetic recovery tests pass |
| B2 | Chapter 3 convex/non-convex NOZ, variation direction, pump filter, condenser model | membership/filter/physics invariant tests pass |
| B3 | Chapter 4 NTE/NDTE, IGTE/IGDTE, recursive BN, PLR root cause | causal graph and root-ranking recovery tests pass |
| B4 | Chapter 5 criterion C, accelerated alignment, CHARM, maximum entropy | sequence/pattern/prediction regression tests pass |
| B5 | CASIM and ConE-AFC | official artifact version acquired, environment locked, paper protocol locally validated |
| B6 | 2025-2026 robustness/early-classification layer | corruption matrix, prefix metrics, calibration coverage, and mixed-perturbation report pass |
| B7 | Chapter 6 evidence dashboard | every benchmark run exports traceable tables/figures without changing metric semantics |

The roadmap is intentionally strict: an algorithm changes from `partial`/`missing` to `verified` only in the same commit that adds its implementation, tests, config, citations, and ARA validation record.
