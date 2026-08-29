# Semantic gene bank

This bank records reusable WHERE × WHY lessons. It is not a leaderboard and it
does not convert a failed gate into method credit.

| Gene | WHERE (activation context) | WHY (causal rationale) | Evidence | Decision |
|---|---|---|---|---|
| IIA-G001 event-fingerprint payload | CTFH/HDAM and related flood classifiers require sparse alarm activations, within-flood event order, and class-aligned episodes. | Persistent process-state windows suppress newly appearing events and destroy discriminative hashes/templates. | PRONTO CTFH has zero consensus hashes in all four classes; HDAM balanced accuracy is 0.1010. | Reuse TEP Alarm/FCC/NPP event adapters; keep PRONTO only as an M1 mismatch sentinel. |
| IIA-G002 coverage needs efficiency | Conformal AFC results are meaningful only when coverage is paired with set size, singleton rate, and prefix behavior. | A prediction set containing every class obtains coverage 1.0 without resolving uncertainty. | PRONTO ConE and Cross-Conformal have coverage 1.0, set size 4.0, singleton rate 0.0 at every prefix. | G1 fails if set-size efficiency is degenerate, even when coverage passes. |
| IIA-G003 task-level match precedes tuning | Apply before any algorithm/dataset sweep. | Hyperparameter search cannot repair absent labels, incompatible event semantics, or leakage across windows from one run. | Current T4 proxy failures and zero leaderboard-eligible splits. | Require M2/M3 admission and G0 before tuning; M1 receives diagnostic-only credit. |

Future entries must include a falsification test and point to an immutable gate-ledger
row. Genes may guide a new protocol, but they may not retroactively change a frozen
held-out evaluation.
