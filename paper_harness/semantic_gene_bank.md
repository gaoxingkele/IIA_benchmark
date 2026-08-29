# Semantic gene bank

This bank records reusable WHERE × WHY lessons. It is not a leaderboard and it
does not convert a failed gate into method credit.

| Gene | WHERE (activation context) | WHY (causal rationale) | Evidence | Decision |
|---|---|---|---|---|
| IIA-G001 event-fingerprint payload | CTFH/HDAM and related flood classifiers require sparse alarm activations, within-flood event order, and class-aligned episodes. | Persistent process-state windows suppress newly appearing events and destroy discriminative hashes/templates. | PRONTO CTFH has zero consensus hashes in all four classes; HDAM balanced accuracy is 0.1010. | Reuse TEP Alarm/FCC/NPP event adapters; keep PRONTO only as an M1 mismatch sentinel. |
| IIA-G002 coverage needs efficiency | Conformal AFC results are meaningful only when coverage is paired with set size, singleton rate, and prefix behavior. | A prediction set containing every class obtains coverage 1.0 without resolving uncertainty. | PRONTO ConE and Cross-Conformal have coverage 1.0, set size 4.0, singleton rate 0.0 at every prefix. | G1 fails if set-size efficiency is degenerate, even when coverage passes. |
| IIA-G003 task-level match precedes tuning | Apply before any algorithm/dataset sweep. | Hyperparameter search cannot repair absent labels, incompatible event semantics, or leakage across windows from one run. | Current T4 proxy failures and zero leaderboard-eligible splits. | Require M2/M3 admission and G0 before tuning; M1 receives diagnostic-only credit. |
| IIA-G004 retain state/edge representation pairs | Alarm-state datasets used by event-fingerprint or template methods. | State duration can carry class information even when the method narrative emphasizes new activations; edge conversion may discard discriminative persistence. | FCC CTFH state BA 0.5125 versus rising-edge BA 0.3656 on the same complete-run split. | Register state versus rising-edge as a paired ablation; do not force one representation globally. |
| IIA-G005 task-matched data can reverse apparent method failure | Apply when a method fails G1 on a representation-mismatched proxy but passes data integrity checks. | A negative proxy result may diagnose absent event/template structure rather than broken code. | HDAM rises from PRONTO BA 0.1010 to FCC BA 0.99375; CTFH moves from zero to 62 consensus hashes. | Keep both results, re-test on a second task-matched dataset, and deny paper-score credit until P2/P3 closure. |
| IIA-G006 enumerate closures, not every frequent subset | Dense alarm transactions with CHARM/closed-pattern mining. | Generating all frequent subsets before closure filtering pays exponential cost for patterns that must later be discarded; each distinct vertical TID intersection already determines one closure. | Parent exceeded 4 min and 754 MB observed memory; direct TID-closure enumeration finished the unchanged FCC run in 1.54 s and passed a brute-force equivalence test. | Replace the runtime cell while preserving the frozen mathematical output; retain the failed parent record. |

Future entries must include a falsification test and point to an immutable gate-ledger
row. Genes may guide a new protocol, but they may not retroactively change a frozen
held-out evaluation.
