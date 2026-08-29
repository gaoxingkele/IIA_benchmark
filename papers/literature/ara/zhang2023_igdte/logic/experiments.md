# Experiments

The Chapter 4.2 harness executes the book-specified fuzzy triangular granules,
modified OPTICS reachability-peak clustering, trend-preserving labels,
second-order empirical IGTE/IGDTE, lag-derived window sizes, and 19 clustered
surrogates. The frozen run uses seeds 1103/2207/3301 on TEP IDV(1), PRONTO,
and SKAB.

This is a retained negative result. TEP activates two significant IGTE edges
per seed, but they have F1 0 against Book Table 4.8, V1 ranks second, and IGDTE
prunes no edge. PRONTO activates 0.6667 of episodes on average and is
seed-unstable (same-episode direct-edge Jaccard 0.3148). SKAB activates every
episode, but within-valve Jaccard is 0.0476. Across all 21 episode-by-seed
evaluations, no IGDTE-pruned edge occurs.

The exact 2023 paper full text/code, reachability-peak parameters, and exact TEP
realization remain unavailable. Therefore this evidence grants E2 engineering
credit to IGTE only and denies exact-paper, Table 4.8, and direct-causality
claims. Machine-readable evidence:
`experiments/reports/book_ch4_igte_igdte_multidataset_validation.json`.
