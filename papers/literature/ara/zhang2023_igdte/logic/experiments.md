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

## Distinct-lag mechanism closure and iMAKS falsification

The gap-closure harness adds separate source-to-target and mediator-to-target
lags. On three generated delayed chains, pairwise IGTE is
`0.3935 ± 0.0070`, above the mean surrogate threshold `0.00551`, while IGDTE
falls to zero after conditioning on the mediator. This closes the defining
indirect-edge-pruning mechanism but grants E1 controlled evidence only.

The same frozen diagnostic is also applied to the registered synthetic iMAKS
edge `ST02_SEALING_CUR -> ST04_PACKAGING_SPD` with its documented 90-minute
delay. IGTE is `0.05805`, below the three-seed mean threshold `0.08598`, so the
edge is not detected in any run. Together with 0/21 pruning events on acquired
TEP/PRONTO/SKAB, this prevents E2 promotion. Evidence:
`experiments/reports/book_ch4_gap_closure_validation.json`.
