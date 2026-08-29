# Experiments

The Chapter 2 harness reproduces Xu 2012 Examples 1-2 with 500-repetition
stationary Monte Carlo checks for seeds 1103/2207/3301. Every run passes the
frozen FAR/MAR tolerance 0.005 and AAD tolerance 0.15. Industrial Table VII is
recomputed from the published base FAR 0.1486 and MAR 0.1204; all n=2/3/4 rows
match to maximum absolute error 5.94e-5, below the 1e-4 publication-rounding
tolerance.

The complete symmetric on/off delay and threshold-delay search are also run on
TEP, PRONTO, and SKAB. Mean F1 is 0.8678/0.3011/0.1448, showing strong negative
transfer outside TEP. The original steam-pressure payload is unavailable, so
only the named equation/table items receive E4/P2 credit. Machine-readable
evidence: `experiments/reports/book_ch2_multidataset_validation.json`.
