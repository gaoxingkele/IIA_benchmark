# Experiments

Frozen command: `python experiments/paper_harness/chapter3_multidataset/experiment.py --out_dir experiments/paper_harness/chapter3_multidataset/run_1`.

Book Figure 3.2 reproduces `eta=9/13`, and the outside-point dynamic-bound path now executes the Eq. 3.15 closest-normal projection. Across three seeds and nine TEP/PRONTO/SKAB episodes, convex NOZ F1 is 0.7210/0.6766/0.0950. It does not beat Mahalanobis on TEP and has SKAB FAR 0.9744 under severe independent-normal drift. This is E3 named-item plus negative transfer, not the original industrial score.
