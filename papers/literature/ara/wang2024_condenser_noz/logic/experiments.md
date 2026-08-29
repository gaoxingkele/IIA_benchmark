# Experiments

Frozen command: `python experiments/paper_harness/chapter3_multidataset/experiment.py --out_dir experiments/paper_harness/chapter3_multidataset/run_1`.

Equation (3.90) now returns kPa and Eq. (3.96)/(3.100) retains the source-checked d2^(9/4) term. Table 3.5 parameters generate equation-defined operating samples; three bounded fits have minimum goodness 0.9999881 and synthetic pressure-bias F1 0.8694. Per-model 99% Beta-binomial FAR/MAR intervals execute, but the mean FAR upper bound is 0.3500 versus the book target near 0.075. Thus physical fit passes while zone-performance closure fails. No real dataset has verified Pc/Dc/T1/T2 semantics. The 300-MW daily data, 100 parameter sets, V1/V2 ensemble worst-case bounds, Tables 3.6-3.7 and alarm times remain blocked.
