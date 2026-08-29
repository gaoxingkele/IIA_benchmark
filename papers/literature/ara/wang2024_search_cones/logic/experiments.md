# Experiments

Frozen command: `python experiments/paper_harness/chapter3_multidataset/experiment.py --out_dir experiments/paper_harness/chapter3_multidataset/run_1`.

The implementation uses the Eq. 3.18 spherical-coordinate ranges and exact occupied-cone lookup before nearest-cone fallback. Three-seed F1 is 0.6385/0.6661/0.0939 on TEP/PRONTO/SKAB; FAR is 0.5385/0.5600/0.9862. The CSTR input sequence, exact alpha-knee result of 1.4 degrees, t=1395 detection, and source paper data remain unavailable, so no paper-score credit is claimed.
