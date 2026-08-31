# Local validation

Status: **passed**.

- Command: `python -m pytest -q tests/test_cone_afc.py`
- Scope: Eqs. 1-5, Algorithms 1-2, probability/distance and expanding-window invariants.
- Author grid: 250/250 split-model tasks and 95/95 named Tables I-II means within tolerance on the official 18,750-sequence payload.
- Independent same-fold conformal layer: 50/50 MBW-LR folds and 1,800/1,800 coverage/set-size/singleton/empty-rate comparisons match exactly; maximum absolute delta 0.0. Docker and independent base-model/end-to-end gates remain open.
