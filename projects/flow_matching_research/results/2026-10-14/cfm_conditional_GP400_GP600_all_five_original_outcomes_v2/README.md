CFM-TS / Appendix D Sine / conditional Gaussian GP / all five original400-versus600 outcomes

All ten original full-training attempts and all400000 original grid coordinates were audited: nine complete finite native/independent model replays and one complete once-observed numerical failure. The original five paired joint data/training seeds1103..1107 remain present. This closes attempt-outcome accounting; it does not claim ten successful finite score certificates.

| Joint seed | 400-epoch MSE | 600-epoch MSE | 600 outcome |
| ---: | ---: | ---: | --- |
| 1103 | 0.00831553280216 | 0.00132490862803 | complete_finite_full_grid |
| 1104 | 0.00189529249638 | undefined | full_once_observed_numerical_failure |
| 1105 | 0.00180871592798 | 0.00577539099228 | complete_finite_full_grid |
| 1106 | 0.000464727988364 | 0.000290469536904 | complete_finite_full_grid |
| 1107 | 0.00508536131372 | 0.0189099336452 | complete_finite_full_grid |

400 epochs/250000 updates per model: full-five MSE mean 0.00351392610572, sample standard deviation 0.00317608883498; hierarchical joint-seed/whole-trajectory bootstrap95% interval [0.0011524126377398193, 0.00628632545537012],10000 resamples with frozen seed20261014.

600 epochs/375000 updates per model: four finite full80x500 grids and the failed seed1104 trajectory284. Full-five mean, standard deviation, bootstrap interval and paired mean difference remain undefined. No average or uncertainty interval over only the four surviving seeds is promoted as complete performance. Failure fraction1/5 describes this small frozen cohort, not a general population failure rate.

The original first400 epoch loss histories are bitwise identical within every pair. Intermediate model weights/optimizer states were not retained, so their equality and an isolated training-RNG effect are not certified. Successful pairs show both improvement and deterioration; one600 model fails. These observed outcomes do not establish a training root cause or an author-code bug.

The previously idle finite-only coordinator was handed off only after exact PID/creation/command/state/descendant checks. No numerical child was stopped or restarted; original training, solver, score tolerance and resource guards were unchanged. The new coordinator waits for all original outcomes and reports the preserved failure explicitly. Original v1 comparison registration and failed-model proof remain preserved.

This Appendix80-test cohort remains separate from Main20-test/93600-update recipes and the Main400 five-seed cohort. Original-author equivalence and strict TAB remain unproved. The full285 CFM routes and45-paper/681-method-record/273-data-task research scope remain required.

Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ
