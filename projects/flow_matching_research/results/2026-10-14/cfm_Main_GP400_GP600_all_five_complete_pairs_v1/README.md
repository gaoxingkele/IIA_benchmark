CFM-TS / Main-text Sine / conditional Gaussian GP / all five400-versus600 pairs

All ten original complete models passed guarded native replay and independent full20x500-grid review. Paired trajectory IDs, truth and times match exactly; only the registered400/62400 versus600/93600 budget differs. Seed1106 uses its previously verified saved original400 weights without new fitting.

| Joint seed | 400 MSE | 600 MSE |
| ---: | ---: | ---: |
| 1103 | 0.000821472936943 | 0.042202975672 |
| 1104 | 0.0147811041922 | 0.00107519589506 |
| 1105 | 0.00160283207374 | 0.00117822402586 |
| 1106 | 0.0103827486166 | 8.65403332009e+54 |
| 1107 | 0.00612070813879 | 0.00825064120842 |

Full-five400 mean 0.00674177319167, sample standard deviation 0.00591041321951. Full-five600 mean 1.73080666402e+54, sample standard deviation 3.87020135665e+54.

Paired600-minus400 mean difference 1.73080666402e+54; hierarchical paired joint-seed/whole-trajectory bootstrap95% interval [-0.006233636851949108, 6.923226656070526e+54] (10000 resamples, frozen seed20261014).

The catastrophic but finite600 seed1106 error remains in the mean and bootstrap. A lower bootstrap endpoint near ordinary errors can occur when resampling omits that seed or its dominant trajectory; this is not removal from the observed benchmark. Only five outer groups are available, so the interval is descriptive and does not establish a population-wide training effect or an author-code bug. No outlier-trimmed mean is published.

Main20-test recipes stay separate from Appendix80-test recipes. All original first400 loss histories are compared; intermediate weights and optimizer state equivalence are not certified. Original-author equivalence and strict TAB remain unproved. The broader285 routes and45/681/273research scope remain required.

Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ
