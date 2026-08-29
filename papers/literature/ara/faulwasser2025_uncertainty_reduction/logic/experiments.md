# Experiments

No paper-score reproduction is asserted by collection generation alone.

## SOTA Wave 2 next-reduction forecasting (E2/P1)

- ConE set trajectories on grouped TEP/NPP/FCC splits generate observable next-contraction targets for all three seeds.
- Mean jackknife+ RF MAE versus the median-time parent is 12.5743/16.0377, 3.7065/4.0506, and 0.8197/2.3580 minutes.
- Interval coverage is 0.9032/0.8666/0.9230; NPP passes only 2/3 nominal-coverage gates. Paired MAE credit passes 7/9 dataset-seeds.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: 30 trees and at most 40 jackknife rows are a bounded independent validation; official capsule, exact features/split, and Tables 1-4 remain open.
