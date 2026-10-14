GRASP: 35 unique complete original models, 62 score recipe rows, 21 recipe groups

Every listed full1500-epoch model, original update budget and complete validation/test grid has guarded native replay and independent review; current result/checkpoint/score/proof hashes were rechecked. Retain all9120entity models,480training cohorts,10registered seeds and45/681/273scope. Scoring recipes can share a trained model; recipe rows and recipe groups are not independent training seeds or training cohorts.

| Profile | Entity | Sources / flow steps | Complete seeds / 10 | Oracle Best F1 mean | Validation-only F1 mean |
| --- | --- | --- | ---: | ---: | ---: |
| er | swan_sf | 5 / 10 | 3 / 10 | 0.535154 | 0.518959 |
| gnn | swan_sf | 5 / 10 | 3 / 10 | 0.464714 | 0.450930 |
| lin | swan_sf | 5 / 10 | 3 / 10 | 0.535939 | 0.522726 |
| main | swan_sf | 5 / 1 | 3 / 10 | 0.524373 | 0.515497 |
| main | swan_sf | 10 / 10 | 3 / 10 | 0.522897 | 0.507652 |
| main | swan_sf | 1 / 10 | 3 / 10 | 0.522878 | 0.507688 |
| main | swan_sf | 20 / 10 | 3 / 10 | 0.522846 | 0.507707 |
| main | swan_sf | 2 / 10 | 3 / 10 | 0.522810 | 0.507634 |
| main | swan_sf | 5 / 10 | 3 / 10 | 0.522947 | 0.507650 |
| main | swan_sf | 5 / 2 | 3 / 10 | 0.522050 | 0.514419 |
| main | swan_sf | 5 / 20 | 3 / 10 | 0.522773 | 0.507044 |
| main | swan_sf | 5 / 5 | 3 / 10 | 0.522770 | 0.509003 |
| main | swan_sf | 5 / 50 | 3 / 10 | 0.522582 | 0.506814 |
| mean | swan_sf | 5 / 10 | 3 / 10 | 0.526450 | 0.514823 |
| tau_0 | swan_sf | 5 / 10 | 3 / 10 | 0.533216 | 0.519308 |
| tau_0p25 | swan_sf | 5 / 10 | 2 / 10 | 0.526125 | 0.523179 |
| tau_0p5 | swan_sf | 5 / 10 | 3 / 10 | 0.533517 | 0.519277 |
| tau_1 | swan_sf | 5 / 10 | 3 / 10 | 0.542241 | 0.520007 |
| tau_4 | swan_sf | 5 / 10 | 3 / 10 | 0.508165 | 0.497481 |
| tau_8 | swan_sf | 5 / 10 | 3 / 10 | 0.490417 | 0.488875 |
| transformer | swan_sf | 5 / 10 | 3 / 10 | 0.436515 | 0.399533 |

Newly closed: tau8 / Swan / swan_sf / seed1105,1500 epochs/93000 updates,3935 validation and72000 test points. Original5-source/10-flow-step forward budgets: validation800 calls/194300 window evaluations; test14100 calls/3597550 evaluations. Independent oracle AP0.5064573954984611, ROC0.6716647336535052, BestF1 0.4933623245782048; validation-only precision0.3781620236951649, recall0.7050746268656717, F1 0.49228845352230094. No point adjustment; validation controls use no test features or labels for calibration.

Native CPU/GPU score comparisons pass the unchanged rtol0.0002/atol0.001 and metric tolerances. They are not bitwise: maximum validation absolute difference0.00390625, test absolute difference0.27062499994644895; test maximum relative difference8.942361475836056e-7. No tolerances were relaxed.

Current means and sample standard deviations describe only the complete registered seeds shown. Most profiles have3/10 seeds; tau0.25 has2/10. No full ten-seed performance or uncertainty claim. Oracle test-label threshold metrics and validation-only controls remain separate; neither certifies author equivalence or strict TAB.

Legacy union entries retain actual complete per-model/per-recipe records. A shared native container report that omits a model is not treated as its direct replay proof. The catalog resolves the preserved per-model record and hashes its independently verified arrays explicitly. The previous34-model v1 catalog is preserved.
