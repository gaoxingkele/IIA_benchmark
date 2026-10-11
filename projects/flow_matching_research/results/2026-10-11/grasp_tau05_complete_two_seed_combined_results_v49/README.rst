GRASP tau=0.5: complete native two-seed verification and reconciled results
========================================================================

Original seeds1103/1104 each completed1500epochs/93000updates with69244parameters. Both complete native model replays and independent full raw-label/row/array/metric audits passed under unchanged frozen tolerances. Each original5source/10flow recipe covers all3935validation and72000test points, 14900forward calls and3791850window visits. Both full native classifications have zero disagreements. Seed1104 postreview and the combined-publication worker are actually terminal exit0/noabort.

Validation-only no-PA F1mean=0.5239883159796608,sampleSTD=0.0008436443845896992,95percentCI=[0.5164084719162697,0.5315681600430519]. The threshold is the validation99percent quantile with strictgreater comparison; no test calibration. The separately reported test-label oracle BestF1 must retain its distinct protocol. This is only2/10original seeds, not complete ten-seed robustness.

The combined v49table advances only6existingtau0.5 rows over frozen v48. Every8620other rows/timestamps remains unchanged, including all4complete5/5Diffusion-TSStocks statistics. The whole table still contains 8626metric slots,2128measured and6498pending. Mean/sampleSTD/SE and unclippedStudent-t95CI were independently checked using original seed values and the closed-formdf1CauchyCDF.

54relevant tests passed before registration. Native38features versus paper39, paper-derived model and last-epoch choices remain explicit. No full author/TAB alignment is certified. All45papers/681method baseline-ablation/273original-data-task scope and all remaining original experiments stay active.
