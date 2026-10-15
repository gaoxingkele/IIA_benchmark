Full original generation models independently artifact-reviewed:30/70.

Added Diffusion-TS paper-distribution Sines seed42. The previous29 source/result/weight/generated/reference/proof hashes were rechecked and their catalog records are unchanged.

The new original model completed12,000 optimizer updates,24,000 training minibatches,10,000x24x5 reference rows and10,005x24x5 generated rows, with all four five-repeat evaluators. Native Py310 independently restored every model tensor and checked all EMA/optimizer tensors, full arrays and evaluator statistics. This audit does not reinfer generation or retrain learned evaluators.

| New paper-distribution seed42 metric | Five-evaluator mean | Within-model sample STD |
|---|---:|---:|
|context_fid|0.0782511851|0.00368490219|
|correlational|0.0059520975|0.000963503985|
|discriminative|0.0995|0.055691393|
|predictive|0.318334023|1.81083709e-06|

Paper-distribution Sines has1/5 training seeds;43..46 remain required. The released-Sines5/5 cohort remains separate. These are different frozen frequency/phase distributions; neither can replace the other or be pooled for a five-seed interval. Evaluator repetitions never count as training seeds.

Full30-model total:467,000 optimizer updates,934,000 training minibatches,600 evaluator runs. fMRI43 and all other uncompleted original70 slots remain required. Source parameter count235,249 differs from paper232,177; this is recorded, not silently altered. These are generative distribution-reconstruction metrics, not anomaly-detection F1 or strictTAB results. Original45 papers,681 method/baseline/ablation records,273 data tasks and702 paper scalar targets remain in scope.
