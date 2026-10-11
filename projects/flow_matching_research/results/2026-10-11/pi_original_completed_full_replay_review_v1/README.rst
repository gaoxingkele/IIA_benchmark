Pi original PSM seed42 full replay completion
===========================================

The actual original historical mirror model(1,930,289parameters,batch256,3completeepochs,1,554optimizer minibatches) was replayed on all13,238,200overlapping training-score pointsand87,800originally covered test points. All15streams are bitwise identical to saved original outputs. An independent guarded reviewer reread all original/replay arrays and recalculated the pooled threshold, interval point adjustment, confusion counts and metrics. Both actual terminal receipts are exit0 and not resource-aborted.

PA F1:0.9803681732004159. Same source threshold without PA F1:0.0223795025898541. Pointwise AUROC:0.481281323941512;AP:0.2726305082596521. Test-based early stopping and pooled training/test calibration remain;41test-tail points remain omitted. These are local historical mirror source results, not certified journal or strictTAB equivalence. Only one of six original training seeds completed here; retries and replay are not extra seeds.

v42 appends8metrics without changing any previous8,618rows or capture times. There are 8,626metric rows,2,128measured and6,498pending. No seed STD or CI is claimed. All45paper/681method-ablation/273original-data-task obligations remain.48relevant tests passed.
