GRASP tau=0 seed1104 complete native model replay
================================================

The original1500epoch/93000update/69244parameter checkpoint was restored and re-inferred on all 3935validation and72000test points, all5source samples and10flow evaluations per window. Actual native worker exit0, no resource guard abort. CPU/GPU scores are not bitwise identical, but their maximum relative differences are below2.3e-7 and pass the unchanged preregistered rtol2e-4/atol1e-3 bound. There are zero thresholded-prediction disagreements. Full arrays, row identities, forward counts and all metrics independently recomputed.

NoPA validation-only F1:0.5006212309463923; retrospective test-label oracle BestF1:0.512052098769959. Author/TAB equivalence remains uncertified; rawfeature38versus paper39 and declared local model/preprocessing/selection choices remain. No additional training seed is created by replay. This completes only this one original ablation/seed recipe.
