MVTec-AD / Flow Mismatching / conditional / seed1103 / epoch700 passed full1725-image,15-category native scoring, checkpoint replay and independently recomputed image/pixel/region metrics. Heatmap and image-score replay maximum errors both0. Both actual guarded processes exited0 without resource abort.

| Metric (%) | Local epoch700 | Paper Table9 epoch700 | Difference (pp) |
| --- | ---: | ---: | ---: |
| image_AUROC | 86.360229 | 97.400000 | -11.039771 |
| image_AP | 92.994344 | 98.900000 | -5.905656 |
| image_F1_max | 90.072428 | 96.300000 | -6.227572 |
| pixel_AUROC | 87.401959 | 97.100000 | -9.698041 |
| pixel_AP | 37.132502 | 65.300000 | -28.167498 |
| pixel_F1_max | 40.516239 | 63.400000 | -22.883761 |
| pixel_AUPRO | 74.961874 | 91.200000 | -16.238126 |

The seven fully replayed checkpoints100..700 are one correlated training trajectory; no seven-seed mean or confidence interval is claimed. The original2000-epoch/five-seed training and all325 evaluations remain incomplete. F1-max is retrospective test-label oracle evaluation, not strict TAB. Local U-Net/training/preprocessing/inference precision/AUPRO/RNG choices and unavailable author settings remain fidelity gaps; score replay equality certifies the local model, not original-author equivalence. No original A100 FPS claim is certified.
