MVTec-AD / Flow Mismatching / conditional / seed1103 / epoch800.

Complete1725images,15categories and native128x128 heatmaps. Full checkpoint/state/probe replay passed with zero maximum heatmap and image-score error. All image/pixel metrics and analytic independent region-AUPRO were recomputed. Both guarded processes exited0 without resource abort.

| Metric (%) | Local epoch800 | Paper Table9 epoch800 | Difference (pp) |
|---|---:|---:|---:|
| image_AUROC | 86.601571 | 97.800000 | -11.198429 |
| image_AP | 93.183227 | 99.200000 | -6.016773 |
| image_F1_max | 90.309673 | 96.800000 | -6.490327 |
| pixel_AUROC | 87.454019 | 97.400000 | -9.945981 |
| pixel_AP | 37.310107 | 66.000000 | -28.689893 |
| pixel_F1_max | 40.612116 | 63.700000 | -23.087884 |
| pixel_AUPRO | 75.186142 | 91.400000 | -16.213858 |

All seven previously published100..700 checkpoint records were revalidated and remain exactly unchanged. The eight checkpoints belong to one correlated training trajectory; no eight-seed mean or confidence interval is claimed.

Original2000epochs/five seeds and all325evaluations remain incomplete. F1-max selects a threshold from test labels and is not strict TAB calibration. Local U-Net/optimizer/preprocessing/precision/AUPRO/RNG choices and unavailable author settings remain fidelity gaps. Replay equality verifies the local saved model, not author equivalence. Original A100 FPS is unverified.

Paper: https://arxiv.org/abs/2605.23070
