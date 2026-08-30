# P0 Paper-Exact Evidence — 2026-08-30

| Evidence | Observed | Source |
|---|---:|---|
| CASIM author-default mean balanced accuracy | 0.9938186813186813 | `experiments/paper_harness/p0_paper_exact/run_1/final_info.json:26` |
| CASIM duplicate-excluded post-hoc mean balanced accuracy | 0.9934981684981686 | `experiments/paper_harness/p0_paper_exact/run_1/final_info.json:72` |
| ConE default named rows within tolerance | 9 / 10 | `paper_harness/paper_exact/status.v1.json:42-43` |
| CASIM+ConE default mean coverage | 0.9121296732026144 | `experiments/paper_harness/p0_paper_exact/run_2/final_info.json:46` |
| CASIM+ConE default mean set size | 1.6361286274509803 | `experiments/paper_harness/p0_paper_exact/run_2/final_info.json:47` |
| EAC-1NN default set-size delta from paper | +0.024469803921568722 | `experiments/paper_harness/p0_paper_exact/run_2/final_info.json:181` |
| CASIM paused paper-grid tasks | 48 / 70 | `experiments/paper_harness/p0_paper_exact/checkpoint_2026-08-30.json:11-12` |
| ConE paused complete full-grid splits | 0 / 50 | `experiments/paper_harness/p0_paper_exact/checkpoint_2026-08-30.json:24-25` |
| BiP paused completed dataset-model groups | 2 / 8 | `experiments/paper_harness/p0_paper_exact/checkpoint_2026-08-30.json:42-43` |

All three Capsule archive/code/data manifests passed full SHA-256 verification immediately before the checkpoint commit. Partial ConE and BiP outputs are explicitly marked non-final in the checkpoint manifest.
