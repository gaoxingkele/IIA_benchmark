# Spectral Mean Flow 原始生成实验

捕获时间：2026-10-09T03:46:48.555281+00:00。70项任务、14组五训练种子比较；状态：{'pending': 70}。

作者Table1原始六数据集、完整训练步数、完整生成数组及四个原评价器分别冻结。Sines发布代码和论文分布保留两条协议；预检不计论文成绩。

| 方法 | 数据集 | 数据协议 | 指标 | 完成/预定训练种子 | 均值 | STD | 95%下界 | 95%上界 | 论文值 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| spectral_flow | stocks | released_author_data | context_fid | 0/5 | — | — | — | — | 0.008000 |
| spectral_flow | stocks | released_author_data | correlational | 0/5 | — | — | — | — | 0.010000 |
| spectral_flow | stocks | released_author_data | discriminative | 0/5 | — | — | — | — | 0.022000 |
| spectral_flow | stocks | released_author_data | predictive | 0/5 | — | — | — | — | 0.037000 |
| diffusion_ts | stocks | released_author_data | context_fid | 0/5 | — | — | — | — | 0.169000 |
| diffusion_ts | stocks | released_author_data | correlational | 0/5 | — | — | — | — | 0.010000 |
| diffusion_ts | stocks | released_author_data | discriminative | 0/5 | — | — | — | — | 0.085000 |
| diffusion_ts | stocks | released_author_data | predictive | 0/5 | — | — | — | — | 0.037000 |
| spectral_flow | energy | released_author_data | context_fid | 0/5 | — | — | — | — | 0.051000 |
| spectral_flow | energy | released_author_data | correlational | 0/5 | — | — | — | — | 0.732000 |
| spectral_flow | energy | released_author_data | discriminative | 0/5 | — | — | — | — | 0.161000 |
| spectral_flow | energy | released_author_data | predictive | 0/5 | — | — | — | — | 0.251000 |
| spectral_flow | etth | released_author_data | context_fid | 0/5 | — | — | — | — | 0.058000 |
| spectral_flow | etth | released_author_data | correlational | 0/5 | — | — | — | — | 0.040000 |
| spectral_flow | etth | released_author_data | discriminative | 0/5 | — | — | — | — | 0.027000 |
| spectral_flow | etth | released_author_data | predictive | 0/5 | — | — | — | — | 0.123000 |
| spectral_flow | fmri | released_author_data | context_fid | 0/5 | — | — | — | — | 0.116000 |
| spectral_flow | fmri | released_author_data | correlational | 0/5 | — | — | — | — | 0.737000 |
| spectral_flow | fmri | released_author_data | discriminative | 0/5 | — | — | — | — | 0.136000 |
| spectral_flow | fmri | released_author_data | predictive | 0/5 | — | — | — | — | 0.100000 |
| spectral_flow | mujoco | released_author_data | context_fid | 0/5 | — | — | — | — | 0.018000 |
| spectral_flow | mujoco | released_author_data | correlational | 0/5 | — | — | — | — | 0.173000 |
| spectral_flow | mujoco | released_author_data | discriminative | 0/5 | — | — | — | — | 0.005000 |
| spectral_flow | mujoco | released_author_data | predictive | 0/5 | — | — | — | — | 0.008000 |
| spectral_flow | sines | released_author_data | context_fid | 0/5 | — | — | — | — | 0.004000 |
| spectral_flow | sines | released_author_data | correlational | 0/5 | — | — | — | — | 0.027000 |
| spectral_flow | sines | released_author_data | discriminative | 0/5 | — | — | — | — | 0.006000 |
| spectral_flow | sines | released_author_data | predictive | 0/5 | — | — | — | — | 0.093000 |
| spectral_flow | sines | paper_sines_distribution | context_fid | 0/5 | — | — | — | — | 0.004000 |
| spectral_flow | sines | paper_sines_distribution | correlational | 0/5 | — | — | — | — | 0.027000 |
| spectral_flow | sines | paper_sines_distribution | discriminative | 0/5 | — | — | — | — | 0.006000 |
| spectral_flow | sines | paper_sines_distribution | predictive | 0/5 | — | — | — | — | 0.093000 |
| diffusion_ts | energy | released_author_data | context_fid | 0/5 | — | — | — | — | 0.113000 |
| diffusion_ts | energy | released_author_data | correlational | 0/5 | — | — | — | — | 0.788000 |
| diffusion_ts | energy | released_author_data | discriminative | 0/5 | — | — | — | — | 0.154000 |
| diffusion_ts | energy | released_author_data | predictive | 0/5 | — | — | — | — | 0.251000 |
| diffusion_ts | etth | released_author_data | context_fid | 0/5 | — | — | — | — | 0.126000 |
| diffusion_ts | etth | released_author_data | correlational | 0/5 | — | — | — | — | 0.049000 |
| diffusion_ts | etth | released_author_data | discriminative | 0/5 | — | — | — | — | 0.075000 |
| diffusion_ts | etth | released_author_data | predictive | 0/5 | — | — | — | — | 0.121000 |
| diffusion_ts | fmri | released_author_data | context_fid | 0/5 | — | — | — | — | 0.118000 |
| diffusion_ts | fmri | released_author_data | correlational | 0/5 | — | — | — | — | 1.252000 |
| diffusion_ts | fmri | released_author_data | discriminative | 0/5 | — | — | — | — | 0.158000 |
| diffusion_ts | fmri | released_author_data | predictive | 0/5 | — | — | — | — | 0.100000 |
| diffusion_ts | mujoco | released_author_data | context_fid | 0/5 | — | — | — | — | 0.015000 |
| diffusion_ts | mujoco | released_author_data | correlational | 0/5 | — | — | — | — | 0.188000 |
| diffusion_ts | mujoco | released_author_data | discriminative | 0/5 | — | — | — | — | 0.012000 |
| diffusion_ts | mujoco | released_author_data | predictive | 0/5 | — | — | — | — | 0.007000 |
| diffusion_ts | sines | released_author_data | context_fid | 0/5 | — | — | — | — | 0.013000 |
| diffusion_ts | sines | released_author_data | correlational | 0/5 | — | — | — | — | 0.016000 |
| diffusion_ts | sines | released_author_data | discriminative | 0/5 | — | — | — | — | 0.030000 |
| diffusion_ts | sines | released_author_data | predictive | 0/5 | — | — | — | — | 0.095000 |
| diffusion_ts | sines | paper_sines_distribution | context_fid | 0/5 | — | — | — | — | 0.013000 |
| diffusion_ts | sines | paper_sines_distribution | correlational | 0/5 | — | — | — | — | 0.016000 |
| diffusion_ts | sines | paper_sines_distribution | discriminative | 0/5 | — | — | — | — | 0.030000 |
| diffusion_ts | sines | paper_sines_distribution | predictive | 0/5 | — | — | — | — | 0.095000 |

## 剩余原论文任务

- Table2 larger models and SDFormer/ImagenTime
- Table3 long FRED-MD/NN5 models and baselines
- Table4 irregular stock protocols and Koopman VAE
- Table5 physics-informed Pendulum and Koopman VAE
- Figure2 tractability with explicit device allocation limits
- Figure3 full checkerboard and RBF gradient-flow baselines
- Remaining adopted Table1 baseline scores need independent executions

## 比较边界

- Author source Table1 configurations are frozen unchanged; max_epochs denotes optimizer updates, not full passes over the data. Baseline gradient accumulation remains its own published configuration.
- Released Sines samples frequency and phase from [0,0.1]; paper p25 specifies frequency [0,1] and phase [-pi,pi]. Both protocols are registered independently, never selected by resulting metric.
- The original generation experiment uses all stride-one windows and fits MinMaxScaler on the full source series. This is distribution reconstruction, not held-out future-trajectory generalization or strict anomaly detection.
- Raw real CSV/MAT bytes are extracted from the registered existing archive without overwriting it; original author loaders generate frozen cached arrays. Data seed123 is paired across algorithms and model seeds.
- Original generation remains completely unmodified with batch2001, full noise draws and author ODE grids. Chunked fields are excluded: one-step agreement failed to imply full integrated trajectory agreement in a direct 500-step test (maximum difference10.226529 on the diagnostic small model). No sampling equivalence is claimed for chunking.
- Five evaluator repetitions occur inside each complete independently trained model seed. Evaluator CI and between-training-seed STD/CI must be reported separately.
- TensorFlow evaluators originally reset graphs without a fixed graph seed. The runtime reset wrapper now assigns graph seeds0..4; evaluator optimizer steps, networks, split functions and inputs remain unchanged. This reproducibility change is recorded separately from the released unseeded evaluator protocol.
- Author display_scores computes a t-based95% half-width from five evaluations. The printed ± must not be relabeled as standard deviation without paper clarification.
- Torch2.8/CUDA12.6 and TensorFlow2.15/TFP0.23 match recommended versions, but Windows/RTX3090 differs from the tested Linux/H100 setup. All unspecified dependency versions and hardware differences are recorded in a local lock.
- No real leaderboard metric is inferred from interface preflights. All trained steps, full reference/generated arrays, model checkpoints and evaluator raw repeats must be verified before marking a job complete.
