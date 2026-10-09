# CFM-TS 原始连续时间实验

核验时间：2026-10-09T04:12:42.601557+00:00。285项完整预算任务，30组配对原始ODE数据，57组五种子比较。

当前状态：{'completed': 39, 'running': 1, 'pending': 245}。

这是原论文所规定系统上的完整模拟实验，区别于合成烟雾检查；原作者代码和原始随机轨迹未公开，作者等价尚未证明。正文/附录、原文/修正目标公式分别保留；未完成项为空。

| 协议 | 数据集 | 方法/公式 | 轮数 | 完成/预定种子 | MSE 均值 | STD | 95% 下界 | 95% 上界 | 论文 MSE | 本地减论文 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| appendix_d | pendulum | bb_paper_literal | 400 | 5/5 | 2.645927 | 3.089489 | -1.190180 | 6.482035 | 2.310000 | 0.335927 |
| main_text | pendulum | bb_paper_literal | 400 | 5/5 | 2.645927 | 3.089489 | -1.190180 | 6.482035 | 2.310000 | 0.335927 |
| appendix_d | pendulum | bb_paper_literal | 600 | 5/5 | 1.227313 | 1.211893 | -0.277450 | 2.732077 | 2.310000 | -1.082687 |
| main_text | pendulum | bb_paper_literal | 600 | 5/5 | 1.227313 | 1.211893 | -0.277450 | 2.732077 | 2.310000 | -1.082687 |
| appendix_d | pendulum | bb_gaussian_consistent | 400 | 5/5 | 1.000497 | 0.288423 | 0.642372 | 1.358621 | 2.310000 | -1.309503 |
| main_text | pendulum | bb_gaussian_consistent | 400 | 5/5 | 1.000497 | 0.288423 | 0.642372 | 1.358621 | 2.310000 | -1.309503 |
| appendix_d | pendulum | bb_gaussian_consistent | 600 | 5/5 | 1.211188 | 0.593686 | 0.474030 | 1.948346 | 2.310000 | -1.098812 |
| main_text | pendulum | bb_gaussian_consistent | 600 | 4/5 | 1.345653 | 0.591109 | 0.405066 | 2.286241 | 2.310000 | -0.964347 |
| appendix_d | pendulum | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.090000 | — |
| main_text | pendulum | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.090000 | — |
| appendix_d | pendulum | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.090000 | — |
| main_text | pendulum | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.090000 | — |
| appendix_d | pendulum | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.090000 | — |
| main_text | pendulum | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.090000 | — |
| appendix_d | pendulum | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.090000 | — |
| main_text | pendulum | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.090000 | — |
| appendix_d | pendulum | node | 10 | 0/5 | — | — | — | — | — | — |
| main_text | pendulum | node | 400 | 0/5 | — | — | — | — | — | — |
| main_text | pendulum | node | 600 | 0/5 | — | — | — | — | — | — |
| appendix_d | sine | bb_paper_literal | 400 | 0/5 | — | — | — | — | 0.070000 | — |
| main_text | sine | bb_paper_literal | 400 | 0/5 | — | — | — | — | 0.070000 | — |
| appendix_d | sine | bb_paper_literal | 600 | 0/5 | — | — | — | — | 0.070000 | — |
| main_text | sine | bb_paper_literal | 600 | 0/5 | — | — | — | — | 0.070000 | — |
| appendix_d | sine | bb_gaussian_consistent | 400 | 0/5 | — | — | — | — | 0.070000 | — |
| main_text | sine | bb_gaussian_consistent | 400 | 0/5 | — | — | — | — | 0.070000 | — |
| appendix_d | sine | bb_gaussian_consistent | 600 | 0/5 | — | — | — | — | 0.070000 | — |
| main_text | sine | bb_gaussian_consistent | 600 | 0/5 | — | — | — | — | 0.070000 | — |
| appendix_d | sine | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.005000 | — |
| main_text | sine | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.005000 | — |
| appendix_d | sine | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.005000 | — |
| main_text | sine | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.005000 | — |
| appendix_d | sine | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.005000 | — |
| main_text | sine | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.005000 | — |
| appendix_d | sine | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.005000 | — |
| main_text | sine | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.005000 | — |
| appendix_d | sine | node | 10 | 0/5 | — | — | — | — | 0.040000 | — |
| main_text | sine | node | 400 | 0/5 | — | — | — | — | 0.040000 | — |
| main_text | sine | node | 600 | 0/5 | — | — | — | — | 0.040000 | — |
| appendix_d | lotka_volterra | bb_paper_literal | 400 | 0/5 | — | — | — | — | 0.008000 | — |
| main_text | lotka_volterra | bb_paper_literal | 400 | 0/5 | — | — | — | — | 0.008000 | — |
| appendix_d | lotka_volterra | bb_paper_literal | 600 | 0/5 | — | — | — | — | 0.008000 | — |
| main_text | lotka_volterra | bb_paper_literal | 600 | 0/5 | — | — | — | — | 0.008000 | — |
| appendix_d | lotka_volterra | bb_gaussian_consistent | 400 | 0/5 | — | — | — | — | 0.008000 | — |
| main_text | lotka_volterra | bb_gaussian_consistent | 400 | 0/5 | — | — | — | — | 0.008000 | — |
| appendix_d | lotka_volterra | bb_gaussian_consistent | 600 | 0/5 | — | — | — | — | 0.008000 | — |
| main_text | lotka_volterra | bb_gaussian_consistent | 600 | 0/5 | — | — | — | — | 0.008000 | — |
| appendix_d | lotka_volterra | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.008600 | — |
| main_text | lotka_volterra | gp_paper_literal | 400 | 0/5 | — | — | — | — | 0.008600 | — |
| appendix_d | lotka_volterra | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.008600 | — |
| main_text | lotka_volterra | gp_paper_literal | 600 | 0/5 | — | — | — | — | 0.008600 | — |
| appendix_d | lotka_volterra | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.008600 | — |
| main_text | lotka_volterra | gp_gaussian_conditional | 400 | 0/5 | — | — | — | — | 0.008600 | — |
| appendix_d | lotka_volterra | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.008600 | — |
| main_text | lotka_volterra | gp_gaussian_conditional | 600 | 0/5 | — | — | — | — | 0.008600 | — |
| appendix_d | lotka_volterra | node | 10 | 0/5 | — | — | — | — | 0.060000 | — |
| main_text | lotka_volterra | node | 400 | 0/5 | — | — | — | — | 0.060000 | — |
| main_text | lotka_volterra | node | 600 | 0/5 | — | — | — | — | 0.060000 | — |

## 复现边界

- Formal original-task simulations, not TSAD smoke data. Author code and raw simulation files are unavailable; author equivalence is not certified.
- Main text and appendix conflicts are separate full protocols, not selected by test performance. Train/test trajectory IDs are paired across algorithms within each track and seed.
- For LV, 50 observation locations are sampled without replacement from 500 random ground-truth locations. Sine ground-truth evaluation count is not stated; 500 evaluation locations are an explicit local choice.
- Pendulum has one shared training/evaluation trajectory, evaluated at new times. The inconsistent open interval and n=160 are interpreted as t=0,0.1,...,15.9; test times are newly drawn in (0,16). No hidden angular velocity is passed to the learned models.
- Algorithms print a grid from 0 to T, but boundary handling is unspecified. Training target grid is observation first-to-last, with n=observation count, to avoid inventing BB extrapolation; full evaluation remains 0 to T.
- Input64 is interpreted as a projection of observed state and physical time to 64 features followed by three 258-unit hidden layers. Time encoding and LeakyReLU slope are unspecified; raw time and slope0.01 are explicit choices.
- BB sample count is unspecified; one marginal sample per grid location is frozen. GP uses the stated ten joint function samples. Prior initialization m0=y0,P0=1 and standard RTS repairs are explicit.
- GP MLE uses L-BFGS-B max400 iterations, not author-optimizer epochs; observation variance0.001 and joint jitter1e-9 are local choices. GP output dimensions share a kernel.
- NODE uses time-conditioned same-size network and dopri5 with adjoint gradients, with batch units interpreted as trajectories. Irregular-time trajectories are streamed within each batch; the exact averaged gradient is preserved. No test trajectory enters fitting.
- Solver tolerances, CPU/Python versions, random seed identities, MSE averaging and evaluation-grid choices are not specified by the authors and remain reproduction gaps. Raw physical units and full frozen budgets are retained.

## 调度与完整重跑

原调度器在核验已有结果后占用805,806,080字节；新等待父进程约14 MB，真实完整产物核验仍在独立子进程执行。原285个槽位、57组比较及模型/数据/种子/预算/保护门限保持不变。

两次内存保护退出分别按原种子和完整400/600轮预算重跑，已完成并解决原槽位；失败产物保留，重跑不增加统计种子数。首次重跑完成后出现元数据收据目录缺失，已修复元数据写入并接续第二次完整任务。4项针对性测试通过。

[接管记录](transition.json)；[目录修复与接续](receipt_repair_launch.json)；[实际进程观察](runtime_observations.json)；[验证](validation.json)。
