# LS4 原始 Solar、Temperature Rain 完整实验续跑

本轮新增 20 项完整原生实验，20 项真实数据核验全部通过。连同已登记的 FRED、NN5，Monash 上共有 40 项 LS4 实验；本轮新增 3,580,000 次生成器更新，总计 3,870,000 次。登记和数据核验不是训练完成：本快照新增完整训练结果为 0，GPU 完整预检尚未完成，控制器正在等待资源。

原始范围继续保留：45 篇论文、681 条方法／基线／消融文献记录、273 条原始数据／任务记录，以及新发现的支持文献实验义务。不能据此宣布全部实验完成。

## 数据与完整预算

| 数据集 | 完整矩阵 | 训练／测试轨迹 | 分支 | epoch | batch | 潜变量维度 | EMA |
|---|---|---|---|---:|---:|---:|---:|
| Solar Weekly | 137 × 52 | 109／28 | 作者发布配置 | 100000 | 128 | 5 | 0.99 |
| Solar Weekly | 137 × 52 | 109／28 | 论文正文配置 | 7000 | 64 | 5 | 0.999 |
| Temperature Rain | 32072 × 725 | 25657／6415 | 作者发布配置 | 1000 | 128 | 10 | 0.999 |
| Temperature Rain | 32072 × 725 | 25657／6415 | 论文正文配置 | 1000 | 64 | 5 | 0.999 |

每个分支使用 42、1103、1104、1105、1106 五个本地配对种子；原作者历史种子未公开。完整 Temperature 数据含所有 32072 个单变量序列，不能按 422 个测站缩减。保留作者全轨迹预处理后随机 80／20 划分：Solar 按轨迹标准化，Temperature 压缩至 [0,1]。实际逐值比较、原始 DataLoader 行覆盖和完整长度核验均通过。

两份 [Monash 官方 Solar 归档](https://zenodo.org/records/4656151)、[Temperature 归档](https://zenodo.org/records/5129091) 已通过发布方 MD5 和 ZIP CRC 校验。官方 Solar 与原有 HF 文件只在线尾 CRLF／LF 上不同，规范化文本完全一致；Temperature 逐字节相同。原始官方归档、解压成员和原有 HF 数据均保留。第一次逐字节判定失败的证据保留在 `initial_line_ending_identity_failure.json`，新 v2 下载校验单独记录，没有覆写原有文件。

## 原代码评价边界

[LS4 原始论文](https://proceedings.mlr.press/v202/zhou23i.html) 和作者提交 `2b5cb76f89f6b935bbd697771433f3c74126ffca` 为依据，40 个原始源文件的 SHA 绑定保持不变。外部执行包装记录实际更新数、评价预算和检查点；论文配置与作者发布配置分开运行。

分类器和预测器每次评价均完整训练 100 epoch。Temperature 单次分类器 8100 次更新，预测器 5100 次更新。原生分类器测试指标是 21 个 batch 的 BCE 均值之和，预测器是 51 个 batch 的 MSE 均值之和；没有擅自除以 batch 数，也没有将 BCE 当作 F1。作者预测器实际执行末尾 10 个位置的 teacher-forced 单步预测，与论文文字描述的未来多步指标等价性尚未证明。

最终指标使用原代码最后 EMA／模型，单独保存作者根据测试 marginal score 选择的最佳检查点。Temperature 的 `doc=temp_rain` 原路径被正确记录。新增包装保存评价分类器实际 randperm，供训练完成后的独立重推断审计；该审计尚未完成。

## 验证与运行状态

- 34 项相关测试通过，覆盖完整预算、全部样本、原生评价更新数以及缺失评价／短预算拒绝。
- 20 项真实数据核验通过；数据核验进程正常退出，资源保护没有中止它，峰值 RSS 1,712,050,176 字节、private bytes 4,930,048,000。
- 冻结队列 SHA256：`19efe18657f1aadd8ee2d023a1ffb818913d41ab0ab0f1d2a0ce91485e9617e0`。
- 新控制器 PID 34604、创建时间 1791656422.4316647，经实际进程命令和创建时间核验；状态为 `waiting_full_gpu_resources`。
- 完整训练仍要求可用物理内存 32 GiB、commit 42 GiB、显存 14 GiB，保留完整 batch、轨迹、原生架构和 epoch。已有完整模型继续运行。

证据：[完整数据核验](full_data_audit.json)、[资源核验](data_audit_resource_receipt.json)、[发布方获取核验](publisher_acquisition_receipt.json)、[20 项完整预算](registered_full_jobs.csv)、[运行观察](runtime_observations.json)、[验证摘要](validation.json)。这些是带时间戳的进展证据，不是实时完成计数。

SaShiMi 连续高斯任务适配、FLAME 初始条件来源、PhysioNet／USHCN 插补与外推、相关 RNN-VAE／GP-VAE／ODE2VAE／Latent ODE／TimeGAN／SDE-GAN 等基线、其他消融及训练后独立审计仍未完成。已有 FRED／NN5 20 项队列与 GiFlow 140 个原始种子槽位继续保留。
