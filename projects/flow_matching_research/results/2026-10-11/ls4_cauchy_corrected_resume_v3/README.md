# LS4 原始完整实验：Cauchy 回退修正与续跑

本轮保留 FRED、NN5、SolarWeekly、TemperatureRain 的全部 40 个原始种子槽位、两个协议及 3,870,000 次生成器更新。旧控制器、旧数据审计和未成功登记的 v2 文件均保留；新执行使用独立 v3 目录。实际数据审计、控制器身份及资源状态见 [validation.json](validation.json) 和 [runtime_observations.json](runtime_observations.json)。数据审计和接口诊断不是完成的生成实验。

LS4 下载源码的 `models/s4.py` 中，朴素 Cauchy 回退只计算存储的半组复数极点，没有补上共轭项。其 CUDA/PyKeOps 分支及递推实现计算对称共轭对；当前锁定环境没有这两个加速后端，因而使用了不同的数学计算。修正版在私有进程内绑定官方 S4 的共轭对称 `cauchy_naive`，原始下载文件保持逐字节不变。

在原始 Python 3.10/Torch 1.12.1 环境中，审计了完整 FRED 107 条序列及 85/22 轨迹划分，再用相同初始参数和两条完整 728 点真实测试序列比较 S4 卷积与稠密递推：原回退最大差异 1.166693449，修正后 0.0000166595，通过预定 `rtol=atol=1e-4` 标准。[成对真实数据诊断](../ls4_cauchy_corrected_resume_v2/paired_real_fred_diagnostic.json) 是数学一致性证据，不是训练后性能，也不证明历史作者环境等价。

| 数据集 | 完整序列数 × 长度 | 协议 | 种子数 |
|---|---:|---|---:|
| FRED-MD | 107 × 728 | released / paper_literal | 各 5 |
| NN5 Daily | 111 × 791 | released / paper_literal | 各 5 |
| SolarWeekly | 137 × 52 | released / paper_literal | 各 5 |
| TemperatureRain | 32072 × 725 | released / paper_literal | 各 5 |

每个完整任务仍执行原始生成器预算、周期及最终评价；分类器和预测器仍各训练 100 epoch，覆盖完整测试轨迹，保留原始批次聚合定义。没有缩小模型、序列、训练集、种子组或轮数。GPU/内存门槛保持原值：空闲显存 14 GiB、物理内存 32 GiB、提交余量 42 GiB。

切换脚本核对原控制器 PID、创建时间、命令、空闲状态及所有子进程，然后只结束两个尚未启动完整 GPU 模型的控制器；没有中断训练模型。[交接记录](../ls4_cauchy_corrected_resume_v2/full_idle_handoff_v3.json) 保留原始身份和新控制器身份。

一次 v2 登记因假定两种模型配置都含 `model` 字段而失败，未发布队列、未启动模型。原始 FRED/NN5 的架构存于已绑定哈希的 YAML；v3 校验全部原始科学字段，兼容两类配置结构，并增加真实 40 槽位配置结构测试。[失败记录](../ls4_cauchy_corrected_resume_v2/registration_v2_failure_preserved.json) 保留该历史，失败不增加种子。

GiFlow 的旧 seed394 已跑至原生 epoch299 并输出测试数值，但旧更新次数检查仍无法签发完整结果，不能并入已完成指标。预设边界观察器在模型自然退出后完成 v4 交接，新的原始 seed393 已开始训练。旧预测、测试模型和日志保留。CFM-TS 和 Pi 的既有完整模型继续运行；具体身份以运行快照为准。

本轮相关 47 项测试通过。全目标继续保留 45 篇规范论文、681 条方法/基线/消融文献记录、273 条原始数据/任务记录及新增参考论文义务。SaShiMi 高斯自回归基线、LS4 其他原始数据任务、所有尚未完成的原论文实验和消融继续待完成；本轮不将登记或审计计为性能结果。

来源：[LS4 论文](https://proceedings.mlr.press/v202/zhou23i.html)、[原始 LS4 源码](https://github.com/alexzhou907/ls4/blob/2b5cb76f89f6b935bbd697771433f3c74126ffca/models/s4.py)、[官方 S4 共轭 Cauchy 实现](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/models/functional/cauchy.py)。锁定输入、完整预算和路径以 `configs/reproducibility/fm_ls4_cauchy_corrected_registration.v3.json` 及两个后继队列为准。
