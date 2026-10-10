# GT-GAN原版本环境与完整原生命令执行

本轮将Spectral MeanFlow采用的GT-GAN基线从“原文代码与全量数据预处理已准备”推进到：原版本独立环境已安装冻结、4个完整原生模型构造通过、8次对应发布权重严格加载通过、4项完整训练命令已进入共享GPU队列。尚无新增GT-GAN性能结果，全部研究实验仍未完成。

## 已实际完成的验证

- 通过aria2c及17890代理下载Torch 1.9.1/CUDA 11.1与TensorFlow CPU 2.4.1的Windows安装包，均按官方索引SHA-256核验；下载回执保存实际命令、文件大小及校验值。大文件在F盘。
- 独立`.venv/gtgan-author-py38`环境冻结56个包。实际导入核验Torch为1.9.1+cu111、TensorFlow为2.4.1、CUDA工具链为11.1。NumPy 1.20.3、SciPy 1.7.1、Pandas 1.3.3、scikit-learn 0.24.2与原始Conda清单一致。
- Python为3.8.20，历史版本为3.8.10；Windows wheel及部分辅助依赖与历史Linux Conda/MKL不同。TensorFlow wheel的较窄NumPy/six/gast元数据约束按原Conda清单覆盖，实际原生导入通过；这不构成历史作者等价认证。
- 4个0/30/50/70%缺失设置都执行了原作者`main`内完整GT-GAN构造分支。每个模型包含embedder 6168、generator 10227、supervisor 7800、recovery 13182、discriminator 6865个参数，未删减网络。
- 规则Stocks和70%缺失Stocks分别严格加载原发布embedder/recovery预训练权重，以及generator/recovery最终权重，共8次通过。30/50%没有对应原发布权重，不能称它们的权重已复现。
- 对完整245个原下载文件再次核验SHA-256，全部不变；为构造核验及4个训练任务分别准备独立完整副本。只修复规则源程序4处Path加字符串和输出目录创建，移除3个未使用的可选导入，完整AST反向恢复及全部网络类、数据处理函数检查通过。
- 相关19项测试通过，包括完整作者命令、下载树身份、数据处理范围、运行时修复与预算守卫。

[完整模型构造与严格权重加载证据](../gtgan_native_model_audit_v1/native_model_construction_audit.json)；[安装包下载回执](wheel_acquisition_receipt.json)；[原文件复制与修复回执](runtime_preparation_receipt.json)。

## 原命令训练语义与完整预算

| 缺失设置 | 原生嵌入更新 | 联合训练步数 | 嵌入优化器总更新 | 生成器总更新 | 完整周期评估 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 0：发布预训练权重令原循环跳过 | 8500 | 17000 | 12750 | 17 |
| 30% | 10000，随后重载发布70%权重 | 2000 | 14000 | 4000 | 4 |
| 50% | 10000，随后重载发布70%权重 | 2000 | 14000 | 4000 | 4 |
| 70% | 10000，随后重载发布70%权重 | 2000 | 14000 | 4000 | 4 |

不规则原代码即使检测到预训练文件，也先完成10000次嵌入更新，再重新载入发布权重，覆盖这些更新。本队列保留这一行为；不能把它称为全部从头训练，也不静默改变算法以获得更好分数。规则源程序的Path加字符串错误只在从头嵌入训练分支可达，发布预训练分支跳过它；已保留实际TypeError证据和隔离修复。

全部命令保留batch=128、24步、6变量、原sym12async求解器、全部3662生成窗口。每500步执行一次完整评估，每个指标执行10次原评估器训练，共29次完整周期评估、580次指标评估器训练。评估器重复不能当成独立训练种子。判别器按原loss>0.15条件更新，不能预先要求每步都更新。30/50%沿用原70%命令的明确参数扩展，尚未证明历史论文表行等价。

[冻结完整队列](../../../../../configs/experiments/fm_gtgan_native_original_stock_queue.v1.json)。控制器实际PID与创建时间在[启动回执](controller_launch.json)中；本轮核验时为`waiting_full_gpu_resources`，无训练子进程。沿用共享GPU互斥、至少32 GiB可用物理内存、42 GiB提交余量及16 GiB紧急提交余量，额外要求18 GiB空闲显存。等待准入不能记为已开始训练或已完成。

## 其他完整实验仍在推进

本轮已重新核验LS4、MaelNet、CFM、GRASP的实际进程身份与CPU累计工作量。LS4原始PhysioNet第一项500轮任务观察到第25轮、2513次实际更新；GRASP剩余6个完整模型重推断已完成4个，当前仍在完整测试集评分。观察快照不是它们最终完成证明。

[实际进程核验](runtime_observations.json)；[LS4训练快照](ls4_active_full500_progress.json)；[GRASP完整重推断快照](grasp_full_replay_progress.json)。此前完整[7257条指标表](../all_algorithm_dataset_metrics_v14/results.html)继续保留1786条实测值与5471条缺结果；本轮环境、构造、队列和测试证据不增加性能条数。
