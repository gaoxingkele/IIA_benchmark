# GT-GAN 全八项原始实验独立回放队列

全部八项回放已经登记并实际启动；当前等待原始训练/已发布权重评估的完整成功结果，回放完成数为 0。
153 项相关测试通过，作者 Python 3.8 冷启动及环境锁核验通过，配置冻结 307 个来源文件。
测试使用独立恢复夹具验证执行入口及拒绝条件，不是作者原生数值一致性或 benchmark 性能证明。

| 原始任务 | 完整窗口 × 长度 × 变量 | 完整生成次数 | 训练后评估器数 | 每次判别/预测训练步数 |
| --- | --- | --- | --- | --- |
| gtgan_original_stock_missing_0__native_command__seed7777 | 3662 × 24 × 6 | 17 | 340 | 2000 / 5000 |
| gtgan_original_stock_missing_0p3__native_command__seed7777 | 3662 × 24 × 6 | 4 | 80 | 2000 / 5000 |
| gtgan_original_stock_missing_0p5__native_command__seed7777 | 3662 × 24 × 6 | 4 | 80 | 2000 / 5000 |
| gtgan_original_stock_missing_0p7__native_command__seed7777 | 3662 × 24 × 6 | 4 | 80 | 2000 / 5000 |
| gtgan_original_energy_missing_0p0__released_evaluation__native_command__seed7777 | 19712 × 24 × 28 | 1 | 20 | 300 / 5000 |
| gtgan_original_energy_missing_0p5__released_evaluation__native_command__seed7777 | 19712 × 24 × 28 | 1 | 20 | 300 / 5000 |
| gtgan_original_energy_missing_0p0__native_command__seed7777 | 19712 × 24 × 28 | 12 | 240 | 300 / 5000 |
| gtgan_original_energy_missing_0p5__native_command__seed7777 | 19712 × 24 × 28 | 17 | 340 | 300 / 5000 |

总计需要 60 次完整生成、236 份原生网络检查点、1,200 个完整训练后的 TensorFlow 评估器，共 717,270 个生成窗口。
四项 Stock 保留 0/30/50/70% 缺失率；30/50% 是原作者 70% 命令的显式参数扩展，不能冒充已有论文表格中的认证结果。
Energy 的两个发布权重评估任务与两个完整训练任务分别保留；发布权重评估不计为生成器训练。
Stock 无缺失分支使用原始 regular loader，没有样条系数；缺失分支和 Energy 保留全部四个完整样条数组的审计。

生成回放恢复全部原生网络、完整输入及 CPU/CUDA RNG 字节状态，执行作者原始采样器与 recovery，全量字节比较包含正负零。
评估器回放恢复原始完整 meta graph 和全部模型/优化器变量，在完整原始测试输入上运行预测，验证变量未变化并逐字节比较所有预测。
父控制器另外复核原始成功运行凭据、结果绑定、全部网络/评估器检查点及完整保存数组，不接受只有成功标记或重新计算分数的报告。
所有回放不调用优化器，不增加训练种子，不分块代替原始全量生成；保留原始资源阈值和共享 GPU 锁。

原 Stock/Energy 队列以及六个正在运行的训练进程身份均已核验，没有停止、改写或移交这些进程。
目前不新增排行榜数值，不声称论文指标一致或 TAB 认证通过。仍保留 45 篇论文、681 条方法/基线/消融、273 条原始数据任务的完整范围。

来源：[原论文](https://arxiv.org/abs/2210.02040)，[作者代码](https://github.com/Jinsung-Jeon/GT-GAN)。
