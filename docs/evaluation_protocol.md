# Evaluation protocol

## 切分与防泄漏

- 按完整 process run、设备或工厂分组，不随机拆散相邻时间点。
- 阈值、延时、deadband、归一化、字典和洪泛模板只用训练组拟合；验证组选择超参；测试组冻结。
- prefix 评测按洪泛已观测比例 `{10, 20, 30, 50, 70, 100}%` 输出曲线和 AUC；不得用完整序列特征反推前缀。
- open-set 测试至少留出一个根因类别，unknown 样本不并回任何已知类。
- 同一仿真随机种子或同一设备派生的窗口必须在同一 split。

## 指标

| 任务 | 必报 | 推荐补充 |
|---|---|---|
| 报警设计 | FAR、MAR、AAD | precision/recall/F1、报警率/10 min |
| 洪泛检测 | event/episode precision、recall、F1 | onset delay、duration IoU |
| 洪泛分类 | macro-F1、balanced accuracy | AUROC、prefix AUC、unknown AUROC |
| 不确定性 | empirical coverage、mean set size | singleton rate、coverage gap |
| 根因排序 | Top-1、Top-3、MRR | 边 precision/recall、lag error |
| next alarm | top-1/top-k accuracy | NLL、Brier、lead time |
| 鲁棒性 | clean 与各强度分数 | absolute/relative degradation、worst group |

FAR/MAR 默认按样本时间计算；若数据为事件日志，必须另外报告事件口径并明确分母。AAD 在异常开始到首次报警之间计算；漏报时以该异常段长度截尾，并单独报告漏报率。

## 鲁棒性矩阵

四种独立扰动：随机缺失 activation、插入 spurious tag、时间 jitter、检测延迟。建议强度为缺失率 `{0.05, 0.10, 0.20}`、虚警 `{1, 3, 5}`、jitter 为该数据集事件间隔中位数的 `{0.1, 0.3, 0.5}`、延迟 `{1, 3, 5}` 个采样周期。每个强度至少 10 个固定随机种子，报告均值与 95% bootstrap CI。

## 公平性规则

同一任务使用统一适配后事件表、同一 split 文件和同一 metric 版本。模型资源至少记录训练/推理墙钟时间、CPU/GPU、峰值内存和参数量。报告必须附配置、Git revision、数据 audit、随机种子；synthetic smoke 结果不得进入排行榜。
