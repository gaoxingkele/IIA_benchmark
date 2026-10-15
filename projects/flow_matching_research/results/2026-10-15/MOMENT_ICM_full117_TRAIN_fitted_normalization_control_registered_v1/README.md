# Tiny／ICM：117数据集的TRAIN拟合归一化对照

原117数据集、234个完整官方TRAIN/TEST的来源预处理已实际完成，每例成功保护回执及报告哈希重新核对。相关MOMENT作者代码在读取TRAIN和TEST时分别StandardScaler.fit(self.data)，因此缩放使用各自完整分布；不使用TEST标签，但不是TRAIN统计量固定的推断流程。此前来源分支完整保留，不用当前对照替换或追溯认证论文。

新对照覆盖同一26多变量+91单变量数据集。逐坐标验证原完整TRAIN缓存后直接引用它；TEST复用从全部官方TRAIN拟合的标准化器，明确禁止任何TEST fit/update。原未标准化例外保留（本范围21个）；这些例外要求新旧完整值、mask、标签、实例行相等。其余数据记录新旧所有TEST坐标的变化数及最大绝对差，标签和实例顺序必须完全相同。

不同原始长度的TEST用来源插值函数对齐到TRAIN标准化器的原始长度，然后应用TRAIN统计量；这是显式对照适配，记录原TEST长度和TRAIN长度。来源NaN插值、变长序列预处理、13种插值例外、剩余补齐/下采样规则、全部实例和通道均保留。来源变长TEST内部统一长度的处理仍保留，所以这里只认证归一化统计量不由TEST拟合，不将所有预处理宣称为完全归纳式或无传导信息。

20项检查通过，含实际原作者解析器/插值/StandardScaler的同长、异长、NaN和未标准化例外；检查固定TRAIN统计量、禁止TEST fit、全部坐标/标签/顺序和原保护控制器。测试使用夹具，不能解释为benchmark性能。冷检查确认117来源完成证据和234原始分割，未导入NumPy/Torch或启动数据worker。

新独立元数据owner实际运行，沿用同一共享CPU锁及16GiB物理/21GiB提交准入、16GiB紧急下限、3GiB RSS/私有内存上限。未缩减样本、通道、窗口或网格，不覆盖原始数据及来源缓存。每例以原环境Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2运行，失败或部分尝试保留且不自动重跑。

本包是归一化对照实现、原117来源完整绑定和实际新队列登记。快照新对照已完成0/117；尚未运行新的Tiny/ICM分类特征提取或SVM，分类准确率、模型间比较和全部702论文指标实验均未称完成。完整PILE两模型训练及后续预测/SVM仍须真实结果，至少45篇/681方法记录/273数据任务及新增实验范围不变。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[相关MOMENT源代码](https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de)、[UCR／UEA数据](https://www.timeseriesclassification.com/)。
