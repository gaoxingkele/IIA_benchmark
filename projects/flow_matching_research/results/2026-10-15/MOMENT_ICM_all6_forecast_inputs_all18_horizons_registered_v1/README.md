# Tiny／ICM：六个原始预测数据集、全部18组horizon的独立输入链路

原6数据集Exchange、ETTh1/2、ETTm1/2、Weather全部保留；lookback256、horizon96/192/384、全多变量通道、stride1。相关作者预测配置对Exchange和Weather使用70/10/20，而已闭合预训练缓存是60/10/30，两套划分明确分开。ETT继续使用来源固定12/4/4个月切点。原始CSV及旧缓存不覆盖。

使用冻结原LongForecastingDataset类，训练段拟合StandardScaler，三个角色用同一训练统计量。val/test带256点历史，但完整所有forecast目标必须位于各自新时间段；检查全部窗口起点和目标末端边界，独立记录每个horizon的窗口/输入坐标/目标坐标数。每个角色只保存完整TC矩阵、原CSV行映射及共享mask，逐字节保存全部来源坐标；实际源__getitem__对每horizon首/中/末窗口核对history/forecast转置、维度、mask和端点，不冒称逐窗独立重建了所有值，也不生成数百倍重复窗口张量。

固定ETT月份可能不使用CSV末尾额外月份；这些原行仍在原文件保留，选中末行及未进入来源固定月份的行数明确记录。原全CSV cubic插值在划分前发生，保持且披露；不能据训练拟合缩放就宣称整个插值严格无未来信息。此包不认证历史作者ICM精确预处理等价。

17项检查通过，覆盖真实原类在模拟Exchange和ETTh1的完整三角色/三horizon、首中末窗口、训练统计、所有目标边界、原始文件保留、6源18组合完整性、资源/配置防缩减、冷导入以及原保护控制器一致性。夹具不是benchmark预测结果。保留先前1项测试比较失败：全模块与单函数编译在Python3.12对module调用有等价NULL标志编码；最终在相同编译上下文递归核对原字节码/常量，只允许两条描述性状态名变化。未失败或重跑科学数据worker，也未改保护规则。

已实际启动独立元数据owner，沿用原同一共享CPU锁与16GiB物理/21GiB提交准入、16GiB紧急下限、3GiB RSS/私有内存上限。原源环境Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2保留；等待原资源准入，不旁路其他训练或更改预算。数据worker须完成全部6源、18horizon、54角色组并成功保护退出才算完整。

当前包是全量预测输入执行实现和实际等待队列，尚未产出完整数据证明、训练预测器或新的MSE/MAE。相关预测配置还提供5epoch、train batch64/val batch256、OneCycleLR及AMP等来源参数，已另表冻结为后续执行依据；它们不证明未公开ICM历史配方。两模型完整PILE预训练、forecast head/β微调、Graph/Concat/Crossformer/iTransformer/NBEATS及全部702论文标量目标继续保留，8fold/4fold与freezeβ的对应未建立，不能擅自等同。至少45篇/681方法记录/273数据任务及新增实验范围不变，整个研究目标未完成。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[相关作者代码](https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de)、[Autoformer数据](https://github.com/thuml/Autoformer)。
