# Sequence Modeling with Spectral Mean Flows

路径和实验参数以仓库 configs 为真源。dataset_mapping.json 只记录来源与存在性，不能作为精确 split-ready 声明。

原始文件只读；预处理写衍生目录并记录输入哈希、输出哈希与实体分组；数据物理位置沿用 F 盘存储配置。

LS4 原始四个 Monash 数据集的当前执行配置：`configs/experiments/fm_ls4_cauchy_corrected_fred_nn5_queue.v3.json`、`configs/experiments/fm_ls4_cauchy_corrected_solar_temperature_queue.v3.json`。修正仅绑定官方共轭 Cauchy 回退；全部原始科学配置、轨迹、种子和训练预算逐项比较，旧队列原样保留。

SaShiMi 连续高斯自回归的当前完整配置为 `configs/experiments/fm_sashimi_monash_gaussian_queue.v2.json`：四数据集、两结构、五配对种子，443 万次生成器更新；原文十二项指标在登记文件及模型配置中绑定 PDF 第 7 页。最后模型与 EMA 的评价分别保存，均不替代作者专用流程认证。
