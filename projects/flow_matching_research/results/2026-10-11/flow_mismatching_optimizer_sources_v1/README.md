# StableAdamW 原始材料及可调用的相关作者实现

已下载并核验 Dinomaly 固定提交 1f252be03a918789b19848f0ca37166e7c28dada 的全部 197 个文件，以及 optimi 固定提交 17a3d5422f3e41bb8973e1e4c9b39206c0d93b6d 的全部 66 个文件。完整 ZIP CRC、Git fsck、每个 Git blob SHA1 及源文件字节一致。所有原始源码及许可均保存在 F:\aicoding\IIA_Data，未改写。

Stable and low-precision training for large-scale vision-language models（arXiv:2304.13013v2，NeurIPS 2023）全文 29 页已下载，完整 PDF SHA256/严格解析通过；第 10 页 Algorithm 2 已渲染并目视核对。未声称全部 29 页已人工审阅。

已提供独立载入 optimizers/StableAdamW.py 的完整原始类和配置，使用目标论文登记的 lr=1e-4、weight_decay=1e-4、betas=(0.9,0.99)、eps=1e-6、AMSGrad=true、RMS 阈值 1。9 项源码、数值和状态校验通过，无跳过；包括独立 NumPy 递推、梯度突增/零值、AMSGrad/非 AMSGrad、恢复状态和拒绝源码篡改。

同名优化器存在实际区别：Dinomaly 在 RMS 中使用加 epsilon 的 AMSGrad 去偏分母，权重衰减使用未裁剪学习率；原始 StableAdamW Algorithm 2 使用去偏二阶矩计算 RMS，并以裁剪后的学习率执行衰减。原优化器论文不定义 AMSGrad 扩展。这些是实现差异，未判定为作者错误。Flow Mismatching 作者具体实现身份仍待核实，相关作者可调用实现不等于目标论文原始实现。

本批证据补齐训练组件素材和 callable，未执行完整图像训练，未增加性能指标或种子；所有 45 篇论文、681 个方法/基线/消融记录、273 个原始数据任务继续保留。MVTec 全量原始数据核验已完成，其预处理与现有原始实验继续运行。
