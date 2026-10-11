# Flow Mismatching 原始图像实验：补齐数据前置条件

本轮补齐了此前未登记的原论文 MVTec-AD、VisA 数据拉取与完整审计流程。它们属于既有 45 篇论文范围中的图像异常检测任务，不能作为时间序列 TAB 对比成绩。

| 数据 | 原始规模与划分依据 | 当前状态 |
|---|---|---|
| VisA | 作者原始 20220922 TAR；12 类、10,821 图像身份，9,621 正常、1,200 异常；完整 `1cls.csv` | 完整原始包下载中；作者源码及划分身份审计通过 |
| MVTec-AD | 原始 15 类数据；目标 5,354 图像；保留 train/test/ground_truth 目录 | 完整原始包下载中；已取得可访问链接及 Anomalib 的 SHA256 |

VisA 原作者仓库冻结到 `2a692ab575001cbde74d402d897a7286086c6199`。作者 ZIP 的 14 个文件逐个与 Git blob、Git 对象内容比对，全部通过，并运行了 `git fsck --strict --full`。完整 10,821 个原始图像身份唯一、训练／测试身份不重复、训练仅包含正常条目。此检查是**源码及 CSV 身份检查**，尚未证明这些原始图像和掩码均已下载、解码完成。

两个下载均通过 `aria2c` 与登记代理 `http://127.0.0.1:17890` 执行，原始包与提取目录实际位于 F 盘的数据根目录下。路径由 configs 控制，原始数据不会被预处理覆盖。下载结束后，控制器自动继续完整校验、逐文件 SHA256、全部图像与掩码解码、原始划分与像素掩码对齐检查。原始失败或中断产物必须保留。

MVTec-AD 旧的公开链接实测返回 404；Anomalib 当前维护的公开链接返回 200，登记完整长度 5,264,982,680 字节和 SHA256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`。VisA 原始包的官方长度为 1,929,840,640 字节；S3 multipart ETag 只登记为对象版本信息，不能误称为整包 MD5 或作者 SHA256。VisA 将记录本地流式 SHA256，并完成全部归档、图像和作者划分核验。

还进行了独立直连的 1 MiB Range 读取，两者均为 206 且与正在通过代理拉取的原始包前缀 SHA256 一致。此小范围探测不代表整包下载完成，也没有因此终止或替换正在运行的下载。

相关 9 项路径、完整划分身份和异常掩码对应关系回归检查通过。它们只验证拉取／审计逻辑，不能称为论文性能。

仍需完成：两项整包下载与全部实际图像核验；原文完整训练配方、StableAdamW 优化器及评估实现的精确登记；原始完整多种子训练；全部原文消融与竞争方法的复现。现有 U-Net 与评分组件不会被冒充为已完成原作者实验。

- [作者源码及完整划分身份证据](original_author_and_split_metadata_audit.json)
- [本轮实际下载与训练句柄](actual_download_and_training_handles.json)
- [原始 VisA 数据及作者划分说明](https://github.com/amazon-science/spot-diff)
- [MVTec-AD 官方数据说明](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Anomalib 维护的完整包下载与校验值](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/src/anomalib/data/datamodules/image/mvtecad.py)

完整运行产物：`flow_mismatching_visa_original_data_v1/full_original_visa_data_audit.json` 与 `flow_mismatching_mvtec_original_data_v1/full_original_mvtec_data_audit.json` 将在各控制器完整执行成功后生成；本轮尚无这两个成功产物。
