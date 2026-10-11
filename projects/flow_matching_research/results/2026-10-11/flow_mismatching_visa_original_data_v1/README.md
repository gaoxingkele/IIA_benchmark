# VisA 完整原始数据核验

VisA 20220922 完整压缩包已下载、完整解包并核验：12 类、10,821 张图像、1,200 张异常掩码。原始 one-class CSV 划分为 8,659 张正常训练图像与 2,162 张测试图像。

独立校验逐一读取压缩包内全部 12,037 个文件，并对比其 SHA256、文件长度及本地解包文件；另核对固定提交 2a692ab575001cbde74d402d897a7286086c6199 的作者压缩包全部 14 个文件与原始 CSV 顺序、标签、掩码绑定。所有检查通过。训练/测试文件内容 SHA256 无交叉重复；未认证像素级重复或近重复。原始文件未修改。实际数据位于 F:\aicoding\IIA_Data 下。

原始压缩包为 1,929,840,640 字节，本地完整 SHA256 为 2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362。没有发布方完整文件校验和；S3 multipart ETag 不当作文件 MD5。

原始下载与解码记录见 full_original_visa_data_audit.json；独立包内字节和划分检查见 independent_full_archive_and_native_split_verification.json。相关测试 15 项通过，其中新增独立校验测试 4 项。

这是完整数据准备证据。完整预处理仍在执行；作者未公开的插值、RGB 范围和统计量等设置采用明确记录的本地选择，尚未认证与论文完全一致。未增加训练种子或性能指标，现有 v38 指标表逐字节保留。MVTec 下载和已有 TimeGAN、RCGAN、Pi-Transformer、CrossAD、BRITS、CFM 完整实验继续执行。
