# IEEE DataPort 数据访问申请邮件草稿

> 当前执行环境没有邮件连接器，以下内容尚未发送。请使用厦门大学邮箱发送，
> 并在下载后保持原始归档文件名，将文件放到文中约定目录。

- 收件人：丁威 `<dingwei@stu.xmu.edu.cn>`；陈经安 `<chenjingan@stu.xmu.edu.cn>`
- 抄送：`iamafan@126.com`
- 主题：请协助申请并下载智能工业报警 Benchmark 所需 IEEE DataPort 数据集

丁威、陈经安：

你们好！

我们正在构建 Intelligent Industrial Alarm（IIA）Benchmark，需要使用与专著案例、
TEP/PRONTO 报警数据形态相近的 IEEE DataPort 数据集完成报警洪泛检测、聚类与分类、
开放集识别和根因分析验证。烦请使用 IEEE DataPort 账号访问下列记录，确认并遵守各记录
的许可条款后，下载其中的完整原始归档：

1. Tennessee Eastman Process Alarm Management Dataset  
   https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset  
   DOI: `10.21227/326k-qr90`
2. Nuclear Power Plant Alarm Dataset  
   https://ieee-dataport.org/open-access/nuclear-power-plant-alarm-dataset  
   DOI: `10.21227/g2fa-9y43`

请不要转换、重命名或解压后覆盖原文件。下载完成后，请把原始归档及 DataPort 页面显示的
文件清单、许可说明、版本/发布日期和校验值（若页面提供）一并交付。项目建议落盘目录为：

- `data/public_datasets/tep_alarm_dataport/`
- `data/public_datasets/npp_alarm_dataport/`

收到文件后，Benchmark 会先执行哈希、文件清单、压缩包路径安全和 schema 审计，再建立
训练/验证/测试分组；未经审计的文件不会用于排行榜结果。

谢谢！

IIA Benchmark 项目组
