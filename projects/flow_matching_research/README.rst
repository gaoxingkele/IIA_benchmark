流匹配算法研究
==============

研究任务
--------

完整任务与验收条件见 `tasks.json <tasks.json>`_：

* FM-001：论文、后续版本、作者代码、权重与原始实验数据。
* FM-002：按实体与时间恢复数据边界，冻结预处理和数据划分。
* FM-003：固定作者环境和版本，保留原始代码并记录修复补丁。
* FM-004：实现流匹配目标函数变种与异常评分，验证可调用实现。
* FM-005：分别执行论文、TAB 与验证集校准三种评估协议。
* FM-006：在原论文数据上进行多种子复现并核对报告指标。
* FM-007：完成 TEP、SKAB、PRONTO 等工业数据迁移和预处理。
* FM-008：分析协议、校准、评分、预算和分布变化造成的性能差异。
* FM-009：发布带出处、哈希、补丁和失败记录的研究报告。

素材证据见 `materials/catalog.json <materials/catalog.json>`_ 与
`materials/acquisition_report.json <materials/acquisition_report.json>`_。
这些文件记录本地素材路径；大文件保存在登记的素材根目录。

获取边界
--------

素材记录保留实际访问状态。MaelNet、SHCL、JFI 的完整论文仍有访问缺口；
KGL 为开放获取论文，但当前原文服务器返回 403；Pi-Transformer 的 arXiv
版本已取得，期刊版本的下载端点仍被拒绝。CFM-TS 页面返回验证内容，不能
计作完整论文。部分原始作者代码尚未找到，已取得的库或其他作者代码不替代它们。

CER-E 需要研究申请，GANF 的原始 PMU 数据未公开，ImageNet 与 Kaggle 原始
特征文件仍有授权条件。公开替代文件、当前作者发布版本与原实验同字节版本
分别记录；生成数据、CelebA-HQ 重建和实验预处理仍属于后续可执行任务。

这个独立项目集中管理 Flow Matching 时间序列异常检测研究，以及支撑研究的
论文复现、目标函数变种、缺失值填补对照和工业数据迁移任务。

``tasks.json`` 保存研究任务及依赖；``materials/catalog.json`` 保存素材清单与
下载证据。数据存在和论文复现成功是两个独立状态，只有经过完整实验和协议核验
的结果才能进入排行榜。

实现、下载脚本和实验配置通过 ``benchmark`` Git 子模块关联
``gaoxingkele/IIA_benchmark`` 的固定版本。原始大数据保留在已登记的数据路径，
该项目仓库保存出处、版本、校验值与缺口，不重复上传大数据。

克隆项目与检查素材：

.. code-block:: powershell

   git clone --recurse-submodules https://github.com/gaoxingkele/flow-matching-research.git
   cd flow-matching-research
   $env:IIA_BENCHMARK_ROOT = 'D:\aicoding\IIA_benchmark'
   python tools/check_materials.py

若在新机器获取素材，先读取 ``materials/acquisition_report.json``，使用固定版本
子模块中的 ``configs/acquisition`` 下载登记和对应脚本。需要机构权限、账号授权
或作者尚未发布的资产保持明确缺口；未取得的文件不会标记完成。

异常检测比较分别保留论文原协议、固定版本 TAB 协议以及验证集校准的部署协议。
点调整、测试标签选阈值、测试分数参与校准分别记录。比较固定随机种子、数据组、
超参数和时间戳覆盖，报告置信区间、告警预算与故障类型表现。

当前资料拉取不代表所有算法已经训练或复现成功。正在运行的旧插补队列继续留在
原工作区，新增异常检测实验需要经过数据清单和实现状态门槛。
