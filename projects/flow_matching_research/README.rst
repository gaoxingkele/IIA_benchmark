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

2026-10-07 用户提供的两个压缩包已通过完整性和标题核验，补齐 MaelNet、SHCL、
KGL、JFI、CFM-TS 全文，并新增 Pi-Transformer 期刊版本、MOMENT long-context
和 SensitiveHUE 全文。原压缩包及各版本保留；SWaT 文件中的目标论文按章节页码登记。
部分原始作者代码尚未找到，已取得的库或其他作者代码不替代它们。

CER-E 需要研究申请，GANF 的原始 PMU 数据未公开，ImageNet 与 Kaggle 原始
特征文件仍有授权条件。新全文确认的 JFI HP 私有炼化数据也未公开。
公开替代文件、当前作者发布版本与原实验同字节版本
分别记录；生成数据、CelebA-HQ 重建和实验预处理仍属于后续可执行任务。

这个独立项目集中管理 Flow Matching 时间序列异常检测研究，以及支撑研究的
论文复现、目标函数变种、缺失值填补对照和工业数据迁移任务。

``tasks.json`` 保存研究任务及依赖；``materials/catalog.json`` 保存素材清单与
下载证据。数据存在和论文复现成功是两个独立状态，只有经过完整实验和协议核验
的结果才能进入排行榜。

实现、下载脚本和实验配置通过 ``benchmark`` Git 子模块关联
``gaoxingkele/IIA_benchmark`` 的固定版本。原始大数据保留在已登记的数据路径，
该项目仓库保存出处、版本、校验值与缺口，不重复上传大数据。

本机数据位置由 ``configs/storage/data_storage.v1.json`` 登记，实际目录为
``F:/aicoding/IIA_Data/public_datasets``。旧的 ``data/public_datasets`` 路径通过
Windows 目录映射访问新位置，已冻结配置及其哈希保持有效。

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

ARA 参考文献工程
---------------

`ara/README.md <ara/README.md>`_ 收录45篇独立参考、49份全文版本。
每篇按逻辑、证据、实现和追溯四层保存方法摘要、可证伪主张、原协议实验设计、
页码/哈希、代码及数据映射。已登记71条作者报告数值，尚未转录的表格明确保留待核状态。
工程验证通过不代表论文性能已复现。

更新和检查从 ``benchmark`` 子模块根执行：

.. code-block:: powershell

   python -m scripts.literature.build_flow_matching_ara
   python -m scripts.literature.verify_flow_matching_ara

配置真源为 ``configs/reproducibility/flow_matching_ara.v1.json``。
大文件全文和完整提取文本仅保留本地；工程摘要与追溯记录进入 Git。

代码及消融覆盖审计
------------------

`ara/code_coverage.md <ara/code_coverage.md>`_ 汇总45篇参考的主方法与已审读消融，
`ara/code_coverage.json <ara/code_coverage.json>`_ 保存来源提交、代码树哈希、静态入口、
消融原文页码、既有修复补丁及运行记录。71个源码资源目录，25篇有静态定位的本地入口；本轮新增110个方法/变体配置。
``ara/code_completion.md`` 汇总实际实现、测试和未闭合项；``ara/paper_method_inventory.json``
记录45篇主项、515条对比基线提及和121条变体/组件轴。源码目录和配置数量均不等于独立算法数量。
全部方法、消融、依赖与论文等效性尚未逐项验收，论文全文齐全不能替代可复现流程。

配置真源为 ``configs/reproducibility/fm_code_coverage.v1.json``。从benchmark根更新：

.. code-block:: powershell

   python -m scripts.flow_matching.audit_code_coverage

当前CPU合成输入接口测试通过，仅验证接口。审计不启动训练、不修改现有队列；
缺失实现与未覆盖消融继续纳入FM-001、FM-003、FM-004和FM-008，数值复现由FM-006验收。
