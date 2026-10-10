流匹配算法研究
==============

算法与数据集实验结果
------------------

2026-10-11 发布的完整清单见 `可筛选全部指标 <results/2026-10-11/all_algorithm_dataset_metrics_v8/results.html>`_，
`全量 CSV <results/2026-10-11/all_algorithm_dataset_metrics_v8/all_algorithm_dataset_metrics.csv>`_，
以及 `F1 数据集矩阵 <results/2026-10-11/all_algorithm_dataset_metrics_v8/overview.md>`_。
本次快照共 7117 条汇总指标，1776 条有实测值，5341 条缺结果；指标条数不是训练次数。
严格异常检测有 730 个登记组合，210 个完成预定重复、9 个部分重复，511 个暂无完整运行。
严格异常检测、TAB 诊断、作者流程、原生方法、插补、历史记录与论文报告值分别列出。
这份清单完整保留当前登记结果与缺口，全论文和全消融实验尚未全部完成。
总表捕获后核验的 `Pi 双种子结果 <results/2026-10-11/pi_two_full_seed_audit_v2/README.md>`_ 单独保留；总表 Pi 仍为较早单种子记录。

`LS4 完整四数据集续跑 <results/2026-10-11/ls4_cauchy_corrected_resume_v3/README.md>`_ 已修正当前环境的共轭 Cauchy 回退，
40 项完整数据审计通过，40 个原始槽位及 387 万次生成器更新预算保持不变。
两个修正版控制器正在等待完整 GPU 预检资源；审计和数学一致性诊断不作为生成性能结果。

`SaShiMi 高斯自回归完整任务 <results/2026-10-11/sashimi_monash_gaussian_resume_v2/README.md>`_ 已补齐四个 Monash 数据集的
40 个生成器槽位、443 万次更新预算。40 项完整数据审计及两种结构的真实完整 FRED 流程预检通过，
原生分类器／预测器各完成 100 次更新；控制器等待完整 GPU 预检，尚无完整生成性能。
基线专用作者配置未公开，该实现标为论文重建；本地结构敏感性对照不标为论文消融。

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

算法与数据集实验指标
------------------

`最新完整指标清单 <results/2026-10-09/native_resource_recovery_0945/algorithm_dataset_metrics.md>`_ 列出算法—数据集技术指标与待完成项。
严格/TAB/插补等表保留09:20快照；`完整Excel <results/2026-10-09/complete_metrics_0920/complete_experiment_metrics.xlsx>`_ 含16个工作表。
`09:47统计协议补充 <results/2026-10-09/native_resource_recovery_0945/README.md>`_ 已核验HBOS/COPOD原登记80/80个槽，
两项同种子、同预算恢复不作为新增独立重复。原始快照、失败和训练历史均保留。
新GiFlow串行预检与完整队列共用GPU锁，预检通过后才开始正式训练；预检不计入性能。
全45篇论文、681条已审读方法/基线/消融的实验义务仍未完成。

`完整实验续跑与调度交接核验 <results/2026-10-09/tsad_light_handoff/README.md>`_ 新增GRASP mean消融的完整1500轮训练和全测试评分，
严格运行增至950次。空闲GPU调度进程交接释放约790 MiB，四套插补前置条件不变。
原生GRASP公平后继队列保留全部9120项/480组；交接观察器等待当前完整模型自然结束后再处理，
后继在释放GPU锁后让出2秒给其他原生队列。登记和等待不代表已完成交接或GiFlow训练。

`2026-10-09 02:16完整指标表 <results/2026-10-09/complete_metrics_0216/README.md>`_ 覆盖378个异常检测配置—数据集组合、165次完整严格运行、164次范围评估、39次完整插补运行及未完成项。
`Excel下载 <results/2026-10-09/complete_metrics_0216/complete_experiment_metrics.xlsx>`_ 包含9个可筛选工作表，共6309行。
历史协议、论文转录值与严格运行分别保存。`01:50旧指标快照 <results/2026-10-09/complete_metrics/README.md>`_ 保持原数值。
逐实体、逐阈值和全部数值字段还保留为CSV及压缩CSV，数值来源和SHA256可追溯。
`2026-10-08旧快照 <results/2026-10-08/README.md>`_ 保留其原始任务范围。

完整未完成实验续跑
----------------

2026-10-09执行进度 <results/2026-10-09/README.md>_ 已冻结42个可训练窗口主方法/消融、7个完整数据集、5个种子，
合计1470个基础实验；另登记105个不改变最终目标的SB/SF2M数值修复重跑。CPU续跑及修复队列已启动，
另加入175个窗口基线任务：6种既有方法以及USAD有符号损失对照。GPU续跑等待现有插补队列完成。
另加入140个真实两/三阶段reflow与40/60轮训练对照任务，保存教师参数、ODE端点配对、损失及几何统计。
该轨匹配异常检测输入，尚未证明等效于原图像实验；总epoch对照未匹配额外ODE计算开销。
另已启动MaelNet官方作者轨：6套发布配方、5个完整原生数据集、5个种子，共150个流水线任务和600个阶段。
依次训练AnomalyTransformer、DCDetector、MaelNetS2并运行完整作者DQN流程；各种子隔离检查点，保留原stride、轮数和早停。
此轨保留测试标签奖励与PA环境状态，独立记录为作者协议诊断，不并入严格TAB成绩。Python/Torch版本差异及接口预检结果已登记。
固定版本TAB的VUS和Affiliation正在对完整保存分数逐项补算；原论文、其他任务、工业异常检测、作者适配器和完整TAB训练流程仍是未完成义务。
非重叠训练窗口比部分作者stride=1少很多梯度更新，相同epoch数不等于相同训练预算，不能据当前低分判定原论文无效。

2026-10-08表未覆盖旧MTSAD目录的137份历史运行记录；不能据该表推断整个项目没有异常检测成绩。
历史数值、协议和来源已补入2026-10-09报告，并与新验证集校准、无PA、完整时间点覆盖的本地严格轨分开。

.. code-block:: powershell

   python -m scripts.flow_matching.register_tsad_execution
   python -m scripts.flow_matching.start_tsad_execution
   python -m scripts.flow_matching.register_entropic_repair
   python -m scripts.flow_matching.register_maelnet_author
   python -m scripts.flow_matching.start_maelnet_author_execution
   python -m scripts.flow_matching.summarize_tsad_execution
   python -m scripts.flow_matching.export_complete_metrics

导出后续完整指标时使用独立快照配置，保留已有快照：

.. code-block:: powershell

   python -m scripts.flow_matching.export_complete_metrics --config configs/reproducibility/fm_result_table.2026-10-09T0216.json

以上命令从benchmark根目录运行；已冻结输入和结果有哈希门禁，不覆盖原始数据或历史实验。

完整算法—数据集技术指标清单：results/2026-10-09/all_algorithm_dataset_metrics_1106/README.md，6867条汇总指标保留协议、捕获时间及未完成空缺。原始CFM-TS连续时间任务已登记30组配对ODE数据、11个模型配置、285项完整预算任务及57组五种子比较；实际结果和剩余差异单列。全论文及消融实验尚未全部完成。
