本轮新增CrossAD完整GPU结果种子1103，并将核心CFM/OT-CFM/SB-CFM单细胞原作者完整120条件接入独立CUDA执行计划。

CrossAD / GECCO / seed1103实际完成原20epoch、8,640训练forward；全部原作者评估和新进程完整checkpoint回放通过。所有四阶段guard均退出0且无资源中止。独立收集器又逐字节比较全部138,521个train/validation/test分数，检查完整69,261测试标签及源行号。验证集上尾1%阈值、无PA：F1 0.40037951，ROC-AUC 0.98642660，AP 0.54661278。该GPU队列现有2/139完整条件，GECCO主要五种子只完成1103、1105，不计算两种子的替代均值或区间；严格TAB和历史论文一致性未认证。

新的单细胞GPU队列实际控制器PID32340已启动，当前等待原32GiB提交内存准入；未启动新的数值worker，实际完成GPU模型0、真实GPU预检0。它没有抢占现有GPU或改变CPU训练。此注册和测试通过不能描述为已经跑完24预检或120训练。

完整120轴保留CFM/OT-CFM/SB-CFM、作者公开donor32606和独立public donor13176、released_yaml和paper_declared两个配方、两个leaveout及42..46五种子。先检查全部24真实组合：完整原128/512批次、全部9,093参数、实际反传、验证loss、101点原ODE、增强ODE及60点archival路径，另对每批所有5维初始速度逐项比较CPU/GPU，原2e-6容差保持，更新数0。全部24必须通过才允许正式训练。首个计划条件为原CPU尚未完成的released_yaml/OT-CFM/donor32606/leave1/seed43；其余全部条件保持。

完整原作者Py3.10/Torch1.12等包锁保持；模型/loss/耦合/AdamW/学习率/weight decay/128或512batch/1000epoch上限/原patience和6000秒paper cap/solver/完整所有细胞均不缩减。原CPU-only包装器仅在进程内绑定GPU可见性、tensor placement、Lightning accelerator和必须有CUDA的guard；原作者磁盘源码不改写。全FP32，关闭TF32和autocast。CUDA与CPU的训练数值等价未认证，相同整数seed的设备执行不增加独立种子数，也不抵扣CPU完成项。

每个正式条件须先完成原完整native测试，再由新GPU进程严格重载validation-selected全部权重及optimizer矩、核对实际完整epoch/update/选择规则，重跑所有完整测试timepoint的全部标量；保存全部预测/真实cell数组，并用全SciPy float64欧氏cost及POT精确LP独立计算W1。原native2e-6标量容差及全矩阵cost扰动界限保持，两套LP均要求OPTIMAL。任何失败保留，并等待隔离修复，无盲目重启。

GPU与CrossAD共用同一独占锁；原CPU锁和全部源任务保持。原16GiB物理/32GiB提交准入、16GiB紧急余量、6GiB RSS和8GiB private上限原样保留，再加18GiB VRAM准入、75%分配限制与2GiB紧急VRAM检查。GPU瞬时观察失败在数值guard中仅记录并继续轮询，不据此重新启动worker；资源保护只处理自身新子进程。新大型产物均写入F盘。

4项相关结构/全轴/独立LP/资源检查通过。保留一份注册前失败XML：最初测试只检查外层code.co_names，遗漏isclose所在的生成式；已改为递归检查全部嵌套code，未改变任何数值公式、阈值或预算。真正GPU数值预检尚待执行。

donor/历史预处理身份、原作者heldout validation参与和whitening边界、RK45/rk4论文措辞差异继续披露。历史论文数值及严格TAB认证均未取得。原285个CFM-TS、120/150/90/60原生单细胞、DSB/TrajectoryNet/CNF以及完整45论文/681方法基线消融/273原始数据任务/702标量目标全部保留，尚未全部完成。
