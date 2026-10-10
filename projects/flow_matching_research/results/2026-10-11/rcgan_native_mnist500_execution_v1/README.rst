RCGAN 作者原生 MNIST 完整500轮队列
================================

本轮接入作者提供的mnistfull设置：无条件RGAN（cond_dim=0）、500轮、batch28、D1/G1、G/D隐藏宽度100、latent5。五个原始登记种子1103至1107均保留；每个种子642000次D更新、642000次G更新及500次完整评估。首个seed1103原生工作进程已启动。完成量须以各工作目录result.json及完整验证为准，目前没有完成500轮的成绩。

全部原始范围仍为45篇论文、681项方法/基线/消融记录、273项原始数据任务及采用基线；这些义务没有缩减为MNIST或已有结果。

先核验了作者完整评估：原60k官方MNIST训练图片按源代码分为36k/12k/12k，所有划分像素、标签和行身份验证；官方10k测试图片不用于这一作者划分。核带宽遍历全部144000000个验证图片对，原计算为有限20.4295991101166，没有负距离或NaN，不对生产训练的原带宽函数做数值修正。

原评估完整生成4984个样本；996个样本用于带宽拟合，3988个样本用于后续MMD评估，最多2000次原RMSProp更新、原停止差值阈值0.001。本次初始化模型容量检查按原停止条件实际完成2次更新，不是把预算缩成2步，更不是GAN基准性能。

检查发现不可接受的核矩阵缓存：原MMD值和XX/XY/YY矩阵逐位一致，但996点拟合的原sigma第一维梯度为0.010763021185994148，缓存梯度为0.0001336451678071171。该缓存优化器被拒绝。生产训练保留原fit函数、原梯度、原RMSProp和完整停止预算，不使用Gram缓存。

原每次NumPy输入的评估调用新增320个图操作和75038368字节常量；反复500轮会持续增长。生产入口仅复用3988点原始输入占位符及原MMD函数图。完整数据下，与作者字面NumPy调用的优化后MMD逐位相同；没有抽样、减少数据、放宽容差或替换MMD公式。两条源MMD辅助函数的对角线惯例不一致也已记录，生产保留发布源码惯例，不能默认为论文定义完全等价。

作者源码60个Git文件全部再次验证且没有编辑。实验只在内存中移除静态证明未被引用的eval导入，避免未调用分类评估模块的Keras依赖阻挡此原MMD训练任务；不存在虚构评估器。作者完整500轮循环及GAN训练函数保持原样，在每轮末增加观察和检查点保存。原TSTR分类、eICU、采用基线的其他数据和配方仍是未完成义务。

每轮记录全量评估数组、原生sigma实际更新数、GAN实际更新数，并保存全部模型/优化器变量、NumPy及Python随机状态。原始素材和数据只读；所有训练输出写入配置登记的新F盘目录。隔离原生环境37项包版本及解释器三个文件哈希冻结，没有修改正在运行的TimeGAN环境。

完整评估通过证据：../rcgan_full_native_mmd_audit_v4/full_original_mnist_mmd_and_sigma_fit.json、independent_full_evaluator_validation.json及原生退出0的execution_receipt。v1、v2、v3失败尝试分别保存；v2诊断AST缺少命名空间绑定的错误在v3独立入口中修正，v4保留原带宽优化后通过。

队列：configs/experiments/fm_rcgan_native_original_mnist500_queue.v1.json；模型配置：configs/models/fm_rcgan_native_original_mnist500.v1.json。launch.json、verified_runtime_observation.json和validation.json记录实际启动、进程身份及测试结果。

没有修改v21数值汇总，没有新增基准性能行。作者等价、严格TAB和全部原论文实验完成标记仍为false。

引用：Esteban, Hyland and Raetsch. Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs. arXiv:1706.02633v2；原作者源码commitf41731b965348259dcd94b0dcb1374d3e1c4ca7d。
