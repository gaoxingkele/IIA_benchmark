# TrajectoryNet 与 RNODE 原始正则函数的数值和梯度核验

直接加载已固定原始作者代码，在2、5、25维、每批1000个解析样本上执行能量和Jacobian函数，并独立核对闭式数值与参数梯度；三种维度全部通过，相关10项测试通过。
TrajectoryNet的sl2int返回批量平均的特征平方和；RNODE的quadratic_cost返回每样本0.5倍特征平方均值。对同一向量场取批量均值后，前者的值及梯度是后者的2×维数倍；5维时相差10倍。
TrajectoryNet的JFrobint返回Jacobian Frobenius范数除以维数的批量均值，既不是平方Frobenius范数，也不是RNODE的context.sqjacnorm随机估计器。后者测试只确认原回调保留context估计器张量及其梯度；不冒称执行了完整原生随机ODE估计器。

由此不能在两个实现之间直接复制相同能量/Jacobian系数，更不能把图像或S-curve系数当成EB历史实验参数。此项核验为实现保真检查，不是原数据集实验，不增加任何训练完成数。
EB精确Jacobian系数仍未知；两个TrajectoryNet协议中各45个能量条件均保留为待执行。完整源训练、真实数据预检、检查点回放、Reg.CNF基线及其他原始范围继续保留。

原论文附录E.1提供S-curve的系数，E.2只列EB能量系数网格，没有据此绑定EB Jacobian系数。作者仓库issue12和issue26也不能视为作者确认的参数答复。
论文：https://proceedings.mlr.press/v119/tong20a/tong20a-supp.pdf
作者讨论：https://github.com/KrishnaswamyLab/TrajectoryNet/issues/12 、 https://github.com/KrishnaswamyLab/TrajectoryNet/issues/26

原45论文/681方法记录/273数据任务/702指标、全部原论文/TAB/新数据实验目标不变，整体未完成；没有修改或停止已运行数值任务。
