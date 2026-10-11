完整范围评价增量核验与发布
========================

本轮继续执行原定保存分数的VUS与Affiliation评价，将六个算法—数据集组的18项完整评价纳入原有指标槽。
共核验15,256,045个测试点（跨实验累计），包括Independent FM／CICIDS五种子、Reflow两阶段／CICIDS五种子、Reflow三阶段／PSM五种子，以及MOMENT-small release512／CICIDS、PRONTO Day2、Day3各一个固定发布权重。
没有新增训练种子，没有重复评价冒充独立训练。

证据与变化范围
--------------

independent_full_range_increment.json保留18项原生评价的完整记录、源模型与分数SHA、原任务与种子身份、原始及预处理数据校验记录和种子统计。
核验全部原实体、每一个测试时间点、与原预处理数据逐点相同的标签、有限分数、250个阈值、从0至两倍截断中位事件长度的每个整数缓冲区。
独立重算VUS与Affiliation的实体宏平均、定义/未定义计数；种子统计使用样本STD、SE和Student-t95%区间，单固定权重不生成种子不确定性。
Affiliation只读取原验证1%阈值字段；原测试标签ratio诊断保持独立，不参与本次增量。

相对v24，本轮更新18项既有范围指标，其中12项由缺失变为实测。
独立差异核验在independent_range_publication_review.json：原7,283项指标槽和顺序不变，其余7,265条记录的每个字段及采集时间一致。
原点级P/R/F1、AUROC、AP、其他任务与作者协议均未重新采集。
最终完整表为../all_algorithm_dataset_metrics_v26，包含1,805项实测指标和5,478项缺失指标。

验证与失败尝试
--------------

validation.json／validation.txt记录25项通过，覆盖种子重复、预算改变、已有种子丢失、诊断混入、单种子伪不确定性、原整数缓冲区、未定义计数、目录创建与防覆盖，以及既有范围数学和合并逻辑。
validation_v26.json／validation_v26.txt记录9项通过，独立检查原指标槽和顺序、增量范围、协议、预算、认证和其他指标时间不变。
共34项相关验证通过。

最初发布调用在完成18项输入核验后因证据父目录未创建而失败，未产生指标目录，也未停止模型或评价工作进程。
preserved_first_publication_failure.json记录失败；failed_first_publisher_source.py和failed_first_test_source.py逐字节匹配首次失败登记中的源SHA。
修复后采用新登记fm_full_range_increment_publication.2026-10-11_v25.v2.json，原登记保留为失败历史。
归档首次失败源时另有准备期断言失败及随后未找到尚未创建的最终登记，保留在preserved_final_registration_preparation_failure.json；原实验结果没有改变。

v25成功产生数值表，但继承了历史前缀一致旗标，该旗标不适用于v25与v24比较。
保留v25全部输出，独立核验差异并发布v26。v26所有数值导出和逐行时间逐字节保留v25，当前一致旗标仅表示v26与其直接前版v25一致。
本轮新增12项实测指标发生在v24到v25的数值增量；v26没有第二次新增指标或训练。

原范围仍为45篇论文、681项方法/基线/消融记录、273项原始数据任务及采用基线；所有实验尚未完成。
完整TAB训练框架及作者论文等价未认证。该核验不把新增范围评价解释成新模型训练或原论文效果复现成功。
