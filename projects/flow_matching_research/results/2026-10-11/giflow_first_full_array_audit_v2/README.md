# GiFlow 全部测试数组和技术指标独立核验

Air36 / released_mirror / main / seed393，实际247轮、11856次更新；全14个测试批次的保存预测、真实目标、掩码位置、原始batch×node×time×channel几何及文件哈希均已核验。

独立float64重算：MAE9.6546902115、MSE359.7984835420、MAPE21.5168332493%、sqrt(native mean batch MSE)18.9683547927。与原生float32结果在预先冻结的rtol=1e-5、atol=1e-5容差内一致。MAPE沿用原生abs(target)>1e-6条件，各batch错误不加权平均，未修改算法、数据、阈值、模型或原生结果。

第一次数组检查错误地假设BLNC轴顺序；原生保存格式是BNLC。该失败的v1脚本／配置原样保留。v2只修正审计中的轴检查，数值容差没有放宽。此处是完整数组和指标复核，不是训练模型重推断，也没有增加训练种子。

[完整数组审计](full_array_audit.json)、[校验与源绑定](validation.json)。原生测试参与验证和EMA重置边界继续保留；不声明作者等价或严格TAB成绩。
