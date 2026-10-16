from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import audit_trajectorynet_rnode_regularizer_semantics_v1 as s
ROOT,read,sha,write=s.ROOT,s.read,s.sha,s.write
c=read(ROOT/s.CONFIG);s.verify(c)
path=Path(c['artifact_root'])/'original_regularizer_value_and_gradient_conformance.json';r=read(path)
assert r['passed'] and r['configuration_sha256']==sha(ROOT/s.CONFIG)
assert [v['dimensions'] for v in r['records']]==[2,5,25] and all(v['batch']==1000 and v['optimizer_updates']==0 for v in r['records'])
assert not r['full_EB_training_complete'] and not r['benchmark_performance_claim']
tests=Path('F:/aicoding/IIA_Data/experiments/TrajectoryNet_RNODE_regularizer_semantics_20261016_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==10
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TrajectoryNet_RNODE_regularizer_conformance_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8')
shutil.copyfile(path,packet/'actual_original_function_value_gradient_report.json')
shutil.copyfile(tests,packet/'validation_10_cases.xml')
for label,p in c['original_regularizers'].items():shutil.copyfile(p,packet/(label+'_original_regularizers.py'))
write(packet/'source_binding.json',dict(original_regularizers=c['original_regularizers'],original_sha256=c['original_sha256'],source_receipts=c['source_receipts']))
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],actual_analytic_cases=3,tests_passed=10,
    new_trained_models=0,new_completed_original_energy_models=0,EB_Jacobian_coefficient_certified=False,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# TrajectoryNet 与 RNODE 原始正则函数的数值和梯度核验

直接加载已固定原始作者代码，在2、5、25维、每批1000个解析样本上执行能量和Jacobian函数，并独立核对闭式数值与参数梯度；三种维度全部通过，相关10项测试通过。
TrajectoryNet的sl2int返回批量平均的特征平方和；RNODE的quadratic_cost返回每样本0.5倍特征平方均值。对同一向量场取批量均值后，前者的值及梯度是后者的2×维数倍；5维时相差10倍。
TrajectoryNet的JFrobint返回Jacobian Frobenius范数除以维数的批量均值，既不是平方Frobenius范数，也不是RNODE的context.sqjacnorm随机估计器。后者测试只确认原回调保留context估计器张量及其梯度；不冒称执行了完整原生随机ODE估计器。

由此不能在两个实现之间直接复制相同能量/Jacobian系数，更不能把图像或S-curve系数当成EB历史实验参数。此项核验为实现保真检查，不是原数据集实验，不增加任何训练完成数。
EB精确Jacobian系数仍未知；两个TrajectoryNet协议中各45个能量条件均保留为待执行。完整源训练、真实数据预检、检查点回放、Reg.CNF基线及其他原始范围继续保留。

原论文附录E.1提供S-curve的系数，E.2只列EB能量系数网格，没有据此绑定EB Jacobian系数。作者仓库issue12和issue26也不能视为作者确认的参数答复。
论文：https://proceedings.mlr.press/v119/tong20a/tong20a-supp.pdf
作者讨论：https://github.com/KrishnaswamyLab/TrajectoryNet/issues/12 、 https://github.com/KrishnaswamyLab/TrajectoryNet/issues/26

原45论文/681方法记录/273数据任务/702指标、全部原论文/TAB/新数据实验目标不变，整体未完成；没有修改或停止已运行数值任务。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TrajectoryNet_RNODE_regularizer_conformance.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),tests_passed=10,new_trained_models=0,
    unresolved_EB_Jacobian_coefficient=True,all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODEL,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/TrajectoryNet_RNODE_regularizer_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),new_trained_models=0,actual_analytic_cases=3)))
