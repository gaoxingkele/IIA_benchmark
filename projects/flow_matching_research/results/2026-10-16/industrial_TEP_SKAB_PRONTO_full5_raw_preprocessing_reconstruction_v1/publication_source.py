from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import audit_industrial_full_raw_preprocessing_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration()
rp=Path(c['report_path']);r=read(rp)
assert r['passed'] and r['configuration_sha256']==sha(ROOT/s.CONFIG)
assert r['full_datasets']==5 and [x['dataset'] for x in r['records']]==c['required_datasets']
assert all(x['passed'] and x['all_registered_arrays_bitwise_identical'] and not x['original_or_prepared_data_rewritten'] for x in r['records'])
xml=ROOT/'tmp/industrial_full5_raw_reconstruction_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==13 and all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
sources={}
for record in r['records']:
    for pin in record['raw_source_files']:
        assert sha(ROOT/pin['path'])==pin['sha256']
        if pin['path'] in sources:assert sources[pin['path']]==pin
        sources[pin['path']]=pin
assert len(sources)==61 and sum(x['array_count'] for x in r['records'])==337
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_TEP_SKAB_PRONTO_full5_raw_preprocessing_reconstruction_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
shutil.copyfile(rp,out/'actual_full5_raw_reconstruction.json')
shutil.copyfile(ROOT/s.CONFIG,out/'frozen_reconstruction_configuration.json')
shutil.copyfile(xml,out/'relevant_fresh13_validation.xml')
write(out/'all61_unique_raw_source_bindings.json',dict(sources=list(sources.values()),raw_reference_count=sum(len(x['raw_source_files']) for x in r['records']),
    PRONTO_reuses_three_raw_days_across_rotations=True,all_raw_bytes_preserved=True))
rows=[]
for x in r['records']:
    counts=x['role_points'];positive=x['role_positive_points'];n=counts['test'];p=positive['test']
    rows.append(dict(dataset_or_registered_fold=x['dataset'],train_points=counts['train'],validation_points=counts['validation'],test_points=n,
        train_positive_points=positive['train'],validation_positive_points=positive['validation'],test_positive_points=p,
        test_positive_prevalence=p/n,all_alert_micro_F1_reference=2*p/(n+p),reference_alarm_fraction=1.,
        features=x['feature_count'],train_groups=x['role_group_counts']['train'],validation_groups=x['role_group_counts']['validation'],
        test_groups=x['role_group_counts']['test'],raw_file_references=len(x['raw_source_files']),exactly_reconstructed_arrays=x['array_count'],
        label_assisted_train_validation_normal_selection=x['normal_selection_is_label_assisted'],
        validation_includes_native_anomalies=x['validation_contains_native_anomalies'],
        feature_coordinates=sum(counts.values())*x['feature_count'],array_payload_elements=sum(a['elements'] for a in x['all_array_payloads']),
        all_array_bytes_identical=True,new_complete_model_performance_results=0))
with (out/'all5_group_roles_label_populations_and_raw_reconstruction.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(out/'execution_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),source_report_path=str(rp),source_report_sha256=sha(rp),
    unique_raw_files=61,registered_dataset_fold_tasks=5,physical_dataset_families=['TEP classic','SKAB','PRONTO'],
    complete_archive_arrays=337,full_feature_coordinates=sum(x['feature_coordinates'] for x in rows),
    PRONTO_rotations_not_independent_datasets=True,all_original_and_prepared_files_preserved=True,
    production_loaders_and_preparation_functions_not_invoked=True,NumPy_and_pandas_dependency_independence_not_claimed=True,
    new_training_updates=0,new_neural_forwards=0,new_benchmark_performance_results=0,
    original_scope=r['original_scope'],all_original_experiments_complete=False))
table='\n'.join(f"| {x['dataset_or_registered_fold']} | {x['train_points']} | {x['validation_points']} ({x['validation_positive_points']}异常) | {x['test_points']} | {x['test_positive_prevalence']:.4%} | {x['all_alert_micro_F1_reference']:.6f} | {x['exactly_reconstructed_arrays']} |" for x in rows)
(out/'README.md').write_text(f'''# TEP、SKAB与PRONTO三折：完整原始数据预处理重建

从61份不重复原始文件重新计算5个登记任务，TEP classic、SKAB及PRONTO三个留出日，全部337个归档数组的dtype、shape和全部内容字节相同。
独立实现原始文件选择、分组与区间、标签、训练独占中位数修复、均值/标准差及恒定通道规则，完整特征值、时间、源行号和Fault字符串均逐项核对。
源文件和准备归档在核验前后哈希一致，没有写回任何原始数据或现有NPZ。全部任务均保留完整测试点和尾部。
未调用原数据加载器或原prepare/normalize函数；SKAB/PRONTO的CSV解析仍共享pandas，算术共享NumPy，不宣称软件库层面的独立性。

| 登记任务 | 训练点 | 验证点 | 测试点 | 测试异常占比 | 全点告警micro F1参考 | 完整数组 |
|---|---:|---:|---:|---:|---:|---:|
{table}

全点告警参考固定报出100%测试点，只展示标签比例对应的F1，并非相同告警预算的竞争算法或新增模型性能。
TEP测试21个完整故障实验、52变量，注入从零基160开始；正常train d00与独立正常validation d00_te分别使用，测试阳性比例83.33%得到重建确认。
SKAB保留33个完整测试实验与8个传感器，other/1.csv作为预选验证实验含188/745个原始异常点。
因此SKAB验证1%分位数针对混合验证人口，不代表正常点1%误报保证；既有原始结果不能因其假定正常而被重新解释。

PRONTO每折的训练日、验证日与测试日完全不同。训练/验证仍使用原始Fault标签选取所有连续Normal区间，测试日完整保留；该过程是明确的标签辅助正常选择。
12个原报警字段完整检查后排除，保留一致的17个过程通道与Fault评价标签；连续正常片段之间不跨越故障间隔。
三个折复用同一组原始日期，不能视作三个独立数据集或独立重复；所有源角色、点数和原始坐标分别公开。
PRONTO day2阳性率及全点告警参考尤其高，比较F1时必须同时报告阳性率、排名质量、正常误报和评价协议。

全部337数组、{sum(x['feature_coordinates'] for x in rows):,}个登记角色特征坐标逐字节核验；跨PRONTO折重复的原始正常点不冒充独立观测。
13项相关边界检查通过，包括仅TRAIN拟合、非有限修复、来源重叠、缺失角色/特征和尾部/标签字节变化拒绝。
这是原始预处理步骤的验收，新增模型训练、神经网络前向和基准性能结果均为0。

此证据补充先前MOMENT工业评估包的TEP/SKAB原始输入完整重建；SWAN另属其他输入轨道，不在本包覆盖范围内。
既有模型/消融的种子、超参数、阈值、资源门槛和旧报告均保持。经典TEP不能抵扣未完成的多模式TEP任务，Tiny/ICM、历史论文复现及完整TAB训练策略分别验收。
全部45篇/681方法记录/273数据任务/702论文标量和原论文/TAB/新数据范围继续保留，整体尚未完成。

数据来源与原协议DOI保留于冻结配置：TEP 10.1016/0098-1354(93)80018-I，SKAB 10.1007/s41060-022-00355-4，PRONTO 10.5281/zenodo.1341583。
''',encoding='utf-8',newline='\n')
shutil.copyfile(ROOT/(s.MODULE.replace('.','/')+'.py'),out/'reconstruction_source.py')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_full5_raw_preprocessing_reconstruction.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),registered_dataset_fold_tasks=5,
    complete_raw_reconstructed_arrays=337,unique_raw_files=61,new_model_performance_results=0,all_original_experiments_complete=False))
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
files=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
write(ROOT/'tmp/industrial_full5_raw_preprocessing_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(out),files=len(files),registered_dataset_fold_tasks=5,raw_files=61,arrays=337,summary=rows)))
