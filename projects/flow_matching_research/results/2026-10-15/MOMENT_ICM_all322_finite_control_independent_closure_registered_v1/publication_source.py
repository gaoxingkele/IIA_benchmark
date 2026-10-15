"""Publish a fully scoped/tested actual independent closure queue, without success claims."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as v

ROOT,read,sha,write=v.ROOT,v.read,v.sha,v.write
config=read(ROOT/v.CONFIG);previous,jobs,runtime=v.verify(config)
base=ROOT/config['state_root'];handle=read(base/'controller_launch.json');state=read(base/'status.json')
proc=psutil.Process(handle['pid']);assert abs(proc.create_time()-handle['create_time'])<.001 and proc.cmdline()==handle['command']
assert handle['configuration_sha256']==sha(ROOT/v.CONFIG)
source_state=read(ROOT/runtime['state_root']/'status.json')
validation=ROOT/'tmp/MOMENT_ICM_all322_finite_closure_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==4
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_all322_finite_control_independent_closure_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {
    'frozen_all322_independent_closure_configuration.json':ROOT/v.CONFIG,
    'verify_moment_icm_anomaly_finite_control_v1.py':ROOT/'scripts/flow_matching/verify_moment_icm_anomaly_finite_control_v1.py',
    'test_fm_moment_icm_anomaly_finite_control_closure_v1.py':ROOT/'tests/test_fm_moment_icm_anomaly_finite_control_closure_v1.py',
    'actual_independent_controller_launch.json':base/'controller_launch.json',
    'validation_4_checks.xml':validation,
}.items():shutil.copyfile(path,folder/name);assert sha(path)==sha(folder/name)
write(folder/'independent_controller_status_snapshot.json',state)
write(folder/'required_upstream_preparation_status_snapshot.json',source_state)
write(folder/'implementation_and_execution_status.json',dict(required_sources=322,required_splits=966,required_arrays=6762,
    prerequisite_configuration=v.producer.CONFIG,actual_producer_guard_configuration=v.metadata.CONFIG,
    independent_closure_configuration=v.CONFIG,source_data_citation=config['data_citation'],
    exact_independent_values_NaN_bytes_raw_rows_and_target_masks_required=True,
    same_frozen_interpolation_scaler_libraries_used=True,independent_lowlevel_library_implementation=False,
    actual_live_owner=dict(pid=handle['pid'],create_time=handle['create_time'],state=state['state']),
    unit_tests_passed=4,real_source_validation_cases=3,all3splits_each_real_case_verified=True,
    changed_target_mask_with_updated_hash_rejected=True,resource_pre_guard_remains_cold=True,
    full322_source_preparation_verified=False,full322_independent_closure_complete=False,
    new_neural_results=0,historical_source_equivalence_certified=False,all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 322项有限输入对照：独立全量核验器已验证并登记

登记范围为原322个失败来源、全部966个划分和6762个数组。等待上游v3数据工作器在v4冷调度保护下的真实成功报告及退出保护回执；不是检查点或文件存在就判成功，也不把本轮3例测试算作322项全量闭合。

独立步骤不调用原数据类或其修复函数，而按完整原CSV重新构建header推断、三次插值、训练前缀StandardScaler及原60/10/30时间范围（保留test尾行省略）。使用同一冻结Pandas/SciPy/sklearn库，不声称独立重写底层插值库。重新核验原始文件SHA256，逐坐标比较保存的原NaN字节、有限原值、训练均值零替换、原CSV记录号、原观测mask、有效目标mask、残留替换mask及pad mask；核对原全322只读审计的非有限值范围/数量和拟合统计。全部值字节精确比较，没有另选或放松数值容差。

4项检查通过：3个真实来源分别核对全部3个划分与7个数组，另验证篡改有效目标mask后即使重新计算哈希仍被拒绝；冷启动检查确认前置检查没有提前初始化资源API。合成/局部检查不是全量缓存或性能结果。

实际独立控制器已启动并保持原前置条件及16/21/16 GiB门槛、3 GiB上限、共享锁。原始数据、1,627个已闭合成功缓存、322条原依赖失败和v3失败元数据入口均保留。此包只证明实现、验证及队列登记，尚不证明全322有限输入对照准备/独立闭合完成。

这是明确改变缺失值处理的输入对照；保留原完整CSV插值的未来依赖，不能称历史作者等价、严格因果或TAB认证。有效目标mask还必须接入完整PILE神经训练循环；神经预训练、推理、SVM和新性能结果为0。至少45篇/681方法记录/273数据任务及新增范围不变，全部研究目标未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。数据：[AutonLab/Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)，精确快照及全部原始文件索引/哈希在冻结配置。大数组保留F盘。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=4,real_source_validation_cases=3,
    actual_full322_cache_closed=False,files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_all322_finite_control_closure_registered.2026-10-15_v1.json'
write(ara,dict(configuration=v.CONFIG,evidence=folder.relative_to(ROOT).as_posix(),required_sources=322,required_arrays=6762,
    actual_owner_pid=handle['pid'],actual_owner_create_time=handle['create_time'],tests_passed=4,
    independent_math_and_original_NaN_bytes_verifier_implemented=True,actual_full322_closure_complete=False,
    new_neural_results=0,all_original_experiments_complete=False))
targets=[v.CONFIG,'scripts/flow_matching/verify_moment_icm_anomaly_finite_control_v1.py',
    'tests/test_fm_moment_icm_anomaly_finite_control_closure_v1.py',folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_all322_finite_closure_registration_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(required_sources=322,arrays=6762,tests_passed=4,actual_state=state['state'],publication_files=len(paths),full322_closed=False)))
