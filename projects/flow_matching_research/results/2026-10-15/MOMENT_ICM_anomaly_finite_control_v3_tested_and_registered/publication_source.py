"""Archive tested explicit finite-data control and actual prerequisite-gated owner."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p
from scripts.flow_matching import run_moment_icm_finite_control_cold_metadata_v4 as metadata

ROOT,read,sha,write=p.ROOT,p.read,p.sha,p.write
config=read(ROOT/p.CONFIG);jobs=p.verify(config)
runtime=read(ROOT/metadata.CONFIG);metadata.pre_guard(runtime)
old_base=ROOT/config['state_root'];base=ROOT/runtime['state_root']
handle=read(base/'controller_launch.json');status=read(base/'status.json')
owner=psutil.Process(handle['pid']);assert abs(owner.create_time()-handle['create_time'])<.001
assert owner.cmdline()==handle['command']
assert handle['configuration_sha256']==sha(ROOT/metadata.CONFIG)
validation=ROOT/'tmp/MOMENT_ICM_anomaly_finite_control_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==8
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
metadata_validation=ROOT/'tmp/MOMENT_ICM_anomaly_finite_cold_metadata_validation_20261015_v1.xml'
suites=list(ET.parse(metadata_validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==5
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
audit_config=read(ROOT/p.audit.CONFIG);audit_base=ROOT/audit_config['state_root']
audit_path=Path(audit_config['artifact_root'])/p.audit.REPORT;audit_proof=read(audit_path);audit_guard=read(audit_base/'guard_receipt.json')
assert audit_proof['passed'] and audit_proof['original_successful_caches_independently_closed']==1627
assert audit_proof['source_audits']==audit_proof['sources_with_residual_nonfinite']==322
assert audit_proof['complete_original_cache_arrays']==14643 and audit_proof['residual_nonfinite_coordinates']==109347
assert audit_guard['passed'] and audit_guard['exit_code']==0 and not audit_guard['resource_guard_aborted']
assert audit_guard['proof_sha256']==sha(audit_path)
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_anomaly_finite_control_v3_tested_and_registered'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {
    'frozen_all322_finite_control_configuration.json':ROOT/p.CONFIG,
    'prepare_moment_icm_anomaly_finite_control_v3.py':ROOT/'scripts/flow_matching/prepare_moment_icm_anomaly_finite_control_v3.py',
    'test_fm_moment_icm_anomaly_finite_control_v3.py':ROOT/'tests/test_fm_moment_icm_anomaly_finite_control_v3.py',
    'actual_controller_launch.json':base/'controller_launch.json',
    'validation_8_checks.xml':validation,
    'frozen_cold_metadata_v4_configuration.json':ROOT/metadata.CONFIG,
    'cold_metadata_v4_entry.py':ROOT/'scripts/flow_matching/run_moment_icm_finite_control_cold_metadata_v4.py',
    'test_cold_metadata_v4.py':ROOT/'tests/test_fm_moment_icm_finite_control_cold_metadata_v4.py',
    'validation_5_cold_metadata_checks.xml':metadata_validation,
    'preserved_failed_v3_metadata_launch.json':old_base/'controller_launch.json',
    'preserved_failed_v3_metadata_stderr.log':old_base/'controller.stderr.log',
    'actual_full1627_cache_closure_and322_missingness_audit.json':audit_path,
    'actual_missingness_audit_guard.json':audit_base/'guard_receipt.json',
    'actual_missingness_audit_worker_launch.json':audit_base/'worker_launch.json',
    'actual_missingness_audit_terminal_status.json':audit_base/'status.json',
}.items():shutil.copyfile(path,folder/name);assert sha(path)==sha(folder/name)
write(folder/'actual_controller_status_snapshot.json',status)
write(folder/'implementation_and_execution_status.json',dict(configuration=p.CONFIG,required_sources=322,
    original1627_successful_exports_preserved=True,original322_dependency_failure_proofs_preserved=True,
    source_raw_or_original_cache_modified=False,explicitly_different_preprocessing_track=True,
    real_source_validation_cases=[dict(id=jobs[i]['id'],raw=jobs[i]['raw'],all3splits_tested=True) for i in [0,160,321]],
    validation_cases_are_full322_preparation_completions=False,tests_passed=13,
    pre_science_v3_metadata_failure_preserved=True,old_scientific_worker_launched=False,
    unchanged_data_worker_cold_metadata_entry=metadata.CONFIG,
    independent_original_cache_sources_closed=1627,independent_original_cache_arrays=14643,
    complete_missingness_source_audits=322,residual_nonfinite_source_coordinates=109347,
    actual_owner=dict(pid=handle['pid'],create_time=handle['create_time'],state=status['state']),
    full322_preparation_passed=False,independent_full322_cache_closure=False,
    target_validity_aware_reader_and_reconstruction_loss_mask_implemented=True,
    complete_PILE_neural_training_runner_integration=False,new_neural_performance_results=0,
    strict_temporal_independence_certified=False,historical_source_equivalence_certified=False,
    all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 残留缺失值的明确预处理对照 v3：实现、验证及登记

实现只针对原失败的322个OPPORTUNITY来源，原1,627成功缓存、322条依赖失败记录、原数据与原类代码全部保留。它是单独的有限输入预处理对照，不是未经修改的历史论文复现。

按原类运行三次插值、划分及StandardScaler，在每个原划分保存完整原输出（含NaN字节）。有限原坐标全部逐字节不变；仅将残留非有限标准化输入替换为0，即已拟合训练均值对应值。没有额外丢时间点、改模型/缺失率/训练预算、修改原source cache或放宽有限值保护。训练拟合统计若仍不合法（例如全部缺失）直接拒绝，不凭空创造统计量。

原始缺失点和原输出非有限点均不能作重建目标。额外保存原信号观测mask、残留替换mask、有效目标mask，另有完整原输出、有限输入、原CSV行号和原窗口padding mask，共7个数组/划分。原有pad mask与有效目标mask不同。新完整256点窗口读取器返回各项身份；损失mask保留MOMENT的1=输入保留、0=预测约定，显式与有效目标及pad mask取交集，避免将填补0或原始缺失点当作真值。

13项检查通过：8项数据检查含3个真实来源，分别核对全部train/val/test、完整有限坐标的字节、所有原CSV行和有效目标身份、首中末窗口及覆写拒绝；另验证符号零、原NaN保留、原缺失但已插值为有限的点仍不成为有效目标、全缺失/无效训练统计拒绝和mask约定。5项冷调度检查验证预检查保持资源API可冷启动、不能替换工作器或放松门槛、已有科学任务不得按元数据故障重试。合成反例只验证逻辑，不是性能。真实3例也不是全部322项准备完成。

前置全量审计已真实完成：1,627个原成功来源的14,643个数组、133,589,346个原信号坐标独立核验通过；原训练门槛选中全部1,627个来源、72,417,674个训练窗口。兼容环境完整审计322个失败来源的966个原划分、10,293,666个输出坐标，322个均残留非有限值，合计109,347个。完整实际证明及无资源中止的成功回执随包保留。该审计成功表示故障被完整查清，不表示322项已修复。

v3首次元数据入口在前置审计已通过时重复安装冷资源API，故在启动任何科学工作器前报错退出；原启动记录和stderr保留，没有旧worker_launch或数据输出。新的独立v4调度只修复这一初始化路径，实际执行的仍是冻结v3数据工作器、相同322项、变换、dtype及16/21/16 GiB门槛、3 GiB上限与共享锁。没有修改原科学代码、启动第二次科学尝试或绕过MOMENT复核。v4控制器实际存活，状态见快照；尚没有322项全量准备或独立闭合成功证明。

原完整CSV插值仍在划分前，原header推断及普通test尾行省略仍保留，因此不称严格因果/时间独立或TAB认证。新的有效目标读取与损失mask函数已实现；完整PILE神经训练器尚未接入它们，不得只使用有限输入而遗漏目标mask。后续还需全量实际准备、独立逐坐标核验、全部数据分支采样/分组/epoch与token核算、Tiny/ICM训练及完整下游评估。新神经性能结果为0。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。精确322项原始来源和已有修复环境哈希在冻结配置内。所有新大数组写到配置指定F盘，原始范围至少45篇/681方法记录/273数据任务及新增实验保持，完整研究目标未完成。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=13,
    actual_full322_preparation_complete=False,new_neural_results=0,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_anomaly_finite_control.2026-10-15_v3.json'
write(ara,dict(paper_id='moment_long_context',configuration=p.CONFIG,evidence=folder.relative_to(ROOT).as_posix(),
    required_source_cases=322,real_source_validation_cases=3,all3splits_of_each_validation_case_checked=True,
    original_finite_values_byte_identical=True,valid_target_reader_and_loss_mask_implemented=True,
    actual_full322_cache_preparation_complete=False,full_neural_training_integration_complete=False,
    historical_source_equivalence_certified=False,all_original_experiments_complete=False))
targets=[p.CONFIG,metadata.CONFIG,'scripts/flow_matching/prepare_moment_icm_anomaly_finite_control_v3.py',
    'scripts/flow_matching/run_moment_icm_finite_control_cold_metadata_v4.py',
    'tests/test_fm_moment_icm_finite_control_cold_metadata_v4.py',
    'tests/test_fm_moment_icm_anomaly_finite_control_v3.py',folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_anomaly_finite_control_v3_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(tests_passed=13,real_source_validation_cases=3,required_sources=322,independent_original_caches_closed=1627,
    all322_missingness_audits_completed=True,actual_owner_state=status['state'],exact_publication_files=len(paths),full322_complete=False)))
