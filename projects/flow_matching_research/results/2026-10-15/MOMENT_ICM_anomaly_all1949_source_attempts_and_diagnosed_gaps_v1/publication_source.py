"""Archive all original source attempts honestly, including322 failures and pending audit."""
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import audit_moment_icm_anomaly_missingness_v2 as audit
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as r

ROOT,read,sha,write=r.ROOT,r.read,r.sha,r.write
previous=read(ROOT/r.producer.CONFIG);jobs=r.producer.verify(previous)
success,failed,original_pins=r.source_partition(previous,jobs)
assert len(success)==1627 and len(failed)==322
failed_map={j['id']:j for j in failed}
audit_config=read(ROOT/audit.CONFIG);audit.verify(audit_config)
audit_base=ROOT/audit_config['state_root'];audit_status=read(audit_base/'status.json')
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_anomaly_all1949_source_attempts_and_diagnosed_gaps_v1'
assert not folder.exists();folder.mkdir(parents=True,exist_ok=False)
validations=[('tmp/MOMENT_ICM_anomaly_pretraining_validation_20261015_v2.xml',5,'preparation_validation.xml'),
    ('tmp/MOMENT_ICM_anomaly_closure_validation_20261015_v1.xml',8,'future_complete_cache_checker_validation.xml'),
    ('tmp/MOMENT_ICM_anomaly_mixed_closure_validation_20261015_v2.xml',6,'future_disjoint_partition_validation.xml'),
    ('tmp/MOMENT_ICM_anomaly_dependency_and_missingness_validation_20261015_v2.xml',8,'dependency_and_missingness_validation.xml')]
for file,count,name in validations:
    suites=list(ET.parse(ROOT/file).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    shutil.copyfile(ROOT/file,folder/name)
history=[]
for file,count,diagnosis in [
    ('tmp/MOMENT_ICM_anomaly_pretraining_validation_20261015_v1.xml',1,
     'New NASA audit used test-file length for scaler extent; original source uses full paired train. Explicit train-row input corrected before full-source registration.'),
    ('tmp/MOMENT_ICM_anomaly_scipy_recovery_validation_20261015_v1.xml',3,
     'Fresh compatible SciPy resolves dependency ImportError but complete unchanged source cubic interpolation leaves NaNs in three real failed files. No repair job registered/launched. Tests now assert nonfinite guard rejection and preserve original failures; no tolerance or source behavior changed.')]:
    p=ROOT/file;assert sum(int(s.get('failures',0)) for s in ET.parse(p).getroot().iter('testsuite'))==count
    history.append(dict(path=file,sha256=sha(p),original_UTF8=p.read_bytes().decode('utf-8'),failures=count,diagnosis=diagnosis))
write(folder/'preserved_pre_registration_validation_failures.json',dict(records=history,
    full_source_original_attempt_failed_cases=322,diagnosed_repair_attempt_launched=False))

entries=[];families={};peak_rss=peak_private=coordinates=0
for job in jobs:
    directory=Path(previous['artifact_root'])/job['family']/job['id']
    if job['id'] in success:
        p=directory/'pretraining_preparation.json';proof=read(p)
        assert proof['passed'] and proof['job']==job and sha(p)==success[job['id']]['sha256']
        assert proof['configuration_sha256']==sha(ROOT/r.producer.CONFIG)
        assert set(proof['splits'])=={'train','val','test'}
        coordinates+=sum(s['time_rows'] for s in proof['splits'].values())
        entry=dict(id=job['id'],family=job['family'],raw=job['raw'],NASA_paired_train_raw=job['train_raw'],
            source_preparation_status='source_worker_complete_all3splits',cache_directory=str(directory),
            source_proof_sha256=sha(p),source_environment=previous['environment'],
            independent_full_cache_closure='pending_registered_readonly_audit',
            splits={name:dict(time_rows=s['time_rows'],source_dunder_len=s['source_dunder_len'],
                source_length_gate_selected=s['source_length_gate_selected'],source_training_windows=s['eligible_stride1_window_count'],
                source_scaler_fit_extent_rows=s['source_scaler_fit_rows'],files=s['files']) for name,s in proof['splits'].items()},
            certified_model_input_ready=False)
    else:
        p=directory/'failed_preserved.json';proof=read(p)
        assert sha(p)==failed_map[job['id']]['original_failure']['sha256'] and proof['message']==r.REASON
        entry=dict(id=job['id'],family=job['family'],raw=job['raw'],NASA_paired_train_raw=job['train_raw'],
            source_preparation_status='original_dependency_ImportError_preserved',original_failure_path=str(p),
            original_failure_sha256=sha(p),source_environment=previous['environment'],
            compatible_environment_probe='residual_nonfinite_found_in_all3_representative_files',
            full322_missingness_audit='registered_and_pending',certified_model_input_ready=False)
    entries.append(entry)
    families.setdefault(job['family'],[]).append(dict(id=job['id'],status=entry['source_preparation_status'],
        original_evidence_path=str(p),sha256=sha(p),original_UTF8=p.read_bytes().decode('utf-8')))
for family,records in families.items():
    p=Path(previous['artifact_root'])/family/'family_preparation.json'
    gp=ROOT/previous['state_root']/'attempts'/family/'guard_receipt.json';g=read(gp)
    assert g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
    peak_rss=max(peak_rss,g['peak_tree_rss_bytes']);peak_private=max(peak_private,g['peak_tree_private_bytes'])
    write(folder/'original_source_evidence'/f'{family}.json',dict(original_UTF8_bytes_recoverable=True,
        source_family_proof_sha256=sha(p),original_family_proof_UTF8=p.read_bytes().decode('utf-8'),
        guard_sha256=sha(gp),original_guard_UTF8=gp.read_bytes().decode('utf-8'),records=records))

index='configs/reproducibility/fm_moment_ICM_anomaly_source_attempt_index.2026-10-15_v1.json'
assert not (ROOT/index).exists()
write(ROOT/index,dict(schema_version=1,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
    producer_configuration=r.producer.CONFIG,missingness_and_independent_success_cache_audit=audit.CONFIG,
    original_sources_attempted=1949,families=18,source_worker_successes=1627,original_dependency_failures=322,
    successful_cache_signal_coordinates=coordinates,successful_source_arrays=14643,
    source_success_is_not_independent_cache_closure=True,complete1949_finite_input_cache=False,
    records=entries,neural_fit_inference_SVM=0,all_original_experiments_complete=False))
shutil.copyfile(ROOT/index,folder/'frozen_all1949_source_attempt_index.json')
for name,source in {
    'frozen_full1949_source_preparation_configuration.json':ROOT/r.producer.CONFIG,
    'frozen_unexecuted_full1949_closure_v1_configuration.json':ROOT/'configs/reproducibility/fm_moment_ICM_PILE_all1949_anomaly_closure.2026-10-15_v1.json',
    'frozen_actual_readonly_audit_configuration.json':ROOT/audit.CONFIG,
    'actual_source_controller_launch.json':ROOT/previous['state_root']/'controller_launch.json',
    'actual_all18_original_family_terminal_status.json':ROOT/previous['state_root']/'status.json',
    'actual_readonly_audit_controller_launch.json':audit_base/'controller_launch.json',
}.items():shutil.copyfile(source,folder/name);assert sha(source)==sha(folder/name)
write(folder/'readonly_audit_status_snapshot.json',audit_status)
package=read(ROOT/r.PACKAGE);package_base=Path(package['directory'])
for name in ['package_verified.json','installed_manifest.json','private_install.log','aria2_download.log']:
    shutil.copyfile(package_base/name,folder/('private_SciPy_'+name))
shutil.copyfile(ROOT/r.PACKAGE,folder/'private_SciPy_registered_acquisition.json')
write(folder/'explicit_reproduction_and_execution_status.json',dict(
    original_full1949_attempts='complete:1627successful_source_exports_and322preserved_dependency_failures',
    complete1949_finite_input_cache='not_complete',
    full1949_closure_v1='registered_not_executed_all1949successful_prerequisite_unsatisfied',
    SciPy1101_private_package='official_wheel_checksum_verified_private_install_full1373file_manifest_base_env_unmodified',
    failed_only_recovery_v2='registration_blocked:compatible_environment_leaves_nonfinite_source_values',
    mixed_complete_closure_v2='unregistered_future_component_not_executed',
    original1627_independent_cache_closure_and_full322_missingness_audit=dict(configuration=audit.CONFIG,
        actual_status_at_archive=audit_status['state'],guarded_controller_registered=True,success_not_yet_certified=True),
    full1949_source_header_and_splits='unaltered_related_source_control',
    strict_finite_preprocessing_adapter='still_required_separately',
    model_training_or_performance_results=0,tests_passed=27,
    test_fixtures_are_plumbing_not_performance=True,all_original_experiments_complete=False))
buffer=io.StringIO(newline='');fields=['family','id','source_preparation_status','source_training_rows','source_training_gate','source_training_windows','raw_path']
writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writeheader()
for e in entries:
    train=e.get('splits',{}).get('train',{})
    writer.writerow(dict(family=e['family'],id=e['id'],source_preparation_status=e['source_preparation_status'],
        source_training_rows=train.get('time_rows'),source_training_gate=train.get('source_length_gate_selected'),
        source_training_windows=train.get('source_training_windows'),raw_path=e['raw']['path']))
(folder/'all1949_source_attempts.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
(folder/'README.md').write_text(f'''# 异常检测分支：全部 1,949 次原来源准备已执行，322 项仍有缺口

18 个家族的全部 1,949 个原来源引用均已实际尝试。1,627 项完成原 train/val/test 三个划分，保存 14,643 个数组、{coordinates:,} 个信号坐标；322 项在原环境中因 Pandas 2.2.2 要求 SciPy≥1.10 而 SciPy 为 1.9.3 失败，全部来自 OPPORTUNITY。18 个工作进程均正常退出、没有资源中止；家族进程退出 0 与每个来源成功是不同概念。OPPORTUNITY 的家族 passed=false、322 条失败均原样保留，没有把它们算作完整缓存。

原 AnomalyDetectionDataset 完整类通过私有 AST 运行，header 推断、时间划分、StandardScaler、三次插值、stride1/长度256、原生 float64 均保留。成功输出的一维信号、原 CSV 记录号、共享全1窗口 mask 与当次原输出逐字节比较；短序列不填充为训练样本，保留原负 __len__。源读取规则仍包括数值首记录被默认 header 推断省略、普通 test 最末行省略、NASA 81 个训练配对及完整训练文件拟合（含内部验证行）。这些是相关来源行为，不等于历史 ICM 实现认证或严格 TAB 规则。

只为诊断失败项，通过代理用 aria2c 下载并校验官方 [SciPy 1.10.1 Windows CPython3.10 wheel](https://pypi.org/project/scipy/1.10.1/)，SHA256 为 fae8a7b898c42dffe3f7361c40d5952b6bf32d10c4569098d276b4c547905ee1。安装到 F 盘独立目录，1373 个安装文件逐个哈希；未修改现有模型/数据环境。3 个真实失败文件在新进程中已能进入原三次插值，但仍产生非有限值；完整有限值保护拒绝保存。不能称这322项已修复，也不能通过丢行、零填充、放宽保护或排除文件使结果虚假完整。

已登记并启动独立保护的完整审计：重新核验 1,627 个成功来源的所有缓存坐标/哈希/原始记录身份，并在兼容环境完整审计 322 项的全部三个划分、原始缺失值及插值/缩放后残留缺失范围。归档时状态为 {audit_status['state']}；该任务保留原数据与既有缓存，沿用 16/21/16 GiB 门槛及 3 GiB 树上限。此包不提前认证该审计完成。针对残留缺失值的明确有限数据适配器和严格预处理对照仍需另行实现与核验。

27 项最终相关检查通过。两份登记前失败 XML 原字节保留：一次新 NASA 审计把 test 长度误作 scaler 范围，在全量登记前纠正；另一次兼容环境测试原假设有限值可立即恢复，被三个真实文件反例否定，现验证非有限输出确实被原保护拒绝。没有放宽容差、原算法热修改或失败训练重试。未启动已知仍不具备前提的全1949缓存闭合与失败项再准备；各候选模块的登记/执行状态单独明确记录。

新 configs 来源尝试索引逐项列出成功缓存、失败原文、哈希、原数据/配对文件、训练长度门槛及窗口数。模型输入完整准备、Tiny/ICM 预训练、推理、SVM 和下游性能均未由本轮新增；性能结果为 0。先前 157 分类、9 长预测、47 Monash 索引保持独立。完整 PILE 采样、分组、epoch/token、Tiny/ICM 实际训练、原生复核及全部下游对比仍在范围；至少 45 篇 / 681 方法记录 / 273 数据任务及后来新增范围不变，完整研究目标未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。大型数据、缓存和独立安装包保留在配置指定的 F 盘。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_attempts=1949,
    original_source_export_successes=1627,preserved_dependency_failures=322,tests_passed=27,
    original_source_peak_RSS_bytes=peak_rss,original_source_peak_private_bytes=peak_private,
    complete1949_finite_cache=False,new_neural_results=0,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/moment_ICM_anomaly_all1949_source_attempts.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',source_configuration=r.producer.CONFIG,attempt_index=index,
    actual_missingness_audit=audit.CONFIG,evidence=folder.relative_to(ROOT).as_posix(),original_source_attempts=1949,
    source_export_successes=1627,preserved_dependency_failures=322,complete1949_finite_cache=False,
    diagnosed_source_nonfinite_after_compatible_environment=True,neural_training_epochs=0,
    current_goal_turn_classification='progress_all1949_original_anomaly_source_attempts_with1627exports_and322diagnosed_gaps',
    all_original_experiments_complete=False))
files=['scripts/flow_matching/prepare_moment_icm_PILE_anomaly_v1.py','scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
    'scripts/flow_matching/recover_moment_icm_anomaly_scipy1101_v2.py','scripts/flow_matching/verify_moment_icm_PILE_anomaly_v2.py',
    'scripts/flow_matching/audit_moment_icm_anomaly_missingness_v2.py','tests/test_fm_moment_icm_PILE_anomaly_v1.py',
    'tests/test_fm_moment_icm_PILE_anomaly_closure_v1.py','tests/test_fm_moment_icm_anomaly_scipy1101_v2.py',
    'tests/test_fm_moment_icm_PILE_anomaly_closure_v2.py','tests/test_fm_moment_icm_anomaly_missingness_v2.py',
    r.producer.CONFIG,audit.CONFIG,r.PACKAGE,index,
    'configs/reproducibility/fm_moment_ICM_PILE_all1949_anomaly_closure.2026-10-15_v1.json']
targets=[*files,folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_anomaly_all1949_attempts_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(original_sources_attempted=1949,source_export_successes=1627,preserved_dependency_failures=322,
    actual_source_cache_signal_coordinates=coordinates,exact_publication_files=len(paths),tests_passed=27,
    full_missingness_and_independent_cache_audit_state=audit_status['state'],complete1949_finite_cache=False,new_neural_results=0)))
