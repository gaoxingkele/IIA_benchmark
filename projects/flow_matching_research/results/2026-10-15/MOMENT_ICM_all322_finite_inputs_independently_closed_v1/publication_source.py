"""Archive both actual successful322-source stages and current all-array hashes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as c

ROOT, read, sha, write = c.ROOT, c.read, c.sha, c.write
config = read(ROOT/c.CONFIG); producer, jobs, runtime = c.verify(config)
assert c.preparation_ready(config)
source_path = Path(producer['artifact_root'])/c.producer.REPORT
closure_path = Path(config['artifact_root'])/c.REPORT
source, closed = read(source_path), read(closure_path)
assert source['passed'] and not source['failed_preserved'] and closed['passed']
assert closed['configuration_sha256'] == sha(ROOT/c.CONFIG)
assert closed['sources'] == 322 and closed['cells'] == 966 and closed['arrays'] == 6762
assert closed['all_source_steps_independently_reconstructed_with_same_libraries']
assert closed['all_model_inputs_finite'] and closed['original1627_caches_and322failures_untouched']
assert closed['source_signal_coordinates'] == 10293666 and closed['residual_replaced_coordinates'] == 109347
assert closed['raw_unobserved_coordinates'] == 109624 and closed['valid_reconstruction_targets'] == 10184042
counts = dict(sources=0, raw_files=0, arrays=0, source_signal_coordinates=0, residual_replaced_coordinates=0,
              raw_unobserved_coordinates=0, valid_reconstruction_targets=0)
individual = []
for job, record, source_record in zip(jobs, closed['records'], source['records']):
    assert job['id'] == record['id'] == source_record['id'] and record['raw'] == job['raw']
    assert sha(ROOT/record['raw']['path']) == record['raw']['sha256']
    proof = Path(source_record['proof'])
    assert sha(proof) == source_record['sha256'] == record['source_proof_sha256']
    individual.append(proof); counts['sources'] += 1; counts['raw_files'] += 1
    for split, facts in record['splits'].items():
        assert facts['all7arrays_independently_reconstructed_byte_equal'] and facts['all_model_inputs_finite']
        assert facts['original_source_NaN_bytes_preserved'] and facts['finite_source_coordinates_unchanged']
        assert facts['raw_row_identity_equal'] and facts['raw_missing_and_source_invalid_targets_excluded']
        assert set(facts['files_sha256']) == set(c.NAMES)
        for key, digest in facts['files_sha256'].items():
            assert sha(Path(record['cache_directory'])/split/c.NAMES[key]) == digest
            counts['arrays'] += 1
        for dest, key in [('source_signal_coordinates','signal_coordinates'),('residual_replaced_coordinates','replaced_coordinates'),
                          ('raw_unobserved_coordinates','raw_unobserved_coordinates'),('valid_reconstruction_targets','valid_reconstruction_targets')]:
            counts[dest] += facts[key]
assert counts == dict(sources=322,raw_files=322,arrays=6762,source_signal_coordinates=10293666,
                     residual_replaced_coordinates=109347,raw_unobserved_coordinates=109624,valid_reconstruction_targets=10184042)
validation = ROOT/'tmp/MOMENT_ICM_all322_post_completion_validation_20261015_v2.xml'
previous_tests = ROOT/'tmp/MOMENT_ICM_all322_actual_closure_validation_20261015_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
checks = sum(int(x.get('tests',0)) for x in suites)
assert checks > 8 and all(int(x.get(k,0)) == 0 for x in suites for k in ['failures','errors','skipped'])
assert sum(int(x.get('failures',0)) for x in ET.parse(previous_tests).getroot().iter('testsuite')) == 4
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_all322_finite_inputs_independently_closed_v1'
folder.mkdir(parents=True, exist_ok=False)
files = dict(actual_all322_preparation=source_path, actual_all322_independent_closure=closure_path,
    frozen_producer_configuration=ROOT/c.producer.CONFIG, frozen_cold_metadata_configuration=ROOT/c.metadata.CONFIG,
    frozen_independent_closure_configuration=ROOT/c.CONFIG, frozen_producer_worker=Path(c.producer.__file__),
    frozen_independent_closure_worker=Path(c.__file__), validation=validation,
    post_completion_validation_source=ROOT/'tests/test_fm_moment_icm_finite_control_post_completion_v1.py',
    preserved_one_time_registration_tests_after_completion=previous_tests)
for stage, cfg, proof in [('preparation', runtime, source_path), ('independent', config, closure_path)]:
    base = ROOT/cfg['state_root']; guard = read(base/'guard_receipt.json')
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['proof_sha256'] == sha(proof)
    module = c.producer.MODULE if stage == 'preparation' else c.MODULE
    assert guard['exact_child_handle']['command'] == [producer['python'],'-B','-X','utf8','-u','-m',module,'--worker']
    files[stage+'_actual_guard'] = base/'guard_receipt.json'
    files[stage+'_actual_worker_launch'] = base/'worker_launch.json'
    files[stage+'_terminal_controller_status'] = base/'status.json'
for label, path in files.items():
    target = folder/(label+path.suffix); shutil.copyfile(path,target); assert sha(path) == sha(target)
for proof in individual:
    dest = folder/'actual_source_case_proofs'/(proof.parent.name+'.json')
    dest.parent.mkdir(exist_ok=True); shutil.copyfile(proof,dest); assert sha(proof) == sha(dest)
write(folder/'all_raw_cache_and_original_target_mask_hashes_currently_rechecked.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),**counts, all_raw_and_cache_hashes_match_actual_closure=True,
    same_original_source_library_interpolation=True, independent_lowlevel_interpolation=False,
    neural_training_or_evaluation_performed=False, historical_source_equivalence_certified=False,
    all_original_experiments_complete=False))
write(folder/'post_completion_validation_transition.json',dict(
    original_one_time_registration_tests_failed_after_output_now_exists=4,
    reason='Old pre-science tests called registration() after successful scientific outputs existed;the original safety gate correctly refused replacement',
    preserved_frozen_v3_v4_source_configs_and_tests_unchanged=True,
    post_completion_checks_use_existing_frozen_runtime_and_readonly_pre_guard=True,
    original_registration_refusal_itself_verified=True,scientific_worker_failed=False,
    actual_preparation_and_independent_exit_codes=[0,0],post_completion_tests_passed=checks))
(folder/'README.md').write_text(f'''# Tiny／ICM：322个有限输入对照实际完成并独立闭合

原冷元数据owner已自然完成第一次数值准备，第二独立进程也实际成功退出，无资源中止。覆盖322个OPPORTUNITY来源、966个train/val/test单元、6762个数组。归档前重新哈希全部322原CSV、322个个案证明、6762个完整数组以及两阶段真实保护证明；没有重新训练或重跑已完成数据worker。

独立阶段不用作者数据类，重新从原CSV解析、原库cubic插值、TRAIN拟合缩放、原时间边界和原始观测身份构建所有值，逐坐标/NaN原字节核对全部7类数组：有限输入、原含非有限值输出、原CSV行、原观测mask、有效目标mask、替换mask和输入mask。使用同一冻结pandas/SciPy/sklearn底层库，不宣称独立实现了插值数值内核。

|核验项|实际数量|
|---|---:|
|来源选中信号坐标|10,293,666|
|残余非有限值替换为训练均值标准化零|109,347|
|原始未观测坐标|109,624|
|允许参与重建损失的有效目标|10,184,042|

仅替换残余非有限输入；所有原本有限坐标逐字节不变，原NaN输出完整另存。原未观测目标即使经插值成为有限数，也继续排除出损失；不删除这些时间行。保持来源test切片末端-1规则，没有相对来源再删行；原CSV末行仍在raw文件，未声称它进入来源test缓存。全部1627个原已完成缓存及322个原失败记录均保留。

真实环境Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.10.1/sklearn1.5.2；SciPy来自原登记F盘私有兼容目录，未修改基础环境。两阶段均沿用16GiB物理/21GiB提交准入、16GiB紧急下限和3GiB RSS/私有上限及同一共享CPU锁。{checks}项完成后检查通过，额外测试夹具不是benchmark结果。保留4项旧登记测试在当前完成态下的失败：它们试图再次registration，原防覆盖门禁因真实输出已存在而正确拒绝。冻结v3/v4源码、配置和旧测试均不修改；新增完成后只读检查使用既有冻结runtime/pre_guard，并验证再次登记确实被拒绝。没有科学worker失败或重试。

这是明确改变缺失输入处理的对照，不是原作者历史流程等价证明。全CSVcubic插值仍在切分前发生，严格时间独立性未认证。数据门禁已从等待变为实际通过；完整2086输入绑定、80,850,189项单轮索引、随机mask审计和两模型39,479更新训练仍各需真实完成。这里没有新的NN训练、预测、SVM、异常F1或论文性能结果。至少45篇/681方法记录/273数据任务和新增实验范围保留，整个目标未完成。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[Time Series PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)。大数组保留原F盘路径，全部路径与哈希随完整独立证明保留。
''', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n# Preserve the old one-time registration test failures byte exact.\npreserved_one_time_registration_tests_after_completion.xml whitespace=-blank-at-eol\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_prepared_sources=322,
    actual_independently_closed_arrays=6762,tests_passed=checks,raw_and_cache_hashes_rechecked=True,
    files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_all322_finite_inputs_independently_closed.2026-10-15_v1.json'
write(ara,dict(actual_prepared_sources=322,actual_closed_cells=966,actual_closed_arrays=6762,
    original1627plus322_scope_retained=True,neural_model_results_produced=0,evidence=folder.relative_to(ROOT).as_posix(),all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),'tests/test_fm_moment_icm_finite_control_post_completion_v1.py'];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_all322_actual_closure_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(actual_closed_sources=322,actual_closed_arrays=6762,tests_passed=checks,exact_files=len(paths))))
