"""Archive the frozen all157 job and a verified live progress checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil
from scripts.flow_matching import prepare_moment_icm_pile_classification_v1 as producer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
config = read(ROOT / producer.CONFIG)
producer.verify(config)
base = ROOT / config['state_root']
launch = read(base / 'controller_launch.json')
process = psutil.Process(launch['pid'])
assert process.create_time() == launch['create_time'] and process.cmdline() == launch['command']
assert launch['configuration_sha256'] == sha(ROOT / producer.CONFIG)
status_bytes = (base / 'status.json').read_bytes()
status = json.loads(status_bytes)
assert status['pid'] == process.pid and status['create_time'] == process.create_time()
assert status['required_datasets'] == 157 and status['completed_count'] == len(status['completed_datasets'])
assert status['completed_count'] > 0 and status['completed_count'] < 157
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_PILE_all157_classification_preparation_started_v1'
folder.mkdir(parents=True, exist_ok=False)
xml = ROOT / 'tmp/MOMENT_ICM_PILE_classification_validation_20261015_v2.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 12
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
for name, path in {
    'frozen_all157_configuration.json': ROOT / producer.CONFIG,
    'prepare_moment_icm_pile_classification_v1.py': ROOT / 'scripts/flow_matching/prepare_moment_icm_pile_classification_v1.py',
    'test_fm_moment_icm_pile_classification_v1.py': ROOT / 'tests/test_fm_moment_icm_pile_classification_v1.py',
    'pretraining_data_validation.xml': xml,
    'actual_controller_launch.json': base / 'controller_launch.json',
}.items():
    shutil.copyfile(path, folder / name)
    assert sha(path) == sha(folder / name)
(folder / 'actual_live_progress_snapshot.json').write_bytes(status_bytes)
originals = {}; complete_cases = coordinates = selected = 0
for name in status['completed_datasets']:
    proof = Path(config['artifact_root']) / name / 'pretraining_preparation.json'
    guard = base / 'attempts' / name / 'guard_receipt.json'
    r, g = read(proof), read(guard)
    assert r['passed'] and r['configuration_sha256'] == sha(ROOT / producer.CONFIG)
    assert g['passed'] and g['exit_code'] == 0 and not g['resource_guard_aborted']
    assert g['proof_sha256'] == sha(proof)
    assert r['job']['dataset'] == name and not r['TEST_signal_values_processed']
    assert not r['model_input_label_values_used'] and not r['record']['label_values_exported']
    assert set(r['record']['files_sha256']) == {'values.npy', 'input_mask.npy', 'original_instance_rows.npy'}
    assert r['record']['instances'] == r['complete_raw_TRAIN_audit']['observed_instances']
    assert r['record']['source_coordinates_copied_and_checked'] == r['record']['instances'] * r['record']['channels'] * 256
    complete_cases += r['record']['instances']; coordinates += r['record']['source_coordinates_copied_and_checked']
    selected += int(r['original_raw_length_selected'])
    originals[name] = dict(source_sha256=sha(proof), guard_sha256=sha(guard),
        original_source_utf8=proof.read_bytes().decode('utf-8'), original_guard_utf8=guard.read_bytes().decode('utf-8'))
    assert hashlib.sha256(originals[name]['original_source_utf8'].encode()).hexdigest() == sha(proof)
    assert hashlib.sha256(originals[name]['original_guard_utf8'].encode()).hexdigest() == sha(guard)
write(folder / 'completed_original_receipts_at_snapshot.json', dict(
    snapshot_only=True, completed_at_snapshot=len(originals), required_total=157,
    original_UTF8_bytes_recoverable=True, records=originals))
failed_xml = ROOT / 'tmp/MOMENT_ICM_PILE_classification_validation_20261015_v1.xml'
failed_suites = list(ET.parse(failed_xml).getroot().iter('testsuite'))
assert sum(int(s.get('failures', 0)) for s in failed_suites) == 3
write(folder / 'preserved_pre_registration_exporter_diagnosis.json', dict(
    original_validation_sha256=sha(failed_xml), original_validation_utf8=failed_xml.read_bytes().decode('utf-8'),
    failures=3, phase='before_config_freeze_and_real_all157_job_launch',
    diagnosis='Source pre-training items have labels=None;downstream classification exporter wrongly required labels.',
    correction='Separate signal-only exporter requests no item labels and writes no label array;original source export unchanged.',
    corrected_validation_passed=12, actual_neural_experiment_failed=False, actual_dataset_attempt_retried=False))
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(), required_classification_candidates=157,
    completed_at_snapshot=len(originals), failed_preserved_at_snapshot=status['failed_preserved'],
    complete_original_TRAIN_cases_at_snapshot=complete_cases, transformed_coordinates_at_snapshot=coordinates,
    raw_length_gate_selected_among_completed_only=selected, controller_exact_handle_live=True,
    original_validation_failure_preserved=True, final_checks_passed=12,
    labels_exported=False, TEST_signal_values_processed=False, original_pretraining_padding_policy=True,
    source_full_TRAIN_scaler_before_validation_split=True, strict_train_only_preprocessing=False,
    actual_pretraining_epochs_executed=0, model_performance_results=0,
    complete_all157_preparation_claim=False, complete_PILE_pretraining_loader_integrated=False,
    other_forecasting_and_anomaly_pretraining_branches_still_required=True,
    all_original_experiments_complete=False)
write(folder / 'launch_and_partial_materialization_summary.json', summary)
(folder / 'README.md').write_text(f'''# 启动完整 PILE 分类分支的 157 个候选数据准备

这轮已为当前相关来源 getter 的全部 157 个分类候选登记新配置并启动真实数据准备进程。登记范围来自完成全 2,441 文件校验的快照，保留来源对 CharacterTrajectories 的明确排除；它与仅覆盖 117 个论文下游评估数据集的缓存不同。此证据包捕获时，控制器 PID/创建时间/完整命令均与真实活动进程一致，已完成 {len(originals)}/157 个候选、{complete_cases:,} 个完整 TRAIN 实例、{coordinates:,} 个变换后坐标。其余任务继续运行；具体时点见原状态快照，不把部分进度称全部完成。

使用固定 MOMENT 源类、解析器及预处理工具，读取完整官方 TRAIN。getter 虽以 TEST 文件为候选引用，来源训练分支在内部构造 TRAIN 路径；没有处理 TEST 信号。预训练使用来源默认 backward pad 和 interpolate downsampling，不套用下游分类脚本的 13 个图像/频谱数据插值例外，因此没有把旧 117 个下游缓存直接当预训练输入。

最初的 3 个实际小型来源检查揭示预训练样本不返回标签，而下游导出器要求标签，导致导出报错。该失败发生在冻结配置及启动全量任务之前，原 XML 按 UTF8 原字节封装保留。已另写无标签导出器，逐坐标/掩码/原 dtype 核对全部通道，保存 values、input_mask 和原实例编号共 3 个数组；不写 labels 文件。之后 12 项检查全部通过，包括三个真实多变量数据集、变长输入、完整实例/通道、只读 TRAIN 及填充策略；没有重试实际全量任务或神经网络实验。

缓存先保留完整 TRAIN 的所有实例和通道。原来源先把通道展平、再按 6/7 切训练/验证，边界可能切开同一实例。为后续附录 D 的多通道模型成对比较，已显式登记按完整实例做 6/7 前缀切分的重建选择，独立通道模型和 ICM 必须共用它；原展平边界数量单独记录，不声称两套切分完全相同。归一化保留来源在完整 TRAIN 上拟合再切验证的规则，包含未来验证样本统计，所以不是严格训练集专属标准化。

即使原时间长度小于 256，所有候选也完整准备并报告。来源的原始长度是否达到 256 单独保存，不能因变换后固定成 256 就把它们悄悄加入来源预训练筛选。后续全语料训练必须明确冻结筛选与分组策略。

每个候选使用独立子进程、固定 Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2 环境、退出码、文件哈希和保护回执。沿用原非神经网络数据任务的 16 GiB 可用内存、21 GiB commit 准入、16 GiB 紧急线和 3 GiB 进程树 RSS/private 上限，既有模型训练和复核的门槛与预算不变；每个候选结束后释放资源槽。原始数据、旧缓存及既有训练进程均未改写。任何真实失败保留，不自动重试、不降低预算或删去候选。

本次为分类分支的真实数据物化，仍需终态确认与独立缓存闭合核验。长预测、Monash、异常检测分支、完整预训练加载器、整个语料的精确步数/token、Tiny/ICM 一整 epoch、最终权重和原生复核、下游预测/SVM/beta/严格预处理对照仍未全部完成。本轮实际神经网络预训练 epoch、新增模型性能结果为 0。研究至少 45 篇论文/681 条方法记录/273 条数据任务及后续新增原文实验的范围保持不变。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。相关来源与历史 ICM 流水线的完全等价性尚未认证。大数组保存于配置指定的 F 盘目录；Git 包仅保存代码、配置、测试和实际证明。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    required_candidates=157, completed_at_snapshot=len(originals), final_checks_passed=12,
    complete_all157_claim=False, model_performance_results=0, all_original_experiments_complete=False,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_PILE_all157_classification_preparation_started.2026-10-15_v1.json'
write(ara, dict(paper_id='moment_long_context', configuration=producer.CONFIG,
    evidence=folder.relative_to(ROOT).as_posix(), controller_handle=launch,
    required_candidates=157, completed_at_snapshot=len(originals), complete_all157=False,
    source_pretraining_default_pad_separate_from_downstream_interpolation=True,
    no_TEST_signal_processing=True, no_model_input_labels=True,
    original_raw_length_gate_explicit=True, original_flatten_first_and_disclosed_casewise_split_separate=True,
    actual_pretraining_epochs=0, model_performance_results=0,
    previous_goal_turn_classification='progress_full117_grouped_coordinate_audit_closed_and_published',
    current_goal_turn_classification='progress_started_full157_actual_pretraining_classification_preparation',
    all_original_experiments_complete=False))
targets = [producer.CONFIG, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/prepare_moment_icm_pile_classification_v1.py',
    'tests/test_fm_moment_icm_pile_classification_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_PILE_classification_started_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), required=157, completed_at_snapshot=len(originals),
    tests_passed=12, failed_at_snapshot=len(status['failed_preserved']), model_performance_results=0)))
