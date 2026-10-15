"""Archive the complete guarded grouped-coordinate audit and callable loader."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import audit_moment_icm_full117_channel_data_v1 as auditor

ROOT, read, sha, write = auditor.ROOT, auditor.read, auditor.sha, auditor.write
config = read(ROOT / auditor.CONFIG)
closure = auditor.verify(config)
base = ROOT / config['state_root']
status, guard = read(base / 'status.json'), read(base / 'guard_receipt.json')
proof_path = Path(config['artifact_root']) / 'full117_grouped_coordinate_audit.json'
proof = read(proof_path)
assert status['active_pid'] is None and status['state'] == 'all117_grouped_coordinate_audit_passed'
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['proof_sha256'] == sha(proof_path)
assert proof['passed'] and proof['configuration_sha256'] == sha(ROOT / auditor.CONFIG)
assert proof['datasets'] == 117 and proof['official_splits'] == len(proof['records']) == 234
assert proof['complete_cached_files_rehashed'] == 936 and proof['original_signal_coordinates'] == 267291904
assert proof['grouped_signal_coordinates'] == 309558784 and proof['grouped_case_items'] == 276772
assert proof['neural_fit_inference_SVM'] == 0 and not proof['full_PILE_training_loader_integrated']
assert not proof['strict_train_only_preprocessing'] and not proof['historical_ICM_pipeline_certified']
originals = {r['dataset']: r for r in closure['records']}
for row in proof['records']:
    original = originals[row['dataset']]['records'][row['split']]
    assert row['cached_files_sha256'] == original['file_sha256']
    assert row['original_instances'] == original['instances'] and row['original_channels'] == original['channels']
    assert row['every_coordinate_and_mask_byte_equal_to_bound_source']
    assert row['every_original_case_and_channel_retained'] and not row['label_file_read_as_model_input']
assert sum(row['original_instances'] for row in proof['records']) == 182780
for split, items, batches in [('TRAIN', 126405, 63), ('TEST', 150367, 75)]:
    plan = proof['separate_split_planning_manifests'][split]
    assert plan['grouped_items'] == items and plan['batch_count'] == batches
    assert plan['every_reference_exactly_once'] and plan['same_manifest_required_for_both_models']
    assert not plan['actual_model_consumption_verified'] and not plan['full_PILE_training_epoch']
xml = ROOT / 'tmp/MOMENT_ICM_channel_data_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 18
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full117_grouped_channel_data_v1'
folder.mkdir(parents=True, exist_ok=False)
for name, path in {
    'frozen_grouped_data_configuration.json': ROOT / auditor.CONFIG,
    'moment_icm_channel_data_v1.py': ROOT / 'scripts/flow_matching/moment_icm_channel_data_v1.py',
    'audit_moment_icm_full117_channel_data_v1.py': ROOT / 'scripts/flow_matching/audit_moment_icm_full117_channel_data_v1.py',
    'test_fm_moment_icm_channel_data_v1.py': ROOT / 'tests/test_fm_moment_icm_channel_data_v1.py',
    'channel_data_validation.xml': xml,
    'actual_controller_launch.json': base / 'controller_launch.json',
    'actual_worker_launch.json': base / 'worker_launch.json',
    'actual_terminal_status.json': base / 'status.json',
    'actual_guard.json': base / 'guard_receipt.json',
    'actual_full117_grouped_coordinate_audit.json': proof_path,
    'actual_final_progress.json': Path(config['artifact_root']) / 'progress.json',
}.items():
    shutil.copyfile(path, folder / name)
    assert sha(path) == sha(folder / name)
buffer = io.StringIO(newline='')
fields = ['dataset', 'split', 'original_instances', 'original_channels', 'original_coordinates',
          'grouped_items', 'group_width', 'grouped_coordinates', 'native_dtype', 'grouped_input_stream_sha256']
writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
writer.writeheader()
for row in proof['records']:
    writer.writerow({k: row[k] for k in fields})
(folder / 'all117_grouped_data_coordinate_coverage.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
write(folder / 'complete_grouped_data_summary.json', dict(
    datasets=117, official_splits=234, original_instances=182780, files_rehashed=936,
    original_coordinates=267291904, emitted_grouped_coordinates=309558784, grouped_case_items=276772,
    all_original_cases_channels_and_coordinates_retained=True, oversampled_duplicates_checked=True,
    dtype_and_masks_byte_equal_to_closed_source=True, all_grouped_item_references_once_per_separate_split=True,
    TRAIN_grouped_items=126405, TEST_grouped_items=150367, reference_batch_size_only=2048,
    TRAIN_planning_batches_only=63, TEST_planning_batches_only=75,
    worker_peak_private_bytes=guard['peak_tree_private_bytes'], worker_peak_rss_bytes=guard['peak_tree_rss_bytes'],
    fixed_source_control_normalization=True, strict_train_only=False,
    actual_pretraining_model_epochs=0, new_model_performance_results=0, complete_PILE_loader_integrated=False,
    historical_ICM_pipeline_certified=False, tests_passed=18, all_original_experiments_complete=False))
(folder / 'README.md').write_text('''# ICM 附录 D 多通道读取器与全部 117 分类数据的逐坐标验证

已实现可调用的多通道分组读取器，以及供独立通道模型和 ICM 共用的完整批次引用清单。按照论文附录 D，单变量保持 1 通道，多变量按 8 通道分组，末组不足 8 通道时仅从末组内过采样补齐。原始每个实例的所有通道都保留，禁止先把通道当成不同实例后再混合成跨通道输入。

原文没有给出分组顺序和过采样随机细节，本实现明确冻结为原通道顺序连续分组；末组使用 seed=13 与数据集身份派生的独立 NumPy RandomState 有放回均匀抽样。同一数据集的分组在所有实例和两个官方划分间相同。它们属于披露的重建选择，不能据此声称与作者历史实现逐位一致。正文的超过 8 通道子采样描述保留为另一项待实现对照，没有用它替换附录 D 分组。

已在先前准备好的完整 117 数据集、234 个官方 TRAIN/TEST 划分上实际执行审计：重新校验 936 个保存文件的 SHA256，核对原处理证明和保护回执，再逐个实例/分组/输出通道对原保存坐标做字节相同比较，包含末组过采样的全部重复坐标及完整 padding mask。原 182,780 个实例、267,291,904 个信号坐标均保留；分组输出含 276,772 个实例组、309,558,784 个坐标。所有核验通过，原始/已处理数据均未改写，数据保持来源原生 dtype。

TRAIN 与 TEST 的引用清单分别生成，拒绝跨划分合批和重复数据集身份，按 1/8 通道形状分桶，保留每个桶最后不足一批的数据。每条引用恰好出现一次，模型初始化和掩码 RNG 不影响数据清单；实际模型消费这些清单尚未验证。以相关 MOMENT 来源的 2,048 作为规划批大小时，TRAIN 为 126,405 条实例组/63 批，TEST 为 150,367 条实例组/75 批。这里只是完整 117 个下游分类缓存的批次规划；它不是整个 PILE 的预训练 epoch、实际训练步数，也不确立 ICM 原文未公开的训练批大小。TEST 数据仅用于独立审计，没有加入训练。

读取器不打开标签作为模型输入；审计只校验标签文件原字节哈希，不为预训练读取标签值。预处理保留此前相关 MOMENT 来源规则，包括部分 TEST 文件自身拟合标准化器，因此仍为来源对照，不能称严格训练集专属预处理。该审计验证通道分组和读取的坐标一致性，没有独立重写来源标准化/插值，也没有认证历史 ICM 流水线。

18 项检查覆盖单变量、少于/等于/大于 8 通道、963/1345 高维通道、末组抽样范围、稳定 RNG、实例/掩码/精度、共享模型输入、两个形状桶的完整尾批、划分隔离和错误基数拒绝。真实 117 数据审计在固定 Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2 环境独立执行，退出码 0、资源保护通过。它沿用原数据任务的 16 GiB 可用内存、21 GiB commit 准入、16 GiB 紧急线、3 GiB 进程树上限；所有原神经网络训练与复核的准入和预算保持不变。

本轮神经网络训练、模型推理、SVM 与性能结果均为 0。完整 PILE 的所有分支选择/多通道分组/随机采样、整个预训练的精确步数和 token、模型环境、训练入口、最终权重保存、原生独立复核、预测和 SVM/beta/严格预处理对照仍需继续。该组件适用于后续完整加载器，117 个下游缓存不会替代完整预训练语料。研究至少 45 篇论文、681 条方法/基线/消融、273 条数据任务及后续新增实验的范围保持不变，全部实验尚未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)，第 8 页附录 D。数据来源与原预处理逐文件证明见固定配置引用的上一轮完整 117 数据闭合记录。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=18, actual_datasets_audited=117, complete_grouped_coordinates=309558784,
    actual_model_results=0, all_original_experiments_complete=False,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_full117_grouped_channel_data.2026-10-15_v1.json'
write(ara, dict(paper_id='moment_long_context', configuration=auditor.CONFIG,
    evidence=folder.relative_to(ROOT).as_posix(), callable_paired_channel_loader_prepared=True,
    original_instances=182780, datasets=117, official_splits=234, complete_grouped_coordinate_check_passed=True,
    complete_PILE_pretraining_loader_integrated=False, source_control_not_strict_train_only=True,
    actual_pretraining_epochs=0, model_performance_results=0,
    previous_goal_turn_classification='progress_audited_one_epoch_contract_and_published_callable',
    current_goal_turn_classification='progress_completed_shared_channel_loader_and_full117_real_coordinate_audit',
    all_original_experiments_complete=False))
targets = [auditor.CONFIG, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/moment_icm_channel_data_v1.py',
    'scripts/flow_matching/audit_moment_icm_full117_channel_data_v1.py',
    'tests/test_fm_moment_icm_channel_data_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_full117_channel_data_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), datasets=117, official_splits=234, tests=18,
    original_coordinates=267291904, grouped_coordinates=309558784, neural_model_results=0)))
