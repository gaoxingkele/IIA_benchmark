"""Archive all9 original-source preparations, full54cell closure and cache index."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import verify_moment_icm_PILE_long_forecast_v1 as verifier

ROOT, read, sha, write = verifier.ROOT, verifier.read, verifier.sha, verifier.write
config = read(ROOT / verifier.CONFIG); previous, jobs = verifier.verify(config)
base = ROOT / config['state_root']; guard = read(base / 'guard_receipt.json')
status = read(base / 'status.json')
assert status['state'] == 'all9_tracks_splits_independent_closure_passed' and status['active_pid'] is None
proof_path = Path(config['artifact_root']) / 'full9_forecast_cache_closure.json'; proof = read(proof_path)
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['proof_sha256'] == sha(proof_path) and proof['configuration_sha256'] == sha(ROOT / verifier.CONFIG)
assert proof['passed'] and proof['sources'] == len(proof['records']) == 9 and proof['cells'] == 54 and proof['array_files'] == 162
assert proof['full_saved_signal_coordinates'] == 26640764 and proof['neural_training_inference_SVM'] == 0
assert not proof['historical_ICM_pipeline_certified'] and not proof['strict_temporal_independence_certified']
source_base = ROOT / previous['state_root']; source_status = read(source_base / 'status.json')
assert source_status['completed_count'] == 9 and not source_status['failed_preserved']
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_PILE_all9_long_forecast_cache_closed_v1'
folder.mkdir(parents=True, exist_ok=False)
for name, path in {
    'frozen_all9_preparation_configuration.json': ROOT / verifier.producer.CONFIG,
    'frozen_all54cell_closure_configuration.json': ROOT / verifier.CONFIG,
    'prepare_moment_icm_PILE_long_forecast_v1.py': ROOT / 'scripts/flow_matching/prepare_moment_icm_PILE_long_forecast_v1.py',
    'verify_moment_icm_PILE_long_forecast_v1.py': ROOT / 'scripts/flow_matching/verify_moment_icm_PILE_long_forecast_v1.py',
    'test_fm_moment_icm_PILE_long_forecast_v1.py': ROOT / 'tests/test_fm_moment_icm_PILE_long_forecast_v1.py',
    'test_fm_moment_icm_PILE_long_forecast_closure_v1.py': ROOT / 'tests/test_fm_moment_icm_PILE_long_forecast_closure_v1.py',
    'actual_full9_original_controller_launch.json': source_base / 'controller_launch.json',
    'actual_full9_original_terminal_status.json': source_base / 'status.json',
    'actual_full54cell_independent_closure.json': proof_path,
    'actual_closure_guard.json': base / 'guard_receipt.json',
    'actual_closure_controller_launch.json': base / 'controller_launch.json',
    'actual_closure_worker_launch.json': base / 'worker_launch.json',
    'actual_closure_terminal_status.json': base / 'status.json',
}.items():
    shutil.copyfile(path, folder / name); assert sha(path) == sha(folder / name)
checks = 0
for name, path, count in [
    ('source_preparation_validation.xml', 'tmp/MOMENT_ICM_PILE_long_forecast_validation_20261015_v1.xml', 5),
    ('independent_cache_closure_validation.xml', 'tmp/MOMENT_ICM_PILE_long_forecast_closure_validation_20261015_v1.xml', 6)]:
    suites = list(ET.parse(ROOT / path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == count
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
    checks += count; shutil.copyfile(ROOT / path, folder / name)
originals = {}; cells = []; maximum_RSS = maximum_private = 0
for row, job in zip(proof['records'], jobs):
    assert row['dataset'] == job['dataset']; name = row['dataset']
    source = Path(previous['artifact_root']) / name / 'pretraining_preparation.json'
    receipt = source_base / 'attempts' / name / 'guard_receipt.json'
    record, g = read(source), read(receipt)
    assert g['passed'] and g['exit_code'] == 0 and not g['resource_guard_aborted'] and g['proof_sha256'] == sha(source)
    assert sha(source) == row['source_proof_sha256'] and sha(receipt) == row['source_guard_sha256']
    maximum_RSS = max(maximum_RSS, g['peak_tree_rss_bytes']); maximum_private = max(maximum_private, g['peak_tree_private_bytes'])
    originals[name] = dict(source_sha256=sha(source), guard_sha256=sha(receipt),
        original_source_utf8=source.read_bytes().decode('utf-8'), original_guard_utf8=receipt.read_bytes().decode('utf-8'))
    for track, splits in row['cells'].items():
        for split, verified in splits.items():
            produced = record['track_split_records'][track][split]
            assert verified['native_headers_and_file_hashes_verified'] and verified['all_signal_coordinates_finite']
            assert verified['raw_row_origin_and_sequence_verified'] and verified['original_constant_window_mask_verified']
            assert verified['bounded_signal_scan']['checked_coordinates'] == verified['coordinates']
            cells.append(dict(dataset=name, track=track, split=split,
                directory=str(source.parent / track / split),
                source_configuration=verifier.producer.CONFIG, source_proof_sha256=sha(source), source_guard_sha256=sha(receipt),
                raw_CSV=job['raw'], raw_numeric_columns=[c for c in record['complete_raw_audit']['CSV_columns'] if c != 'date'],
                rows=verified['time_rows'], channels=verified['channels'], sequence_length=256, stride=1,
                source_raw_row_start=verified['source_raw_time_start'], source_raw_row_end_exclusive=verified['source_raw_time_end_exclusive'],
                complete_window_count=verified['window_count'], files=produced['files'],
                input_context_overlaps=row['input_context_row_overlaps'],
                source_scaler_fitted_rows=produced['source_scaler_fitted_rows'],
                strict_temporal_independence_certified=False, model_consumption_verified=False))
    assert all(c['exactly_different_coordinates'] == 0 for c in row['separate_normalized_OT_comparisons'].values())
assert len(cells) == 54
write(folder / 'all9_original_source_and_guard_receipts.json', dict(original_UTF8_bytes_recoverable=True, records=originals))
index_path = 'configs/reproducibility/fm_moment_ICM_long_forecast_pretraining_cache_index.2026-10-15_v1.json'
assert not (ROOT / index_path).exists()
write(ROOT / index_path, dict(schema_version=1, paper_id='moment_long_context', paper_citation=config['paper_citation'],
    producer_configuration=verifier.producer.CONFIG, closure_configuration=verifier.CONFIG,
    independent_closure_proof=folder.relative_to(ROOT).as_posix() + '/actual_full54cell_independent_closure.json',
    independent_closure_proof_sha256=sha(proof_path), sources=9, cells=54, cached_arrays=162,
    full_saved_signal_coordinates=26640764, tracks=verifier.producer.TRACKS, splits=verifier.producer.SPLITS,
    explicit_track_selection_required=True, tracks_not_pooled_or_treated_as_equal_data_amount=True,
    cache_layout='time_channel_matrix_plus_absolute_CSV_rows_plus_constant256window_mask',
    future_window_read='TC[index:index+256].T preserving configuredstride1 and rawtimeborders',
    future_paired_independent_and_ICM_models_must_share_full_multivariate_grouped_data_and_batch_plan=True,
    entries=cells, strict_temporal_independence_certified=False, historical_ICM_pipeline_certified=False,
    complete_PILE_loader_integrated=False, neural_model_performance_results=0, all_original_experiments_complete=False))
shutil.copyfile(ROOT / index_path, folder / 'frozen_long_forecast_pretraining_cache_index.json')
buffer = io.StringIO(newline='')
fields = ['dataset', 'track', 'split', 'rows', 'channels', 'complete_window_count',
          'source_raw_row_start', 'source_raw_row_end_exclusive', 'directory']
writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n'); writer.writeheader()
for row in cells: writer.writerow({k: row[k] for k in fields})
(folder / 'all54_long_forecast_cache_cells.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
write(folder / 'complete_long_forecast_cache_summary.json', dict(sources=9, tracks=2, time_splits=3, cells=54,
    arrays_rehashed=162, saved_signal_coordinates_including_repeated_context=26640764,
    every_original_source_time_channel_coordinate_saved_byte_equal=True,
    all_saved_array_bytes_headers_row_origins_finite_and_constant_masks_checked=True,
    corresponding_normalized_OT_coordinates_exactly_equal_across_separate_tracks=True,
    peak_preparation_RSS_bytes=maximum_RSS, peak_preparation_private_bytes=maximum_private,
    peak_closure_RSS_bytes=guard['peak_tree_rss_bytes'], peak_closure_private_bytes=guard['peak_tree_private_bytes'],
    original_ETT_unused_raw_tail_and_input_context_overlap_preserved=True,
    independent_target_disjointness_certified=False, independent_scaler_interpolation_value_rewrite=False,
    source_CT_transpose_already_correct_no_axis_fix=True, tests_passed=11,
    actual_neural_training_inference_SVM=0, full_PILE_pretraining_complete=False, all_original_experiments_complete=False))
(folder / 'README.md').write_text('''# 全部 9 个长预测来源的两轨三划分缓存已核验

ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Exchange Rate、National Illness、Traffic、Weather 共 9 个来源均已实际准备。每个来源保留 2 条独立输入轨迹和 3 个原时间划分，共 54 个组合、162 个缓存数组，独立核验通过；其中保存的信号矩阵共 26,640,764 个坐标，包含不同划分中重复的历史上下文，不能作为唯一原始观测数。

两条输入轨迹必须分开：`source_univariate_OT` 保留相关 MOMENT 来源的默认单通道 OT 选择；`paired_full_multivariate` 保留全部原通道（Traffic 达 862 通道），作为后续独立通道模型与 ICM 使用相同多通道数据的显式重建选择。两者数据量不同，不能混合成一次仅改变架构的消融。逐一比较所有对应 OT 归一化坐标，结果完全相同、没有容差或强制修正；这不意味着单通道与完整多通道的全部数据相同。

通过私有 AST 副本执行固定原 `LongForecastingDataset`，保留原时间边界、训练行拟合 StandardScaler、对全 CSV 做 cubic interpolation 后切分及 stride=1。原 `__getitem__` 已正确对时间×通道切片做转置返回通道×时间，实际阅读和真实源数据检查证实这一点；没有声明或修补不存在的轴错误。原始 CSV、已有缓存、模型预算及既有训练进程均未修改。

缓存保存完整时间×通道矩阵、对应原始 CSV 绝对行号和源代码常量 256 点全观测 mask，不把高度重叠的窗口重复展开为巨大的 [窗口,通道,时间] 数组。所有来源信号坐标按原生 dtype 逐字节保存，存储使用 C-order；后续读取 `matrix[index:index+256].T`。来源首、中、末窗口的方向、mask 和端点已核对，不能据此称所有神经网络窗口都已逐个独立重放。

独立核验重新校验所有原 CSV 和 162 个数组的完整哈希，检查完整 shape/dtype、所有信号有限性、每个时间行号的原始起点/顺序、常量 mask，以及独立重建的时间边界和训练 scaler 样本数量。检查采用分块读取并释放映射；完整数据未抽样。来源数据准备和独立核验均逐进程退出码 0、资源保护通过，固定 Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2，沿用原非神经网络数据任务 16/21/16 GiB 准入/紧急线和 3 GiB 进程树上限。未降低任何既有模型/复核门槛、精度或预算。

来源规则也被原样记录：相邻划分各含 256 行历史输入上下文；National Illness 验证段短于 256 行，因此测试输入历史上下文还与训练输入重叠 158 行。ETTh1/2 固定月份规则未用原 CSV 尾部各 3,020 行，ETTm1/2 未用各 12,080 行。输入历史上下文重叠本身不能证明预测目标标签泄漏；本轮没有认证独立目标区间、严格时间独立性或来源全 CSV 插值的因果独立性。这些界限须在严格对照中另行处理，不能静默改写原始复现轨迹。

11 项检查通过，其中 3 个真实来源子测试各检查两轨三划分的全部源矩阵、scaler 行数、CSV 行索引和窗口方向，还覆盖 Fortran 源矩阵的无损分块存储、拒绝覆盖、错误轨迹/划分、原时间边界、非零验证行起点和 OT 差异保留。实际 54 组合另有完整来源/保护证明链及独立数组核验结果。本轮没有独立重做 scaler/插值数值，没有进行神经网络训练、推理、SVM 或新增模型性能结果。

新的 configs 索引逐组合保存目录、原始注册文件、来源/保护哈希、原数值列名、原行范围及窗口数，要求训练入口明确选轨，配对的独立通道与 ICM 必须共用相同分组数据和批次计划。已经闭合的 157 个分类缓存保持独立。Monash、异常检测分支、完整 PILE 采样/分组和实际步数/token、Tiny/ICM 一整 epoch、最终权重和原生复核、下游预测/117 数据集 SVM/beta/严格预处理对照仍需继续。至少 45 篇论文/681 方法记录/273 数据任务及后续新增实验范围不变；整个研究目标尚未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。所有大数组保存在配置指定的 F 盘，Git 保存代码、配置、表格和真实证明。相关 MOMENT 来源与历史 ICM 流水线的完整等价性仍未认证。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), sources=9,
    cells=54, arrays=162, tests_passed=11, original_source_and_guard_chains=9, new_model_results=0,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}, all_original_experiments_complete=False))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_PILE_all9_long_forecast_cache_closed.2026-10-15_v1.json'
write(ara, dict(paper_id='moment_long_context', preparation_configuration=verifier.producer.CONFIG,
    closure_configuration=verifier.CONFIG, cache_index=index_path, evidence=folder.relative_to(ROOT).as_posix(),
    sources=9, separate_tracks=2, splits=3, cache_cells=54, arrays=162,
    complete_independent_array_source_time_audit_passed=True, original_input_context_overlap_retained=True,
    source_axis_transpose_correct_no_fix=True, strict_temporal_independence_certified=False,
    actual_neural_epochs=0, new_model_performance_results=0,
    previous_goal_turn_classification='progress_full157_classification_closure_and_index_published',
    current_goal_turn_classification='progress_all9_long_forecast_preparation_54cells_and_independent_closure_passed',
    all_original_experiments_complete=False))
targets = [verifier.producer.CONFIG, verifier.CONFIG, index_path, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/prepare_moment_icm_PILE_long_forecast_v1.py',
    'scripts/flow_matching/verify_moment_icm_PILE_long_forecast_v1.py',
    'tests/test_fm_moment_icm_PILE_long_forecast_v1.py', 'tests/test_fm_moment_icm_PILE_long_forecast_closure_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_all9_long_forecast_closed_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), sources=9, cells=54, arrays=162, tests=11,
    peak_source_RSS=maximum_RSS, peak_source_private=maximum_private, model_performance_results=0)))
