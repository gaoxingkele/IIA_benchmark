"""Archive source-normalized47Monash caches, full independent closure and index."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import verify_moment_icm_PILE_monash_v1 as verifier

ROOT, read, sha, write = verifier.ROOT, verifier.read, verifier.sha, verifier.write
config = read(ROOT / verifier.CONFIG); previous, jobs = verifier.verify(config)
base = ROOT / config['state_root']; status = read(base / 'status.json'); guard = read(base / 'guard_receipt.json')
assert status['state'] == 'all47_independent_Monash_cache_closure_passed' and status['active_pid'] is None
path = Path(config['artifact_root']) / 'full47_Monash_cache_closure.json'; proof = read(path)
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted'] and guard['proof_sha256'] == sha(path)
assert proof['passed'] and proof['configuration_sha256'] == sha(ROOT / verifier.CONFIG)
assert proof['sources'] == len(proof['records']) == 47 and proof['arrays'] == 235 and proof['metadata_JSONL_files'] == 47
assert proof['total_source_series'] == 620064 and proof['full_signal_coordinates'] == 158736384
assert proof['original_count_gate_sources'] == 33 and proof['separate_temporal_control_sources'] == 28
assert proof['original_selected_training_items'] == 371162 and proof['temporal_control_training_items'] == 278699
assert proof['neural_fit_inference_SVM'] == 0 and not proof['historical_ICM_pipeline_certified']
source_base = ROOT / previous['state_root']; source_status = read(source_base / 'status.json')
assert source_status['completed_count'] == 47 and not source_status['failed_preserved']
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_PILE_all47_Monash_cache_closed_v1'
folder.mkdir(parents=True, exist_ok=False)
for name, source in {
    'frozen_all47_preparation_configuration.json': ROOT / verifier.producer.CONFIG,
    'frozen_all47_closure_configuration.json': ROOT / verifier.CONFIG,
    'moment_icm_monash_streaming_v1.py': ROOT / 'scripts/flow_matching/moment_icm_monash_streaming_v1.py',
    'prepare_moment_icm_PILE_monash_v1.py': ROOT / 'scripts/flow_matching/prepare_moment_icm_PILE_monash_v1.py',
    'verify_moment_icm_PILE_monash_v1.py': ROOT / 'scripts/flow_matching/verify_moment_icm_PILE_monash_v1.py',
    'test_fm_moment_icm_monash_streaming_v1.py': ROOT / 'tests/test_fm_moment_icm_monash_streaming_v1.py',
    'test_fm_moment_icm_PILE_monash_closure_v1.py': ROOT / 'tests/test_fm_moment_icm_PILE_monash_closure_v1.py',
    'actual_all47_source_controller_launch.json': source_base / 'controller_launch.json',
    'actual_all47_source_terminal_status.json': source_base / 'status.json',
    'actual_all47_independent_cache_closure.json': path,
    'actual_closure_guard.json': base / 'guard_receipt.json',
    'actual_closure_controller_launch.json': base / 'controller_launch.json',
    'actual_closure_worker_launch.json': base / 'worker_launch.json',
    'actual_closure_terminal_status.json': base / 'status.json',
}.items():
    shutil.copyfile(source, folder / name); assert sha(source) == sha(folder / name)
checks = 0
for name, source, count in [
    ('native_item_equivalence_validation.xml', 'tmp/MOMENT_ICM_monash_streaming_validation_20261015_v3.xml', 7),
    ('independent_cache_validation.xml', 'tmp/MOMENT_ICM_Monash_closure_validation_20261015_v2.xml', 6)]:
    suites = list(ET.parse(ROOT / source).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == count
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
    checks += count; shutil.copyfile(ROOT / source, folder / name)
history = []
for source, failures, explanation in [
    ('tmp/MOMENT_ICM_monash_streaming_validation_20261015_v1.xml', 3,
     'Native test constructor omitted downsampling_type andusedclassdefaultlast;actualpretrainingYAMLinterpolatewasalreadyusedbyimplementation. Rawvalues/normalizer werebyteequal;testoptionsmadeexplicit beforefreeze/fulljoblaunch.'),
    ('tmp/MOMENT_ICM_Monash_closure_validation_20261015_v1.xml', 1,
     'Fixture RandomState.choice onWindowsreturnedint32;actualsource cacheindex fromint64DataFramesourcevaluesisint64. Fixturemadeint64;corruptiontestsnowfirstprovevalidbaselinebeforemutating. No dataattempt retried.')]:
    p = ROOT / source; suites = list(ET.parse(p).getroot().iter('testsuite'))
    assert sum(int(s.get('failures', 0)) for s in suites) == failures
    history.append(dict(path=source, sha256=sha(p), original_utf8=p.read_bytes().decode('utf-8'),
        failures=failures, explanation=explanation, actual_neural_or_registered_source_attempt_failed=False))
write(folder / 'preserved_pre_registration_validation_diagnoses.json', dict(records=history,
    no_tolerance_relaxed=True, frozen_numerical_source_unchanged=True, final_checks_passed=13))
originals = {}; entries = []; peak_RSS = peak_private = 0
for row, job in zip(proof['records'], jobs):
    assert row['dataset'] == job['dataset']; name = row['dataset']
    source = Path(previous['artifact_root']) / name / 'pretraining_preparation.json'
    receipt = source_base / 'attempts' / name / 'guard_receipt.json'
    record, g = read(source), read(receipt)
    assert record['passed'] and g['passed'] and g['exit_code'] == 0 and not g['resource_guard_aborted']
    assert g['proof_sha256'] == sha(source) and row['source_proof_sha256'] == sha(source) and row['source_guard_sha256'] == sha(receipt)
    peak_RSS = max(peak_RSS, g['peak_tree_rss_bytes']); peak_private = max(peak_private, g['peak_tree_private_bytes'])
    originals[name] = dict(source_sha256=sha(source), guard_sha256=sha(receipt),
        original_source_utf8=source.read_bytes().decode('utf-8'), original_guard_utf8=receipt.read_bytes().decode('utf-8'))
    raw_header = dict(frequency=None, forecast_horizon=None, contain_missing_values=None, contain_equal_length=None)
    with (ROOT / job['raw']['path']).open('r', encoding='cp1252') as stream:
        for line in stream:
            text = line.strip()
            if text.startswith('@data'): break
            parts = text.split(' ')
            if len(parts) != 2: continue
            if parts[0] == '@frequency': raw_header['frequency'] = parts[1]
            elif parts[0] == '@horizon': raw_header['forecast_horizon'] = int(parts[1])
            elif parts[0] == '@missing': raw_header['contain_missing_values'] = verifier.producer.streaming.metadata.boolean(parts[1])
            elif parts[0] == '@equallength': raw_header['contain_equal_length'] = verifier.producer.streaming.metadata.boolean(parts[1])
    result = row['result']; output = record['record']
    assert result['all_raw_names_lengths_chunks_and_indices_reconciled'] and result['exact_seed13_shuffle_and_splits']
    assert result['all_signals_finite_and_exact_source_padding_masks']
    entries.append(dict(dataset=name, directory=row['cache_directory'], raw_TSF=job['raw'], source_TSF_metadata=raw_header,
        source_proof_sha256=sha(source), source_guard_sha256=sha(receipt), series=result['series'],
        channels=1, sequence_length=256, dtype=output['dtype'], mask_dtype=output['mask_dtype'],
        files_sha256=output['files_sha256'], source_seed=13, source_split_positions=result['original_split_positions'],
        source_count_gate_selected=result['original_count_gate_selected'],
        separate_temporal_control_selected=result['separate_temporal_control_selected'],
        source_training_rows=result['source_training_rows'], source_padding_coordinates=result['source_padding_coordinates'],
        normalization_full_source_series_before_resampling_and_masking=True,
        strict_masked_target_independent_normalization=False))
write(folder / 'all47_original_source_and_guard_receipts.json', dict(original_UTF8_bytes_recoverable=True, records=originals))
index_path = 'configs/reproducibility/fm_moment_ICM_Monash_pretraining_cache_index.2026-10-15_v1.json'
assert not (ROOT / index_path).exists()
write(ROOT / index_path, dict(schema_version=1, paper_id='moment_long_context', paper_citation=config['paper_citation'],
    producer_configuration=verifier.producer.CONFIG, closure_configuration=verifier.CONFIG,
    independent_closure_proof=folder.relative_to(ROOT).as_posix() + '/actual_all47_independent_cache_closure.json',
    independent_closure_proof_sha256=sha(path), sources=47, arrays=235, metadata_files=47,
    total_source_series=620064, full_signal_coordinates=158736384,
    selection_policy_must_be_explicit=['original_source_count_field', 'separate_training_temporal_control'],
    original_count_gate_sources=33, temporal_control_sources=28,
    original_selected_training_items=371162, temporal_control_training_items=278699,
    cache_order='raw_source_order_apply_source_shuffle_order_then_split_position_indices',
    model_random_mask_not_cached=True, entries=entries,
    complete_normalization_resampling_values_independently_reconstructed=False,
    strict_masked_target_independent_normalization=False, historical_ICM_pipeline_certified=False,
    complete_PILE_training_loader_integrated=False, neural_model_performance_results=0, all_original_experiments_complete=False))
shutil.copyfile(ROOT / index_path, folder / 'frozen_Monash_pretraining_cache_index.json')
buffer = io.StringIO(newline='')
fields = ['dataset', 'series', 'source_training_rows', 'source_count_gate_selected', 'separate_temporal_control_selected',
          'source_padding_coordinates', 'dtype', 'directory']
writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n'); writer.writeheader()
for row in entries: writer.writerow({k: row[k] for k in fields})
(folder / 'all47_normalized_Monash_cache_and_selection.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
write(folder / 'complete_Monash_cache_summary.json', dict(sources=47, arrays=235, metadata_JSONL_files=47,
    full_source_series=620064, cached_signal_coordinates=158736384,
    original_count_gate_sources=33, separate_temporal_gate_sources=28,
    original_gate_training_items=371162, temporal_gate_training_items=278699,
    complete_finite_coordinates_exact_masks_raw_names_lengths_chunk_ID_shuffle_split_audit_passed=True,
    normalized_native_source_equivalence_tests_real_datasets=6, all_items_and_three_splits_compared_for_those_six=True,
    native_full_eager_frames_materialized_for_all47=False,
    peak_preparation_RSS_bytes=peak_RSS, peak_preparation_private_bytes=peak_private,
    peak_independent_closure_RSS_bytes=guard['peak_tree_rss_bytes'], peak_independent_closure_private_bytes=guard['peak_tree_private_bytes'],
    tests_passed=13, independent_all_normalization_resampling_values_reconstructed=False,
    strict_masked_target_independent_normalization=False, actual_neural_model_fit_inference_SVM=0,
    all_original_experiments_complete=False))
(folder / 'README.md').write_text('''# 全部 47 个 Monash 来源的归一化输入缓存已核验

47/47 个登记来源均已实际完成归一化和源 getitem 重采样，独立核验通过：620,064 条原来源序列、235 个数组及 47 个逐序列元数据文件，共 158,736,384 个固定 256 点信号坐标，另有对应完整 mask。没有因来源筛选未选中而省略任何文件。源码计数门槛选中 33 个文件、371,162 个训练条目；单独的训练时间长度对照选中 28 个文件、278,699 个训练条目。两套数据组成和预算不同，必须明确选择，不得混为仅改变结构的消融。

新流式实现每次读取一条原 TSF 序列，用相同 Python float 解析为 float64 并构建原 PandasArray；对 5 个特殊长序列，先按原代码截掉尾部余数、分成 256 点 NumPyArray 块，再逐块归一化。保留两种数组类型以保持源 mean/std 的计算行为，执行固定来源的归一化 lambda `(i-i.mean())/(i.std(ddof=0)+1e-7)` 及完整原 `__getitem__`，配置明确为预训练 YAML 的 backward pad/interpolate。类构造器自身默认的 last 截段与 YAML 不同，未把两者混用，也未宣称这个配置差异是作者算法错误。

缓存按原来源行序保存信号、mask、行号、原长度和精确 seed13 洗牌索引，再用原 0.6/0.1/0.3 位置范围定义训练/验证/测试。5 个数组和逐行 JSONL 保留原序列名、原序列号、特殊分块号和原时间范围；源代码特殊尾部丢弃也被独立核对。每个输出坐标和 mask 与当次原函数输出逐字节比较；写出和检查都分块释放映射。完整当前快照没有 ? 或非有限数值，未悄悄采用不同的字符串缺失值转换。

6 个真实来源（NN5 daily、FRED-MD、US births、Saugeen day、Bitcoin、M1 monthly）以原完整数据类 scale=True 为对照，逐项比较全部训练/验证/测试样本，归一化后信号和 mask 均逐字节相同，涵盖长序列插值、短序列填充、等长特殊分块。更大的 47 个原全表没有全部以 eager 数据类重新构建；本轮不作该项内存密集认证，也不把流式重用算术与历史 ICM 完整流水线等价混为一谈。

独立检查重新校验全部原 TSF 和 235 个数组/47 个 JSONL 的完整哈希，独立读原始名称/长度/特殊分块，核对所有行号、seed13排列、三个划分及原训练长度统计；访问全部信号验证有限性，按原长度逐坐标核对 backward padding mask 和常量零填充。两套筛选及条目数也与此前完整原输入审计一致。检查没有独立重写所有源归一化/插值数值，模型随机掩码尚未产生，没有训练、推理、SVM 或新性能结果。

13 项最终检查通过。两次登记前的测试失配保留原 XML 字节：一次对照构造器省略了 interpolate 参数而使用类默认 last；另一次测试夹具在 Windows 得到 int32 排列，而实际源索引来自 int64 DataFrame。均在对应全量验证配置冻结和执行前纠正，没有放松容差或重试真实失败实验。47 个真实准备任务均有成功退出码、完整证明和资源保护回执，沿用原非神经网络数据任务 16/21/16 GiB 门槛及 3 GiB 进程树上限；既有模型与复核预算均未修改。原始 TSF 和旧缓存保留。

来源按完整序列统计归一化后再进行模型掩码和重采样；验证/测试序列也使用自身完整统计。本缓存保留该来源规则，不能称严格训练集专属、掩码目标统计独立或历史 ICM 实现认证。后续严格对照必须另行登记处理，不能静默改变原始复现。

新的 configs 索引记录全部目录、哈希、源 TSF 频率/预测步长等元数据、两套筛选标志、精确洗牌与划分规则。已闭合的 157 分类和 9 长预测来源保持独立。异常检测分支、完整 PILE 采样/分组/实际步数和 token、Tiny/ICM 一整 epoch、最终权重与原生复核、下游预测/117 数据集 SVM/beta/严格预处理对照仍需继续；至少 45 篇论文/681 方法记录/273 数据任务及后来新增实验范围不变，全部研究目标尚未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。大数组在配置指定 F 盘，Git 保存代码、配置、表格及完整实际证明。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), sources=47,
    arrays=235, source_series=620064, coordinates=158736384, tests_passed=13, new_model_results=0,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}, all_original_experiments_complete=False))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_PILE_all47_Monash_cache_closed.2026-10-15_v1.json'
write(ara, dict(paper_id='moment_long_context', preparation_configuration=verifier.producer.CONFIG,
    closure_configuration=verifier.CONFIG, cache_index=index_path, evidence=folder.relative_to(ROOT).as_posix(),
    sources=47, arrays=235, full_source_series=620064, complete_independent_cache_audit_passed=True,
    original_source_gate_selected=33, separate_temporal_gate_selected=28,
    strict_masked_target_independent_normalization=False, historical_ICM_pipeline_certified=False,
    actual_neural_epochs=0, new_performance_results=0,
    previous_goal_turn_classification='progress_all9_forecast_tracks_splits_cache_closed_and_published',
    current_goal_turn_classification='progress_all47_source_normalized_Monash_caches_and_independent_closure_passed',
    all_original_experiments_complete=False))
targets = [verifier.producer.CONFIG, verifier.CONFIG, index_path, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/moment_icm_monash_streaming_v1.py', 'scripts/flow_matching/prepare_moment_icm_PILE_monash_v1.py',
    'scripts/flow_matching/verify_moment_icm_PILE_monash_v1.py', 'tests/test_fm_moment_icm_monash_streaming_v1.py',
    'tests/test_fm_moment_icm_PILE_monash_closure_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_all47_Monash_closed_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'), previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), sources=47, arrays=235, source_series=620064, tests=13,
    preparation_peak_RSS=peak_RSS, preparation_peak_private=peak_private, model_performance_results=0)))
