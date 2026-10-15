"""Publish the separate real downstream-input registration and live guard owner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
import yaml
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as s

ROOT, read, sha, write = s.ROOT, s.read, s.sha, s.write
config = read(ROOT/s.CONFIG); s.verify(config)
probe = read(ROOT/'tmp/MOMENT_ICM_forecast_input_all18_cold_probe_20261015_v1.json')
assert probe['passed'] and not probe['numpy_imported'] and not probe['torch_imported']
validation = ROOT/'tmp/MOMENT_ICM_forecast_input_all18_validation_20261015_v3.xml'
failure = ROOT/'tmp/MOMENT_ICM_forecast_input_all18_validation_20261015_v2.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests', 0)) for x in suites) == 17
assert all(int(x.get(k, 0)) == 0 for x in suites for k in ['failures', 'errors', 'skipped'])
assert sum(int(x.get('failures', 0)) for x in ET.parse(failure).getroot().iter('testsuite')) == 1
base = ROOT/config['state_root']; launch, state = read(base/'controller_launch.json'), read(base/'status.json')
p = psutil.Process(launch['pid'])
assert abs(p.create_time()-launch['create_time']) < .01 and p.cmdline() == launch['command']
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_all6_forecast_inputs_all18_horizons_registered_v1'
folder.mkdir(parents=True, exist_ok=False)
files = dict(frozen_forecast_input_configuration=ROOT/s.CONFIG, frozen_input_worker=Path(s.__file__),
    frozen_input_tests=ROOT/s.TEST, frozen_source_long_forecast_adapter=Path(s.original.__file__),
    original_forecast_recipe=ROOT/s.RECIPE,
    original_forecast_dataset_source=ROOT/config['original_source_root']/'moment/data/forecasting_datasets.py',
    validation_17_checks=validation, preserved_pre_registration_validation_failure=failure)
for label, path in files.items():
    target = folder/(label+path.suffix)
    shutil.copyfile(path, target); assert sha(path) == sha(target)
write(folder/'actual_cold_probe.json', probe)
write(folder/'actual_controller_launch.json', launch)
write(folder/'actual_controller_status_snapshot.json', state)
write(folder/'actual_live_handle_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2]),
    children=[dict(pid=c.pid, create_time=c.create_time(), command=c.cmdline()) for c in p.children(recursive=True)],
    actual_complete_forecast_input_proof_exists=(Path(config['artifact_root'])/s.REPORT).exists(),
    all_original_experiments_complete=False))
recipe = yaml.safe_load((ROOT/s.RECIPE).read_text(encoding='utf-8'))
default = yaml.safe_load((ROOT/config['original_source_root']/'configs/default.yaml').read_text(encoding='utf-8'))
write(folder/'source_forecaster_recipe_and_remaining_execution.json', dict(
    original_related_source_lookback=recipe['seq_len'], paper_requested_lookback=256,
    source_downstream_ratios={k: recipe[k+'_ratio'] for k in ['train', 'val', 'test']},
    source_train_batch=recipe['train_batch_size'], source_validation_batch=recipe['val_batch_size'],
    source_max_epoch=recipe['max_epoch'], source_learning_rate=recipe['init_lr'],
    source_scheduler=recipe['lr_scheduler_type'], source_pct_start=recipe['pct_start'],
    source_weight_decay=recipe['weight_decay'], source_torch_dtype=default['torch_dtype'],
    source_use_amp=default['use_amp'], original_ICM_recipe_equivalence_certified=False,
    actual_forecaster_training_complete=False, all_graph_concat_crossformer_iTransformer_NBEATS_baselines_complete=False,
    fold8_fold4_to_freeze_beta_mapping_established=False, all702_paper_scalar_targets_locally_reproduced=False))
write(folder/'pre_registration_comparison_context_correction.json', dict(
    initial_controller_comparison_test_failures=1,
    reason='Python3.12 uses equivalent different LOAD_GLOBAL/LOAD_ATTR NULL-call flags for imported globals in whole-module versus isolated-function compilation',
    correction='Compare pristine and state-label-adapted source controllers in the same isolated compilation context,recursively matching bytecode,names,locals,closures and constants',
    only_two_descriptive_state_strings_changed=True, numerical_or_resource_guard_code_changed=False,
    actual_scientific_worker_failed=False, final_tests_passed=17))
(folder/'README.md').write_text('''# Tiny／ICM：六个原始预测数据集、全部18组horizon的独立输入链路

原6数据集Exchange、ETTh1/2、ETTm1/2、Weather全部保留；lookback256、horizon96/192/384、全多变量通道、stride1。相关作者预测配置对Exchange和Weather使用70/10/20，而已闭合预训练缓存是60/10/30，两套划分明确分开。ETT继续使用来源固定12/4/4个月切点。原始CSV及旧缓存不覆盖。

使用冻结原LongForecastingDataset类，训练段拟合StandardScaler，三个角色用同一训练统计量。val/test带256点历史，但完整所有forecast目标必须位于各自新时间段；检查全部窗口起点和目标末端边界，独立记录每个horizon的窗口/输入坐标/目标坐标数。每个角色只保存完整TC矩阵、原CSV行映射及共享mask，逐字节保存全部来源坐标；实际源__getitem__对每horizon首/中/末窗口核对history/forecast转置、维度、mask和端点，不冒称逐窗独立重建了所有值，也不生成数百倍重复窗口张量。

固定ETT月份可能不使用CSV末尾额外月份；这些原行仍在原文件保留，选中末行及未进入来源固定月份的行数明确记录。原全CSV cubic插值在划分前发生，保持且披露；不能据训练拟合缩放就宣称整个插值严格无未来信息。此包不认证历史作者ICM精确预处理等价。

17项检查通过，覆盖真实原类在模拟Exchange和ETTh1的完整三角色/三horizon、首中末窗口、训练统计、所有目标边界、原始文件保留、6源18组合完整性、资源/配置防缩减、冷导入以及原保护控制器一致性。夹具不是benchmark预测结果。保留先前1项测试比较失败：全模块与单函数编译在Python3.12对module调用有等价NULL标志编码；最终在相同编译上下文递归核对原字节码/常量，只允许两条描述性状态名变化。未失败或重跑科学数据worker，也未改保护规则。

已实际启动独立元数据owner，沿用原同一共享CPU锁与16GiB物理/21GiB提交准入、16GiB紧急下限、3GiB RSS/私有内存上限。原源环境Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2保留；等待原资源准入，不旁路其他训练或更改预算。数据worker须完成全部6源、18horizon、54角色组并成功保护退出才算完整。

当前包是全量预测输入执行实现和实际等待队列，尚未产出完整数据证明、训练预测器或新的MSE/MAE。相关预测配置还提供5epoch、train batch64/val batch256、OneCycleLR及AMP等来源参数，已另表冻结为后续执行依据；它们不证明未公开ICM历史配方。两模型完整PILE预训练、forecast head/β微调、Graph/Concat/Crossformer/iTransformer/NBEATS及全部702论文标量目标继续保留，8fold/4fold与freezeβ的对应未建立，不能擅自等同。至少45篇/681方法记录/273数据任务及新增实验范围不变，整个研究目标未完成。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[相关作者代码](https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de)、[Autoformer数据](https://github.com/thuml/Autoformer)。
''', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.yaml -text\n# Keep source and prior failed validation byte exact.\noriginal_forecast_dataset_source.py whitespace=-blank-at-eol\noriginal_forecast_recipe.yaml whitespace=-blank-at-eol\npreserved_pre_registration_validation_failure.xml whitespace=-blank-at-eol\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), tests_passed=17,
    registered_datasets=6, registered_horizon_dataset_pairs=18, actual_complete_forecast_input_proof_exists=False,
    preserved_pre_registration_test_failures=1, live_owner=p.pid,
    files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_all6_forecast_inputs_all18_horizons.2026-10-15_v1.json'
write(ara, dict(task='Tiny/ICM downstream forecasting inputs', datasets=6, horizon_dataset_pairs=18,
    original_downstream_splits_separate_from_pretraining=True, actual_forecaster_results_produced=0,
    evidence=folder.relative_to(ROOT).as_posix(), all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(), s.CONFIG,
           Path(s.__file__).resolve().relative_to(ROOT).as_posix(), s.TEST]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_ICM_forecast_inputs_all18_publication_manifest_20261015_v1.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(registered_datasets=6, horizon_dataset_pairs=18, tests_passed=17, live_owner=p.pid, exact_files=len(paths))))
