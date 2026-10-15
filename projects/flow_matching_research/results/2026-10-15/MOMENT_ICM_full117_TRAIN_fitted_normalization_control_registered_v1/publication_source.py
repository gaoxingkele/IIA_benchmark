"""Archive the real source-cache bindings and separate train-only data queue."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import prepare_moment_icm_classification_train_only_v1 as s

ROOT, read, sha, write = s.ROOT, s.read, s.sha, s.write
config = read(ROOT/s.CONFIG); jobs = s.verify(config)
parent = read(ROOT/s.PARENT)
state_root = ROOT/config['state_root']
launch, state = read(state_root/'launch.json'), read(state_root/'status.json')
p = psutil.Process(launch['pid'])
assert abs(p.create_time()-launch['create_time']) < .01 and p.cmdline() == launch['command']
assert state['pid'] == p.pid and state['required_datasets'] == 117
probe = read(ROOT/'tmp/MOMENT_ICM_full117_train_only_cold_probe_20261015_v1.json')
assert probe['passed'] and probe['datasets'] == 117 and not probe['torch_imported'] and not probe['numpy_imported']
validation = ROOT/'tmp/MOMENT_ICM_full117_train_only_validation_20261015_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
checks = sum(int(x.get('tests', 0)) for x in suites)
assert checks == 20 and all(int(x.get(k, 0)) == 0 for x in suites for k in ['failures', 'errors', 'skipped'])
source_path = ROOT/config['source_root']/'moment/data/classification_datasets.py'
tree = ast.parse(source_path.read_text(encoding='utf-8'))
exclusions = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'DATASETS_WITHOUT_NORMALIZATION' for t in n.targets))
applicable_exclusions = sorted(set(exclusions)&{j['dataset'] for j in jobs})
normalization_rows = []
for binding in config['full_original_source_preprocessing_bindings']:
    proof = read(binding['proof_path'])
    normalization_rows.append(dict(dataset=binding['dataset'], original_source_scaled=proof['original_data_standardizer_fitting_performed'],
        original_TEST_specific_fit_preserved=proof['source_TEST_standardizer_fit_preserved_for_diagnostic_control'],
        source_TRAIN=proof['records']['TRAIN'], source_TEST=proof['records']['TEST'],
        source_proof_sha256=binding['proof_sha256'], source_guard_sha256=binding['guard_sha256']))
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full117_TRAIN_fitted_normalization_control_registered_v1'
folder.mkdir(parents=True, exist_ok=False)
files = dict(frozen_train_only_configuration=ROOT/s.CONFIG, frozen_original_source_configuration=ROOT/s.PARENT,
    frozen_control_worker=Path(s.__file__), frozen_source_preprocessing_worker=Path(s.source.__file__),
    frozen_control_tests=ROOT/s.TEST, original_related_classification_source=source_path,
    validation_20_checks=validation)
for label, path in files.items():
    target = folder/(label+path.suffix)
    shutil.copyfile(path, target); assert sha(path) == sha(target)
write(folder/'actual_cold_probe.json', probe)
write(folder/'actual_controller_launch.json', launch)
write(folder/'actual_controller_status_snapshot.json', state)
write(folder/'actual_live_handle_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2]),
    children=[dict(pid=c.pid, create_time=c.create_time(), command=c.cmdline()) for c in p.children(recursive=True)],
    actual_control_datasets_completed=state['completed_count'], all_original_experiments_complete=False))
write(folder/'complete117_original_normalization_and_split_bindings.json', dict(datasets=117, official_splits=234,
    original_source_preprocessing_completed_datasets=117, original_source_rows=normalization_rows,
    original_no_normalization_exceptions_in_scope=applicable_exclusions, new_control_is_not_original_paper_pipeline=True,
    classifier_accuracy_measured=False, neural_model_training_or_inference_performed=False,
    all_original_experiments_complete=False))
(folder/'README.md').write_text(f'''# Tiny／ICM：117数据集的TRAIN拟合归一化对照

原117数据集、234个完整官方TRAIN/TEST的来源预处理已实际完成，每例成功保护回执及报告哈希重新核对。相关MOMENT作者代码在读取TRAIN和TEST时分别StandardScaler.fit(self.data)，因此缩放使用各自完整分布；不使用TEST标签，但不是TRAIN统计量固定的推断流程。此前来源分支完整保留，不用当前对照替换或追溯认证论文。

新对照覆盖同一26多变量+91单变量数据集。逐坐标验证原完整TRAIN缓存后直接引用它；TEST复用从全部官方TRAIN拟合的标准化器，明确禁止任何TEST fit/update。原未标准化例外保留（本范围{len(applicable_exclusions)}个）；这些例外要求新旧完整值、mask、标签、实例行相等。其余数据记录新旧所有TEST坐标的变化数及最大绝对差，标签和实例顺序必须完全相同。

不同原始长度的TEST用来源插值函数对齐到TRAIN标准化器的原始长度，然后应用TRAIN统计量；这是显式对照适配，记录原TEST长度和TRAIN长度。来源NaN插值、变长序列预处理、13种插值例外、剩余补齐/下采样规则、全部实例和通道均保留。来源变长TEST内部统一长度的处理仍保留，所以这里只认证归一化统计量不由TEST拟合，不将所有预处理宣称为完全归纳式或无传导信息。

{checks}项检查通过，含实际原作者解析器/插值/StandardScaler的同长、异长、NaN和未标准化例外；检查固定TRAIN统计量、禁止TEST fit、全部坐标/标签/顺序和原保护控制器。测试使用夹具，不能解释为benchmark性能。冷检查确认117来源完成证据和234原始分割，未导入NumPy/Torch或启动数据worker。

新独立元数据owner实际运行，沿用同一共享CPU锁及16GiB物理/21GiB提交准入、16GiB紧急下限、3GiB RSS/私有内存上限。未缩减样本、通道、窗口或网格，不覆盖原始数据及来源缓存。每例以原环境Python3.10.21/NumPy1.25.2/pandas2.2.2/SciPy1.9.3/sklearn1.5.2运行，失败或部分尝试保留且不自动重跑。

本包是归一化对照实现、原117来源完整绑定和实际新队列登记。快照新对照已完成{state['completed_count']}/117；尚未运行新的Tiny/ICM分类特征提取或SVM，分类准确率、模型间比较和全部702论文指标实验均未称完成。完整PILE两模型训练及后续预测/SVM仍须真实结果，至少45篇/681方法记录/273数据任务及新增实验范围不变。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[相关MOMENT源代码](https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de)、[UCR／UEA数据](https://www.timeseriesclassification.com/)。
''', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n# Preserve the author source docstring trailing spaces byte for byte.\noriginal_related_classification_source.py whitespace=-blank-at-eol\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), tests_passed=20,
    registered_control_datasets=117, actual_source_preprocessing_completed_datasets=117, live_controller_pid=p.pid,
    files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_full117_TRAIN_fitted_control.2026-10-15_v1.json'
write(ara, dict(method='MOMENT-Tiny/ICM downstream classification preprocessing', registered_datasets=117,
    original_source_preprocessing_completed=117, actual_control_datasets_completed=state['completed_count'],
    all_original_TRAIN_TEST_instances_and_channels_retained=True, normalization_test_fit_forbidden=True,
    classifier_accuracy_measured=False, evidence=folder.relative_to(ROOT).as_posix(), all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(), s.CONFIG,
           Path(s.__file__).resolve().relative_to(ROOT).as_posix(), s.TEST]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_ICM_full117_train_only_publication_manifest_20261015_v1.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(registered_control_datasets=117, original_source_completed=117, actual_control_completed=state['completed_count'],
                     live_controller=p.pid, tests_passed=checks, exact_files=len(paths))))
