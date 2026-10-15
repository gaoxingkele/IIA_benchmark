"""Archive complete callable classifier path without claiming pending runs done."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_moment_icm_full_classification_v1 as s

ROOT, read, sha, write = s.ROOT, s.read, s.sha, s.write
config = read(ROOT/s.CONFIG); jobs = s.verify(config)
probe = read(ROOT/'tmp/MOMENT_ICM_full468_classification_cold_probe_20261015_v1.json')
assert probe['passed'] and probe['classifier_cases'] == 468 and not probe['torch_imported'] and not probe['numpy_imported']
assert probe['actually_ready_full_models'] == dict(independent=False, icm=False)
validation = ROOT/'tmp/MOMENT_ICM_full468_classification_validation_20261015_v3.xml'
failure = ROOT/'tmp/MOMENT_ICM_full468_classification_validation_20261015_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests', 0)) for x in suites) == 28
assert all(int(x.get(k, 0)) == 0 for x in suites for k in ['failures', 'errors', 'skipped'])
assert sum(int(x.get('failures', 0)) for x in ET.parse(failure).getroot().iter('testsuite')) == 1
root = ROOT/config['state_root']; launch, state = read(root/'launch.json'), read(root/'status.json')
p = psutil.Process(launch['pid'])
assert abs(p.create_time()-launch['create_time']) < .01 and p.cmdline() == launch['command']
assert state['required_classifier_cases'] == 468 and state['completed_count'] == 0
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full468_SVM_execution_registered_v1'
folder.mkdir(parents=True, exist_ok=False)
files = dict(frozen_full468_configuration=ROOT/s.CONFIG, frozen_model_configuration=ROOT/s.MODEL,
    frozen_classifier_science=ROOT/'scripts/flow_matching/moment_icm_svm_classification_v1.py',
    frozen_guarded_controller=Path(s.__file__), frozen_classifier_tests=ROOT/s.TEST,
    original_source_SVM=ROOT/s.science.SVM_SOURCE,
    original_source_periodic_evaluation=ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original/scripts/evaluation/evaluation.py',
    validation_28_checks=validation, preserved_pre_registration_validation_failure=failure)
for label, path in files.items():
    target = folder/(label+path.suffix)
    shutil.copyfile(path, target); assert sha(path) == sha(target)
write(folder/'actual_cold_probe.json', probe)
write(folder/'actual_controller_launch.json', launch)
write(folder/'actual_controller_status_snapshot.json', state)
write(folder/'actual_live_owner_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2]),
    children=[dict(pid=c.pid, create_time=c.create_time(), command=c.cmdline()) for c in p.children(recursive=True)],
    actual_classifier_cases_completed=0, all_original_experiments_complete=False))
write(folder/'pre_registration_test_assumption_correction.json', dict(
    actual_initial_failures=1, failed_assertion='Native sklearn should reject positive-infinite C',
    native_observation='In the actual author environment sklearn1.5.2,SVC(C=positive_infinity) fits and has finite dual coefficients',
    correction='Keep the exact source infinite candidate and native fitted model;symbolic encoding in JSON logs only',
    source_numerical_SVM_AST_changed=False, production_scientific_attempt_failed=False,
    corrected_checks_passed=28, fixtures_are_benchmark_performance=False))
(folder/'README.md').write_text('''# Tiny／ICM：完整117数据集×两模型×两归一化流程的SVM执行链路

已实现并登记468项分类案例：117个原数据集×独立通道/ICM两个完整PILE最终模型×来源TEST单独拟合/新增TRAIN拟合两条归一化流程。其中234项是原模型/数据集组合，234项是额外预处理对照；不把扩充对照计成新的原论文指标。原702标量目标、预测任务、上下文基线和其他论文范围仍保留。

每例必须先取得实际成功的2086来源、80,850,189训练项、39,479更新、final_post_scheduler完整检查点、训练状态证明和原保护回执。公开small/base/large权重或随机初始化不能满足此门禁。严格加载全部core.*权重，保留重建头；之后仅切换feature输出模式，eval且全冻结。CPUfloat32每次一个完整多变量案例，所有通道同时参与ICM；按来源完整有效patch掩码做每通道时间平均，再展平C×256特征。每次实际forward的池化结果都有独立float64 NumPy归约核对，容差预先固定2e-6。CPU、casebatch1及通道组合/池化是公开的重建选择，不能认证未公开ICM源码或历史结果等价。

完整官方TRAIN和TEST的所有案例、通道、mask、标签和原行序保留为特征；SVM原函数AST不改动，保持原5折、10并行、C网格、gamma及其它参数和作者MAX_SAMPLES10000分层子采样规则，小样本分支保持C100000。大训练集的SVM拟合子采样数另行明确记录，不把它隐去。测试标签不参加C选择。原positive-infinity候选原值传给SVC，实际作者环境可以拟合；日志仅用字符串编码无穷值，不改变模型。如果任何候选产生失败，原警告和失败CV分数按原行为完整保留，不替换候选。

保存全量特征、标签、预测、完整SVM、全部CV审计及混淆/accuracy/macroF1。第二独立进程重载完整分类器，对所有TRAIN/TEST案例重新predict，要求原预测dtype和数组完全一致，再独立标量复算整表。重新检查当前训练和数据绑定以及所有输入/输出哈希；既有成功回执不能许可输入变化。描述性Wilson95区间只表达固定模型测试案例不确定性，未知受试者依赖/主体分组及重训种子差异未解决，不提升为严格排行榜。

28项检查通过，覆盖全部468身份、原资源/容差、未完成预训练前禁止数值导入、真实原环境mask池化/SVM小样本与C∞拟合、模拟CV审计、完整混淆指标、旧证据重用门禁和两归一化流程。夹具和模拟搜索仅验证执行逻辑，不代表完整CV或benchmark指标。保留初次1项失败：错误假定原生sklearn拒绝C∞；实际环境证明可拟合，已纠正测试假设和日志序列化，未失败或重跑任何科学实验。

独立元数据owner已实际启动。沿用原16GiB物理/32GiB提交准入、16GiB紧急下限、10GiB RSS/12GiB私有上限和同一共享CPU锁；worker启动后包含所有SVM子进程一起保护。原数值环境不改，失败/部分执行保留且不自动重试。冷检查未导入NumPy/Torch，两个实际完整模型尚未就绪，因此没有启动分类数值任务，快照完成数0/468；不是完成468实验的声明。

来源周期回调实际只调用zero_shot/anomaly_detection.py，分类执行不会把这些待做AD评估、预测实验或独立完整模型重推理标为完成。原训练/输入队列继续运行；至少45篇/681方法记录/273数据任务及新增实验范围不变，整个目标未完成。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[相关作者代码](https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de)、[UCR／UEA](https://www.timeseriesclassification.com/)。
''', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n# Preserve original source and failed-test evidence byte for byte.\noriginal_source_SVM.py whitespace=-blank-at-eol\npreserved_pre_registration_validation_failure.xml whitespace=-blank-at-eol\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    registered_classifier_cases=468, actual_completed_classifier_cases=0, tests_passed=28, preserved_initial_test_failures=1,
    actual_live_owner=p.pid, files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_full468_classifier_registered.2026-10-15_v1.json'
write(ara, dict(method='Tiny-independent and Tiny-ICM full-PILE features plus original SVM',
    registered_datasets=117, original_model_dataset_cases=234, additional_normalization_control_cases=234,
    required_full_training_updates=39479, actual_completed_classifier_cases=0, tests_passed=28,
    evidence=folder.relative_to(ROOT).as_posix(), periodic_source_AD_callbacks_completed=False, all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(), s.CONFIG, s.MODEL, s.TEST,
    'scripts/flow_matching/moment_icm_svm_classification_v1.py', 'scripts/flow_matching/run_moment_icm_full_classification_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_ICM_full468_classification_publication_manifest_20261015_v1.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(registered=468, actual_completed=0, tests_passed=28, live_owner=p.pid, exact_files=len(paths))))
