"""Publish diagnosed pre-native failures, the minimal correction and live owner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_moment_TAB100_metadata_recovery_v2 as v

ROOT, read, write, sha = v.ROOT, v.read, v.write, v.sha
config = read(ROOT/v.CONFIG); runtime, cases = v.verify(config)
raw_probe = (ROOT/'tmp/MOMENT_TAB100_metadata_recovery_cold_probe_20261015_v2.json').read_bytes()
probe = json.loads(raw_probe.decode('utf-16') if raw_probe.startswith(b'\xff\xfe') else raw_probe.decode('utf-8-sig'))
assert probe['passed'] and probe['selected_cases'] == 39 and probe['original_required_cases'] == 78
assert probe['numerical_workers_started'] == 0 and not probe['torch_imported'] and not probe['numpy_imported']
assert len(probe['proven_pre_native_rejections']) == 12
for identity, evidence in probe['proven_pre_native_rejections'].items():
    assert v.metadata_failure(cases[identity], runtime) == evidence
validation = ROOT/'tmp/MOMENT_TAB100_metadata_recovery_validation_20261015_v2.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
checks = sum(int(s.get('tests', 0)) for s in suites)
assert checks == 31 and all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
state_root = ROOT/config['state_root']
launch, state = read(state_root/'launch.json'), read(state_root/'status.json')
p = psutil.Process(launch['pid'])
assert abs(p.create_time()-launch['create_time']) < .01 and p.cmdline() == launch['command']
assert state['pid'] == p.pid and state['required_cases'] == 39
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_TAB100_pre_native_metadata_defect_recovery_registered_v2'
folder.mkdir(parents=True, exist_ok=False)
files = dict(frozen_recovery_configuration=ROOT/v.CONFIG, frozen_original_full78_configuration=ROOT/v.ORIGINAL,
             frozen_recovery_worker=ROOT/v.__file_relative(), frozen_original_full78_worker=ROOT/v.SOURCE,
             frozen_recovery_tests=ROOT/v.TEST, validation_31_checks=validation)
for label, path in files.items():
    suffix = path.suffix
    target = folder/(label+suffix)
    shutil.copyfile(path, target); assert sha(path) == sha(target)
write(folder/'actual_cold_metadata_probe.json', probe)
write(folder/'actual_controller_launch.json', launch)
write(folder/'actual_controller_status_snapshot.json', state)
write(folder/'actual_live_handle_verification.json', dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2]),
    children=[dict(pid=c.pid, create_time=c.create_time(), command=c.cmdline()) for c in p.children(recursive=True)],
    completed_recovered_cases_at_capture=state['verified_count'], all_original_experiments_complete=False))
for identity in probe['proven_pre_native_rejections']:
    previous = ROOT/read(ROOT/v.ORIGINAL)['state_root']/'attempts'/identity/'native'
    target = folder/'preserved_original_metadata_failures'/identity; target.mkdir(parents=True)
    for name in ['guard_receipt.json', 'launch.json', 'stderr.log', 'stdout.log']:
        shutil.copyfile(previous/name, target/name)
        assert sha(previous/name) == sha(target/name)
(folder/'README.md').write_text('''# MOMENT TAB100：原生重推理前的元数据错误修复

已核对12个small/context100原失败：均以exit1在check_source中拒绝，未进入模型加载和推理、未创建native输出目录、未发生资源中止。旧检查误把官方权重config.json中的seq_len=512当作外部实验窗口，要求等于100。冻结原模型明确保留window100及TAB_always_extra_patch；两者是不同元数据。

修正版在新独立进程中编译原完整模块，仅替换check_source中这一条比较：权重配置长度改为核对实际哈希绑定config.json的seq_len=512。全部其余来源验证保留，原生/第二独立验证函数的字节码与常量相同。没有改原训练、模型权重、窗口、补齐、数组、阈值、指标、容差、线程或保护规则；正在运行的v1源码/配置不改动。

31项检查通过，覆盖原生两窗口补齐、全窗口MSE、完整原生及独立worker、全部78来源/控制和新修复拒绝边界。真实冷检查证实12个失败经这一条元数据纠正后，其余原来源检查全部通过；冷检查未导入NumPy/Torch、未启动数值worker。这些检查不是新benchmark结果。

已登记39个context100槽位（36原始+3常量拟合控制），原78项完整范围不变。新元数据owner已真实启动，遵循原共享CPU锁、16GiB物理/32GiB提交准入、16GiB紧急下限、10GiB RSS/12GiB私有上限；逐项等实际旧回执证明同一前置元数据故障，才进入独立新目录。来源仍在运行或已有数值失败/部分输出的案例不得自动重跑。原12个失败保护/启动/日志逐字节保存，新重推理须实际原生和独立两阶段通过才计完成。

本包登记修正和真实队列，不宣称39项已完成。权重历史等价、缺失原执行保护、全预处理独立重建、完整TAB执行器及论文Tiny/ICM训练仍分开保留。至少45篇/681方法记录/273数据任务及新增实验范围不变，全部研究目标未完成。

来源：[MOMENT论文](https://arxiv.org/abs/2402.03885)、[TAB原代码](https://github.com/decisionintelligence/TAB)、[官方Small权重](https://huggingface.co/AutonLab/MOMENT-1-small)。
''', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.log -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_original_pre_native_metadata_failures=12, registered_recovery_cases=39, tests_passed=31,
    cold_probe_numeric_workers_started=0, live_controller_pid=p.pid,
    files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_TAB100_pre_native_metadata_recovery.2026-10-15_v2.json'
write(ara, dict(method='MOMENT', context=100, metadata_bug_diagnosed=True, actual_failed_pre_native_cases=12,
    recovery_registered_cases=39, original_full78_scope_retained=True, original_numerics_and_resource_policy_retained=True,
    evidence=folder.relative_to(ROOT).as_posix(), all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(), v.CONFIG, v.__file_relative(), v.TEST]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_TAB100_metadata_recovery_publication_manifest_20261015_v2.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(registered=39, diagnosed_actual_metadata_failures=12, live_controller=p.pid, exact_files=len(paths), actual_recovered_complete=state['verified_count'])))
