"""Archive the diagnosed metadata failure and exact idle-only recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import resume_anomaly_transformer_full500_metadata_v1 as recovery

ROOT = recovery.ROOT
config_path = 'configs/runtime/fm_anomaly_transformer_full500_metadata_recovery.v1.json'
config = recovery.runner.read(ROOT/config_path)
original, cases = recovery.verify_recovery(config)
base = ROOT/original['state_root']
launch = recovery.runner.read(base/'metadata_recovery_launch_v1/launch.json')
p = psutil.Process(launch['pid'])
assert p.create_time() == launch['create_time']
assert 'scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1' in p.cmdline()
assert launch['configuration_sha256'] == recovery.runner.sha(ROOT/config_path)
state = recovery.runner.read(base/'status.json')
assert state['pid'] == p.pid and state['create_time'] == p.create_time()
assert state['required_models'] == len(cases) == 500 and state['active_pid'] is None
assert state['verified_model_count'] == 0 and state['failed_preserved'] == {}
assert not (base/'metadata_recovery_launch_v1/stderr.log').read_bytes()
preflight = recovery.runner.read(base/'metadata_recovery_v1/recovery_preflight.json')
assert preflight['prior_exact_owner_missing'] and preflight['numerical_attempts_found'] == 0
xml = ROOT/'tmp/Full500_metadata_recovery_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 10
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))

folder = ROOT/'projects/flow_matching_research/results/2026-10-15/full500_metadata_recovery_v1'
folder.mkdir(parents=True, exist_ok=False)
copies = {
    'frozen_recovery_configuration.json': ROOT/config_path,
    'unchanged_original_500_runtime.json': ROOT/config['original_runtime'],
    'metadata_recovery_source.py': ROOT/'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
    'validation_tests.py': ROOT/'tests/test_fm_full500_metadata_recovery_v1.py',
    'validation.xml': xml,
    'prior_launch.json': base/'metadata_recovery_v1/prior_launch.json',
    'prior_status.json': base/'metadata_recovery_v1/prior_status.json',
    'prior_pending_status.json': base/'metadata_recovery_v1/prior_pending_status.json',
    'prior_failure_stderr.log': base/'metadata_recovery_v1/prior_stderr.log',
    'idle_only_recovery_preflight.json': base/'metadata_recovery_v1/recovery_preflight.json',
    'new_metadata_owner_launch.json': base/'metadata_recovery_launch_v1/launch.json',
}
for name, path in copies.items(): shutil.copyfile(path, folder/name)
write = recovery.runner.write
write(folder/'current_metadata_owner_observation.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), exact_live=True,
    pid=p.pid, create_time=p.create_time(), command=p.cmdline(), private_bytes=p.memory_info().private,
    actual_state=state, original_numerical_runtime_sha256=recovery.runner.sha(ROOT/config['original_runtime']),
    original_numerical_source_receipts_verified=True, new_metadata_stderr_bytes=0,
    numerical_models_completed_by_this_change=0, all_original_experiments_complete=False))
(folder/'README.md').write_text(
    '# 500项完整检查点复核的状态写入故障恢复\n\n'
    '原复核调度器PID35008已退出。保留的完整报错显示，它在替换status.next.json到status.json时收到Windows WinError5“拒绝访问”。'
    '这个记录定位了失败操作，不能单凭该记录认定具体哪个程序持有文件。原状态为等待训练完成，没有活跃数值子进程，'
    '没有任何复核尝试目录、数值产物或已完成/失败复核计数。\n\n'
    '独立恢复入口在确认旧PID及创建时间所指的进程已不存在、原冻结输入哈希不变、没有任何数值尝试后启动。'
    '它在新的元数据解释器中，仅替换原控制器的status.json写入函数：为每次发布创建独立临时文件，'
    '对Windows错误5/32/33最多尝试80次，每次间隔0.1秒；写入内容只计算一次。'
    '达到上限仍会失败并保留旧状态及临时诊断文件。其他错误立即传播，其他路径的写入全部委托给原函数。\n\n'
    '该重试只涉及状态文件重命名，不重试训练、推理或独立数值复核。原500个唯一模型、源文件和配置、'
    '资源保护、2e-6公式容差、2e-12指标容差与逐字节数组比较均未修改。数值子进程继续使用原模块和原运行配置，'
    '不继承元数据解释器内的替换函数。旧报错、旧状态、待替换文件及启动记录均按原字节封存。\n\n'
    '10项检查通过，覆盖访问冲突后相同内容重试、重试耗尽保留证据、其他错误立即失败、非状态写入不重试、'
    'PID身份核验、活跃子进程/已有数值尝试拒绝恢复，以及真实500项冻结契约与哈希核验。'
    '真实新控制器PID33020及创建时间已核实，恢复为等待原始完整训练完成，错误日志为空。\n\n'
    '本次修复没有新增模型性能结果。215项五种子统计仍等待完整复核证据；其他训练及队列继续。'
    '原45篇论文、681个方法/基线/消融记录、273个数据任务的最低范围保持，全部实验尚未完成。\n', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.log -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=10, original_models_preserved=500, new_numerical_models_completed=0,
    files_sha256={p.name: recovery.runner.sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(), config_path,
    'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
    'tests/test_fm_full500_metadata_recovery_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/Full500_metadata_recovery_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=10,metadata_owner=p.pid,new_numerical_models_completed=0)))
