"""Launch only the newly registered metadata owner; preserve every old run."""
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil
from scripts.flow_matching import run_moment_icm_full_forecast_v1 as s

ROOT, read, sha, write = s.ROOT, s.read, s.sha, s.write
config = read(ROOT/s.CONFIG)
jobs = s.verify(config)
validation = ROOT/'tmp/MOMENT_ICM_full54_forecast_validation_20261016_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests', 0)) for x in suites) == 14
assert all(int(x.get(k, 0)) == 0 for x in suites for k in ['failures', 'errors', 'skipped'])
assert not s.ready_models(config)
state_root = ROOT/config['state_root']
assert not state_root.exists(), 'Do not restart or overwrite an existing owner'
for process in psutil.process_iter(['cmdline']):
    command = process.info['cmdline'] or []
    assert s.MODULE not in command, 'Existing forecast owner must be preserved'

state_root.mkdir(parents=True, exist_ok=False)
command = [sys.executable, '-B', '-X', 'utf8', '-u', '-m', s.MODULE]
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'))
with (state_root/'controller.stdout.log').open('xb') as out, (state_root/'controller.stderr.log').open('xb') as err:
    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command,
              configuration_sha256=sha(ROOT/s.CONFIG))
write(state_root/'launch.json', handle)
for _ in range(50):
    assert child.poll() is None, 'Owner exited; preserve logs and diagnose'
    if (state_root/'status.json').exists():
        break
    time.sleep(.1)
status = read(state_root/'status.json')
assert status['pid'] == handle['pid'] and status['create_time'] == handle['create_time']
assert status['state'] == 'waiting_both_actual_full39479update_PILE_models'
assert status['completed_count'] == 0 and status['active_pid'] is None
assert psutil.Process(handle['pid']).cmdline() == command

folder = ROOT/'projects/flow_matching_research/results/2026-10-16/MOMENT_all54_full_forecast_execution_registered_v1'
folder.mkdir(parents=True, exist_ok=False)
copies = {
    'frozen_all54_configuration.json': ROOT/s.CONFIG,
    'model_configuration.json': ROOT/s.MODEL,
    'full_forecast_validation.xml': validation,
    'controller_launch.json': state_root/'launch.json',
    'controller_status_at_registration.json': state_root/'status.json',
    'original_controller.py': ROOT/(s.MODULE.replace('.', '/')+'.py'),
    'original_forecast_implementation.py': ROOT/'scripts/flow_matching/moment_icm_forecast_finetune_v1.py',
    'original_forecast_tests.py': ROOT/s.TEST,
}
for name, source in copies.items():
    shutil.copyfile(source, folder/name)
    assert sha(source) == sha(folder/name)

rows = []
for job in jobs.values():
    binding = s.f.ready_data(job['dataset'], job['horizon'])
    assert binding is not None
    row = dict(job)
    for role, facts in binding['roles'].items():
        row[role+'_origins'] = facts['window']['windows']
    row['optimizer_updates'] = 5*((row['train_origins']+63)//64)
    rows.append(row)
stream = io.StringIO(newline='')
writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
writer.writeheader(); writer.writerows(rows)
(folder/'all54_registered_forecast_cases.csv').write_text(stream.getvalue(), encoding='utf-8', newline='')
hook_receipt = Path('C:/Users/10175/.codex/hook-repairs/verified-current-20261015T162850Z/validation.json')
hook = read(hook_receipt)
assert not hook['last_48h_hook_errors_or_warnings'] and not hook['configuration_changed']
write(folder/'current_hook_verification.json', dict(checked_at_utc=hook['checked_at_utc'],
    disabled_paseo_hooks=5, recent_48h_errors_or_warnings=[], configuration_changed=False,
    experiment_workers_modified_or_stopped=False, local_receipt=str(hook_receipt), receipt_sha256=sha(hook_receipt)))
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(), cases=54,
    datasets=6, horizons=[96, 192, 384], branches=list(s.f.BRANCHES), epochs_per_case=5,
    required_actual_pretraining_updates_per_model=39479, test_checks_passed=14,
    current_owner=handle, owner_state=status['state'], scientific_forecast_cases_completed=0,
    full_native_CUDA_bfloat16_training_validated=False,
    original_resource_policy_unchanged=True, frozen_head_and_beta_gradients_checked_on_CPU_fixtures=True,
    all_original_experiments_complete=False)
write(folder/'registration_summary.json', summary)
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
(folder/'README.md').write_text('''# Tiny/ICM：54 项完整预测实验已登记

范围为原文 Table 2 的三个分支 independent_head、icm_beta_frozen、icm_beta_finetuned，六个数据集 Exchange、ETTh1、ETTh2、ETTm1、ETTm2、Weather，以及三个预测长度 96/192/384，共 54 项。逐项完整窗口数和五轮优化更新数见 CSV。所有预测输入已经实际准备并审计；本次只登记后续模型训练任务，没有产生新的预测指标。

执行器保留相关作者源码的五轮训练主体、AdamW、OneCycleLR、训练批量 64、评估批量 256、全部角色 shuffle、5 个数据 worker、bf16 AMP、验证集选最佳检查点。256 历史长度和本地 Tiny/ICM 核心是已声明的论文重建选择，不认证与未公开历史实现完全相同。未假定附录 8fold/4fold 与 beta 分支存在对应关系。仅增加只读梯度检查；预测数组用磁盘保存，训练、验证和测试的完整抽样顺序均保留。

14 项测试全部通过，包括三个真实 Tiny 核心的头部与 beta 梯度、冻结参数不变、源码 AST 对照、完整五轮控制流程、原点覆盖和独立指标、Windows spawn 数据加载以及数值计算前的前置条件检查。这些 CPU 小样本测试只验证实现与流程；未据此宣称 CUDA/bf16 的科学实验成功。

新控制进程已确认 PID、创建时间和完整命令，当前等待两个完整 PILE 预训练模型。每个前置模型必须实际完成 2086 个来源、80,850,189 项、39,479 次更新并具有成功资源保护和完整检查点证明，之后才允许这 54 项数值训练。原有 GPU 锁、内存门槛、精度和误差容限保持不变。每项须保存完整五轮最终状态、验证选定模型、全部 TEST 预测，并另外完整重新加载模型推理及逐坐标复核。条件块自助区间不冒充训练随机种子不确定性。

本次也核验了当前 Codex 的 Hook 状态：五个此前修复的可选 Paseo 通知处理器均禁用，最近 48 小时未记录 Hook 错误或警告；没有重启实验进程。历史界面的具体错误条目无法仅凭“当前”二字逐条归因。

研究总目标仍未完成。其他论文、基线、消融、Table 1/4/5 和预训练中的周期异常检测任务继续保留，不由本队列替代。

来源：[ICM 原文](https://arxiv.org/abs/2409.13530v1)、[Autoformer 数据](https://github.com/thuml/Autoformer)、[OpenAI Hooks 文档](https://learn.chatgpt.com/docs/hooks)。
''', encoding='utf-8')
write(folder/'archive_validation.json', dict(**summary,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_all54_forecast_execution_registered.2026-10-16_v1.json'
write(ara, dict(registered_cases=54, scientific_cases_completed=0, full_pretrained_models_required=True,
    evidence=folder.relative_to(ROOT).as_posix(), all_original_experiments_complete=False))
paths = [s.CONFIG, s.MODEL, s.TEST, s.MODULE.replace('.', '/')+'.py',
         'scripts/flow_matching/moment_icm_forecast_finetune_v1.py', ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in sorted(folder.rglob('*')) if p.is_file()]
git = lambda cwd,*args: subprocess.check_output(['git','-C',str(cwd),*args], text=True).strip()
assert not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_all54_forecast_publication_manifest_20261016_v1.json',
    dict(previous_root=git(ROOT,'rev-parse','HEAD'), previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),
         paths=sorted(paths)))
print(json.dumps(dict(**summary, exact_files=len(paths))))
