"""Publish measured capacity and tested, unapplied administrator configuration."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.audit_windows_commit_admission_v1 import PROFILES,system_configuration,GIB

evidence=Path(r'F:\aicoding\IIA_Data\environments\windows_commit_capacity\read_only_diagnosis_20261016_v1')
diagnosis=read(evidence/'read_only_capacity_diagnosis.json')
plan=read(evidence/'reviewable_pagefile_plan.json')
validation=read(evidence/'actual_PowerShell_read_only_and_nonadmin_refusal_validation.json')
assert validation['real_Windows_settings_unchanged'] and validation['privileged_apply_not_executed']
assert validation['configuration_for_next_startup_not_installed']
assert not diagnosis['Windows_settings_changed'] and not diagnosis['numerical_processes_stopped']
assert diagnosis['memory']['independent_API_readings_agree']
assert not plan['before']['is_administrator'] and plan['before']['automatic_managed']
assert plan['proposed_paging_files']==[r'C:\pagefile.sys 0 0',r'F:\pagefile.sys 32768 32768']
current=system_configuration()
assert current['paging_files']==plan['before']['paging_files'] and current['automatic_managed']==plan['before']['automatic_managed']
assert not current['target_file_exists']
for row in diagnosis['profiles']:
    assert sha(ROOT/row['configuration'])==row['configuration_sha256']
    assert not row['observed']['actual_experiment_admitted']
    assert not row['counterfactual_with32GiB_pagefile_holding_usage_fixed']['actual_experiment_admitted']
xml=ROOT/'tmp/windows_commit_admission_validation_20261016_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==19
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Windows_commit_capacity_diagnosis_and_reviewed_F32_plan_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.ps1 -text\n*.xml -text -whitespace\n',encoding='utf-8')
for name in ['read_only_capacity_diagnosis.json','reviewable_pagefile_plan.json','actual_PowerShell_read_only_and_nonadmin_refusal_validation.json']:
    shutil.copyfile(evidence/name,packet/name)
shutil.copyfile(xml,packet/'validation_19_passed.xml')
rows=[]
for row in diagnosis['profiles']:
    s=row['settings'];o=row['observed'];p=row['counterfactual_with32GiB_pagefile_holding_usage_fixed']
    rows.append(dict(profile=row['profile'],configuration=row['configuration'],configuration_sha256=row['configuration_sha256'],
        required_physical_GiB=s['minimum_available_memory_bytes']/GIB,required_commit_GiB=s['minimum_commit_headroom_bytes']/GIB,
        observed_physical_GiB=diagnosis['memory']['physical_available_bytes']/GIB,observed_commit_GiB=diagnosis['memory']['commit_available_bytes']/GIB,
        measured_memory_gates_pass=o['memory_gates_pass'],projected_commit_GiB=p['commit_available_holding_charge_fixed']/GIB,
        projected_memory_gates_pass=p['memory_gates_pass'],actual_experiment_admitted=False))
with (packet/'measured_and_counterfactual_frozen_admission.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sources=['scripts/flow_matching/audit_windows_commit_admission_v1.py','scripts/flow_matching/configure_F_pagefile_reviewed_v1.ps1',
    'tests/test_fm_windows_commit_admission_v1.py','scripts/flow_matching/audit_commit_capacity_plan_v1.py','tests/test_commit_capacity_plan_v1.py']
write(packet/'source_and_config_receipts.json',dict(sources=[dict(path=p,sha256=sha(ROOT/p)) for p in sources],
    frozen_numerical_configurations=[dict(path=p,sha256=sha(ROOT/p)) for _,p,_ in PROFILES],
    actual_privileged_apply_tested=False,simulated_privileged_success_and_rollback_tested=True,
    actual_default_read_only_and_nonadmin_rejection_tested=True,Windows_configuration_unchanged=True,
    all_original_experiments_complete=False))
write(packet/'execution_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=diagnosis['original_scope'],new_complete_experiments=0,new_optimizer_updates=0,
    no_pressure_allocation=True,no_threshold_budget_precision_or_tolerance_changes=True,
    no_UAC_or_elevation_attempt=True,no_automatic_restart=True,no_worker_stop=True,
    system_settings_changed=False,configuration_requires_external_administrator_action=True,
    activation_may_require_subsequent_safe_startup=True,actual_capacity_increase_unproven=True,
    full_original_scope_complete=False))
limit=diagnosis['memory']['commit_limit_bytes']/GIB
available=diagnosis['memory']['commit_available_bytes']/GIB
(packet/'README.md').write_text(f'''# Windows 提交容量：实测瓶颈与尚未应用的 F 盘扩容方案

两套 Windows API 独立实测一致：物理总量约63.86GiB，当前提交上限{limit:.3f}GiB，可用提交量{available:.3f}GiB。
实际只有 C:\\pagefile.sys，当前分配4096MiB；全局自动管理开启，注册表配置为 `?:\\pagefile.sys`。
可用物理内存和提交余量是两种约束。4条原始配置的32/40GiB提交准入未满足；配置和真实读数、哈希及完整参数见JSON/CSV。
队列因此等待，不能据此判定训练算法出错，也不能改低准入或紧急保护门槛。

新增配置脚本 `benchmark/scripts/flow_matching/configure_F_pagefile_reviewed_v1.ps1` 默认只读，不请求UAC、不自行提权。
经核对的拟写入配置为 C:\\pagefile.sys 0 0 和 F:\\pagefile.sys 32768 32768：保留C盘按卷系统管理，在F盘增加固定32GiB。
应用时全局 AutomaticManagedPagefile 会从 true 改为 false，改用上述逐卷配置；不能称全局策略不变。
脚本要求F盘固定NTFS且剩余至少96GiB，拒绝覆盖已有F分页文件，核验计划和当前配置一致，先备份再写入，失败尝试恢复原配置。
脚本没有训练进程终止、重启、阈值修改或制造内存压力操作。

19项测试通过，包括规划边界、原配置保留、物理门槛不被提交投影代替、PowerShell解析，以及模拟CIM/注册表的成功写入与故障回滚。
在真实机器上只执行了默认只读模式和非管理员 `-Apply` 拒绝路径；前后注册表与自动管理标志一致，F分页文件未创建。
管理员实际应用分支尚未在真实系统执行，不能把模拟测试写成真实扩容成功。

管理员可在复核计划后执行：

```powershell
powershell.exe -NoLogo -NoProfile -NonInteractive -File "D:\\aicoding\\IIA_benchmark\\scripts\\flow_matching\\configure_F_pagefile_reviewed_v1.ps1" -PlanFile "F:\\aicoding\\IIA_Data\\environments\\windows_commit_capacity\\read_only_diagnosis_20261016_v1\\reviewable_pagefile_plan.json" -Apply
```

此命令未执行。原始计划如已过期，脚本会拒绝覆盖，需重新只读采集并复核。
分页配置可能到后续系统启动才生效；现有训练期间不重启。只有运行时API提交上限与余量确实增加，且所有物理/GPU/锁/稳定采样门槛满足，才能承认实际准入。
表中“增加32GiB”仅假设C盘当前分配和总提交用量不变；后续启动或其他负载会改变这些条件，不承诺全部队列同时可跑，也不把虚拟提交容量等同于实际RAM。

本包新增完整实验数为0、优化器更新为0。现有45篇/681方法基线消融/273原始数据任务/702论文标量目标保持，整体未完成。

依据：[Microsoft 分页文件配置与运行状态](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-pagefilesetting)，
[Microsoft 提交上限与系统管理分页文件](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'packaging_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Windows_commit_capacity_reviewed_F32.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),additional_complete_experiments=0,
    requires_external_administrator=True,Windows_configuration_changed=False,full_original_scope_complete=False))
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=sources[:3]+[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/windows_commit_capacity_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),tests_passed=19,new_completed_experiments=0)))
