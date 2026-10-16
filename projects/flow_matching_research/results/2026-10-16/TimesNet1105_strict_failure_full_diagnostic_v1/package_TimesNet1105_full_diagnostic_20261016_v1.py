from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import diagnose_timesnet1105_full_replay_v1 as s

ROOT,read,sha=s.ROOT,s.read,s.sha
write=s.old.exclusive
c=read(ROOT/s.CONFIG);stage=s.verify(c);base=ROOT/c['state_root'];launch=read(base/'launch.json')
actor=psutil.Process(launch['pid'])
assert actor.create_time()==launch['create_time'] and actor.cmdline()==launch['command']
assert not (base/'controller.stderr.log').read_bytes()
tests=Path(launch['validation_path']);assert sha(tests)==launch['validation_sha256']
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==81
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
assert not (Path(c['artifact_root'])/'full_replay_diagnostic.json').exists()
failure=s.preserved_failure(stage)
assert failure==c['preserved_failure']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet1105_strict_failure_full_diagnostic_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
for label,path in dict(original_native_failure_receipt=Path(failure['receipt']),
        original_native_failure_stderr=Path(failure['stderr']),
        original_point_success=Path(c['point_certificate']['proof']),
        actual_diagnostic_launch=base/'launch.json',actual_diagnostic_status=base/'status.json').items():
    shutil.copyfile(path,packet/(label+('.log' if label.endswith('stderr') else '.json')))
shutil.copyfile(tests,packet/'validation_81_cases.xml')
write(packet/'actual_live_diagnostic_handle.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
    children=[dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())
        for p in actor.children(recursive=True) if p.name().lower()=='python.exe'],
    original_numerical_processes_stopped=False))
write(packet/'original_full_geometry_and_resource_binding.json',dict(
    canonical_id=c['canonical_id'],settings=c['settings'],resource_lock=c['resource_lock'],
    cpu_threads=2,batch_size=128,parameters=28132633,window=100,
    entities=51,train_points=101565,validation_points=26684,test_points=408160,
    original_configured_epochs=3,actual_epoch_update_trace_available=False,
    checkpoint_model_and_score_code_changed=False,original_tolerance_changed=False,
    new_training_updates=0,original_scope=c['original_scope']))
write(packet/'iteration_status.json',dict(original_strict_native_failure_preserved=True,
    original_saved_score_point_audit_passed=True,new_diagnostic_controller_live=True,
    diagnostic_execution_complete=False,tests_passed=81,new_complete_benchmark_conditions=0,
    strict_native_reproduction_certified=False,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# TimesNet seed1105：保留严格失败，执行独立全量重放诊断

本次发现seed1105原3 epoch注册训练及全保存分数审计已完成，但原检查点严格重放失败。原进程退出1、资源守卫未中止，错误明确为“Complete original timestamp scores must match bitwise”。失败发生在分数核对阶段，原输出目录未留下完整重放向量，不能把它算作完整原生复现通过，也不能只根据全局F1接近而忽略逐点差异。

原失败receipt/stderr及成功point证据原样归档，原文件及其哈希继续绑定，原队列继续等待其他种子，不重试或覆盖该失败槽。

新增独立诊断保留28132633参数检查点、CPU float32、2线程、batch128、window100、全部51实体的训练101565/验证26684/测试408160点，以及原28GiB提交内存准入、原应急门槛和3/6GiB进程上限、原共享只读审计锁。诊断不训练，不修改权重、模型代码、原数据、阈值规则或原失败判据。

该诊断将完整保存全部153个原始split流的逐窗口分数、窗口起点和对齐后分数；任何差异逐流量化，没有新增allclose容差。原保存分数的所有既有指标需精确重核，再计算新重放的完整指标，并使用同一个原验证阈值比较预测变化。模型状态及全部源文件末尾再次校验。形状缺失、非有限值、原指标/检查点不一致仍直接失败。

全量诊断结束也不会自动认证原严格复现通过。即便某次新诊断恰好逐点一致，也不足以消除已有失败或证明历史后端原因。原作者等价、完整TAB、训练种子稳健性和全部实验范围均不由此认证。原配置只保存了3 epoch预算，不能冒称已有实际更新轨迹。

81项相关检查通过，包括原精确分数守卫、完整身份与资源绑定、单ULP差异保留、非有限/不完整拒绝、固定阈值严格>规则和禁止诊断覆盖失败认证。独立控制器已按真实PID/创建时间/完整命令确认存活，归档时尚未完成诊断；不计新增完整实验或新训练种子。

全部45论文/681方法记录/273数据任务及原论文/TAB/新数据、完整原有论文指标范围保留，整体任务未完成。
''',encoding='utf-8')
for name in ['launch_TimesNet1105_full_diagnostic_20261016_v1.py','package_TimesNet1105_full_diagnostic_20261016_v1.py']:
    shutil.copyfile(ROOT/'tmp'/name,packet/name)
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet1105_strict_failure_full_diagnostic.2026-10-16_v1.json'
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),original_failure_preserved=True,
    new_diagnostic_controller_live=True,diagnostic_execution_complete=False,tests_passed=81,
    new_complete_benchmark_conditions=0,all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/TimesNet1105_full_diagnostic_manifest_20261016_v1.json',dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),actual_diagnostic_pid=actor.pid,
    original_strict_failure_preserved=True,new_complete_benchmark_conditions=0)))
