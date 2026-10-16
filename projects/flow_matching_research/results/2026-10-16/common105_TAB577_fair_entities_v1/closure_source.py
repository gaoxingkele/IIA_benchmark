from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_common105_TAB577_fair_entities_v1 as s
ROOT,read,sha=s.ROOT,s.read,s.sha
from scripts.flow_matching.timegan_native_common_v1 import write
c=read(ROOT/s.CONFIG);s.verify(c);base=ROOT/c['state_root']
handle=read(base/'launch.json');p=psutil.Process(handle['pid'])
assert p.is_running() and p.create_time()==handle['create_time'] and p.cmdline()==handle['command']
assert not (base/'controller.stderr.log').read_bytes()
tests=Path(handle['validation_path']);assert sha(tests)==handle['validation_sha256']
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==44
assert not any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped'])
failure=Path('F:/aicoding/IIA_Data/experiments/common105_TAB577_fair_entity_continuation_20261016_v1_tests.xml')
assert sum(int(x.get('failures',0)) for x in ET.parse(failure).getroot().iter('testsuite'))==1
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/common105_TAB577_fair_entities_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
for name,path in dict(actual_launch=base/'launch.json',actual_scheduler_state=base/'status.json').items():shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(tests,packet/'validation_44_cases.xml')
shutil.copyfile(failure,packet/'preserved_stale_readiness_failure.xml')
for name in ['handoff_identity.json','handoff.json','previous_status_at_handoff.json','previous_status_original_bytes.json']:
    if (base/name).exists():shutil.copyfile(base/name,packet/name)
handoff=(base/'handoff.json').exists()
source=read(ROOT/s.matched.CONFIG)
write(packet/'unchanged_numerical_program_binding.json',dict(original_configuration=s.matched.CONFIG,
    original_configuration_sha256=sha(ROOT/s.matched.CONFIG),required_cases=577,required_label_views=1726,
    required_entity_label_views=77490,settings=source['settings'],resource_lock=source['resource_lock'],
    original_worker_module=s.matched.MODULE,original_metric_program_source_receipts=source['source_receipts'],
    numeric_source_and_guard_or_grid_modified=False))
write(packet/'iteration_status.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=c['original_scope'],
    scheduler_live=True,handoff_completed_at_capture=handoff,tests_passed=44,
    original_stale_state_test_failure_preserved=True,replacement_readiness_tests_cover_complete_and_missing_guard=True,
    new_completed_training_models=0,new_completed_TAB_cases=0,all_original_experiments_complete=False))
(packet/'README.md').write_text(f'''# 公共577项TAB队列：实体间公平让出复核锁

实际检查TimesNet复核等待的fm_saits_readonly_checkpoint_cpu.lock，Windows Restart Manager只读查询显示打开该文件的进程为公共TAB队列父进程32712；其真实数值子进程正在执行完整实体，不能清除该锁。
原公共队列在一个实体结束后立即进入下一个实体，可持续争用共享复核锁。新增独立元数据调度器沿用已验证的完整TAB边界接管及片段复用逻辑，在每个新完整实体完成后释放原锁并等待20秒，让其他原复核队列有机会进入。
接管必须先取得实际复核锁，再两次检查旧父进程的PID/创建时间/完整命令、无数值后代及精确子进程终止凭据；暂停和替换只针对已确认空闲的元数据父进程。数值子进程不会被停止。
全部577任务、1726标签视图、77490实体标签视图、21指标、每视图201调用、全部阈值/VUS窗口、完整批次、16/28/16GiB门槛、3/6GiB上限、守卫、源/原生凭据及科学输出路径保持原样。旧成功片段重新核验后复用；旧失败与未完成尝试保留，不重试、不重复计数。

44项相关验证通过。最初宽范围验证包含一个旧断言“所有原生模型仍未完成”，现在真实Reflow已完成，故该断言失败。旧文件由运行中的配置固定哈希，保持不变；原失败XML完整保留。新版两项测试保留所有家族原始守卫函数和精确命令检查，接受真正完整的原生凭据，并确认缺少实际review守卫时仍返回未就绪。后续验证仅排除这条失效状态断言，运行其余原测试及替代测试；无指标或资源条件放宽。

观察进程已按精确OS身份确认存活。此归档时接管完成状态={handoff}；实际阶段见actual_scheduler_state.json。启动观察调度器不代表接管已发生，更不代表任何TAB完整任务或数值训练新增完成。
旧数值作业、检查点、原始数据及所有45论文/681方法记录/273数据任务/702指标、原论文/TAB/新数据轨道继续保留；整体目标未完成。
''',encoding='utf-8')
shutil.copyfile('tmp/launch_common105_TAB577_fair_20261016_v1.py',packet/'launch_source.py')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/common105_TAB577_fair_entities.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),tests_passed=44,
    required_TAB_cases=577,handoff_completed_at_capture=handoff,all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,'tests/test_fm_common105_TAB577_readiness_progress_v2.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/common105_TAB577_fair_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),live_pid=p.pid,handoff_completed=handoff)))
