from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_common105_matched577_labels_v1 as scalar
from scripts.flow_matching import replay_common105_matched577_v1 as native
from scripts.flow_matching import run_common105_matched577_TAB_v1 as tab

ROOT=Path.cwd()
OUT=ROOT/'projects/flow_matching_research/results/2026-10-16/common577_full_validation_pipeline_v1'
OUT.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def live(h):
    try:
        p=psutil.Process(h['pid']);actual=p.cmdline()
        same=abs(p.create_time()-h['create_time'])<.01
        if h.get('command'):same=same and actual==h['command']
        return dict(expected=h,alive=p.is_running(),exact_identity=same,actual_command=actual,
            actual_create_time=p.create_time(),cpu_seconds=sum(p.cpu_times()[:2]))
    except psutil.NoSuchProcess:return dict(expected=h,alive=False,exact_identity=False)

modules=[scalar,native,tab]
configs=[scalar.CONFIG,native.CONFIG,*native.CONFIGS.values(),tab.CONFIG]
paths=[m.MODULE.replace('.','/')+'.py' for m in modules]+[m.TEST for m in modules]+configs
snapshots=[]
for i,path in enumerate(paths):
    src=ROOT/path;dest='source%02d%s'%(i,src.suffix)
    shutil.copy2(src,OUT/dest)
    snapshots.append(dict(path=path,sha256=sha(src),snapshot=dest))
validation=[]
for name,expected,filename in [
    ('scalar',10,'common105_matched577_labels_tests_20261016_v1.xml'),
    ('native',21,'common105_native577_tests_20261016_v1.xml'),
    ('TAB',12,'common105_matched577_TAB21_tests_20261016_v1.xml')]:
    source=Path('F:/aicoding/IIA_Data/experiments')/filename
    suites=list(ET.parse(source).getroot().iter('testsuite'))
    totals={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert totals==dict(tests=expected,failures=0,errors=0,skipped=0),totals
    dest=name+'_validation.xml';shutil.copy2(source,OUT/dest)
    validation.append(dict(suite=name,**totals,source=str(source),sha256=sha(source),snapshot=dest))
for module in modules:module.verify(read(ROOT/module.CONFIG))
sc=read(ROOT/scalar.CONFIG);nc=read(ROOT/native.CONFIG);tc=read(ROOT/tab.CONFIG)
source_cases=[dict(case_id=case['id'],status=scalar.original.source_status(case)) for case in sc['cases']]
completed=sum(bool(x['status']) and x['status']['kind']=='source_complete' for x in source_cases)
write('actual577_source_guard_statuses.json',source_cases)
source=read(ROOT/scalar.source.CONFIG)
handles=[]
source_dispatch=read(ROOT/source['state_root']/'status.json')
native_dispatch=read(ROOT/nc['state_root']/'status.json')
states={}
for kind,items,base in [
    ('source',source_dispatch['family_handles'],ROOT/source['state_root']/'families'),
    ('native',native_dispatch['family_handles'],ROOT/'experiments/runs/fm_common105_native577_v1')]:
    for h in items:
        checked=live(h);assert checked['alive'] and checked['exact_identity'],checked
        handles.append(dict(kind=kind,**checked))
        states[kind+'_'+h['family']]=read(base/h['family']/'status.json')
for kind,c in [('scalar',sc),('TAB',tc)]:
    state=read(ROOT/c['state_root']/'status.json')
    h={k:state[k] for k in ['pid','create_time']}
    h['command']=[c.get('controller_python',tc['metric_python']),'-B','-X','utf8','-u','-m',scalar.MODULE if kind=='scalar' else tab.MODULE]
    checked=live(h);assert checked['alive'] and checked['exact_identity'],checked
    handles.append(dict(kind=kind,**checked));states[kind]=state
write('actual15_controller_identities_and_states.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles,states=states))

scientific=[dict(pid=36912,create_time=1792097263.1796665),dict(pid=40268,create_time=1792097260.102444),dict(pid=41992,create_time=1791948094.8603492)]
before=[live(h) for h in scientific];time.sleep(2);after=[live(h) for h in scientific]
write('actual_existing_training_cpu_observations.json',dict(before=before,after=after,interpretation='Exact live PID/create-time/command observations; CPU advancement is activity, not completed scientific results.'))
for name,path in [('original_TAB','experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json'),
    ('resource_recovery','experiments/runs/fm_TAB878_resource_durable_recovery_v2/status.json')]:
    state=read(ROOT/path);owner=live({k:state[k] for k in ['pid','create_time']})
    write('actual_'+name+'_state.json',dict(state=state,owner=owner,
        active_child=live(dict(pid=state['active_pid'],create_time=psutil.Process(state['active_pid']).create_time())) if state.get('active_pid') and psutil.pid_exists(state['active_pid']) else None))

fixtures=[]
base=Path('F:/aicoding/IIA_Data/experiments/common105_native577_unit_fixtures_20261016_v1')
for folder in sorted(base.iterdir()):
    if not folder.is_dir() or not (folder/'fixture_metadata.json').exists():continue
    meta=read(folder/'fixture_metadata.json');family=meta['family'];reports=[]
    for phase,filename in native.component(family).REPORTS.items():
        p=folder/'replay'/'fixture_only_not_benchmark'/phase/filename
        proof=read(p);assert proof['passed'];reports.append(dict(phase=phase,path=str(p),sha256=sha(p),passed=True))
    fixtures.append(dict(family=family,metadata_path=str(folder/'fixture_metadata.json'),metadata_sha256=sha(folder/'fixture_metadata.json'),
        model=meta['model'],reports=reports,synthetic_only_not_benchmark_or_CUDA=True))
assert len(fixtures)==2
write('synthetic_full_local_budget_native_review_only.json',fixtures)
manifest=dict(captured_utc=datetime.now(timezone.utc).isoformat(),new_sources=577,label_views=1726,entity_label_views=77490,
    comparison=nc['original_matched_comparison'],full_scope=nc['original_scope'],snapshots=snapshots,validation=validation,
    passed_tests=43,actual_new_sources_completed=completed,all_original_experiments_complete=False,
    official_TAB_source_commit='5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2',
    changed_numerical_definitions=False,changed_model_budgets=False,changed_resource_policies=False,
    publication_limit='Original TAB author source remains local; archived hashes/configuration do not imply redistribution rights. Synthetic fixtures are plumbing evidence only.')
write('archive_validation.json',manifest)
shutil.copy2(__file__,OUT/'publication_source.py')
(OUT/'README.md').write_text('''# 公共切分 577 项实验的完整审计、回放与 TAB 流程

577 项新源实验接入完整保存分数审计、六类检查点原生回放和独立标签复核，以及 TAB 原始 21 指标评估。保存分数审计与回放覆盖全部 1726 组标签口径；TAB 覆盖 77490 组实体标签。保留原 6 个分数指标、15 个标签指标、每组实体标签 201 次调用、3 个验证阈值、10 个测试集诊断阈值、全部 VUS 整数范围和 250 阈值网格。测试集诊断不能冒充验证集冻结阈值下的性能。

六类回放分别覆盖 FM 110/330、神经方法 340/1020、基线 65/190、Reflow 40/120、TranAD64 10/30、MOMENT 12/36 项模型/标签。复用原完整数值实现，只重新绑定新数据、源守卫、配置、命令及输出身份。原训练预算、模型容量、求解器、精度、随机种子、设备、解释器、数学容差、资源门槛和阶段条件均保留。每项都必须取得实际源运行、分数审计、检查点回放及独立复核证据，才进入相应后续阶段。

TAB 增加逐指标原子保存，输入参数、函数定义与源证据均绑定哈希。中间记录仅为检查点；完整实体通过实际资源守卫后才认证结果。原失败不覆盖，未定义值与异常不补零，不自动重跑资源中止任务。原 16/28/16 GiB 及 3/6 GiB 上限和审计锁保留。

相关测试共 43 项通过（分数审计 10、原生/复核 21、TAB 12），实际注册配置的冷检查通过。两个合成小输入用例保留原本地模型容量和训练预算：DCD 3 轮、独立 FM 20 轮，完成完整检查点重载与独立子进程复核。这些仅验证流程，不能作为真实数据集或 CUDA 成绩，也不构成作者论文效果复现证明。

归档时已逐一核验 7 个源队列、6 个原生队列以及分数审计/TAB 共 15 个控制器的实际 PID、创建时间和完整命令。具体完成数见 archive_validation.json；新源实验仍等待原资源及阶段条件，没有将排队注册记为完成。原 TAB 队列继续处理已真实闭合的 TranAD 源结果；另存其实际实体进度及资源恢复等待状态。已有 CFM、BRITS、CrossAD 实际训练进程的两次 CPU 观测单独保存，活动不等于科学完成。

完整公共对比仍为 873 项/2037 组标签，包含本次 577 项、另外已有的 5 项 DCD 控制及 291 项数据逐字节相同的 SMD 引用。SMD 引用保留原身份，不算新独立重复。原始源队列、45 篇论文、681 项方法/基线/消融记录、273 项数据任务和 702 项论文标量目标，以及其余作者队列、预处理、五训练种子统计和完整 TAB 训练协议要求继续保留；本流程接入不能称为整体完成。

源脚本、测试和配置按 sourceNN 文件保存，映射与 SHA256 见 archive_validation.json。原 TAB 作者代码只保留本地固定提交及文件哈希引用，不因接入而推定具有再分发许可。原始数据、权重及分数数组均留在 F 盘注册目录。
''',encoding='utf-8',newline='\n')
publish=paths+[p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()]
packet=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(publish))
(ROOT/'tmp/common577_pipeline_exact_publication_v1.json').write_text(json.dumps(packet,indent=2),encoding='utf-8')
print(json.dumps(dict(packet=str(OUT),validated_tests=43,actual_live_controllers=len(handles),files=len(publish))))
