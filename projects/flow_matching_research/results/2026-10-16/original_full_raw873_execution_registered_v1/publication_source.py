"""Start the cold873-case dispatcher once and archive reviewable source bindings."""
import csv
from datetime import datetime,timezone
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
from scripts.flow_matching import run_original_full_raw873_experiments_v1 as s

ROOT,read,write,sha=s.ROOT,s.read,s.write,s.sha
c=read(ROOT/s.CONFIG);families=s.verify(c)
xml=ROOT/'tmp/original_full_raw873_execution_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==32
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
assert s.ready_data(c) is None
base=ROOT/c['state_root'];assert not base.exists()
for p in psutil.process_iter(['cmdline']):assert s.MODULE not in (p.info['cmdline'] or [])
base.mkdir(parents=True,exist_ok=False)
command=[sys.executable,'-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command,configuration_sha256=sha(ROOT/s.CONFIG))
write(base/'launch.json',handle)
for _ in range(70):
    assert child.poll() is None,'Do not restart a failed or interrupted dispatcher'
    if (base/'status.json').exists():break
    time.sleep(.1)
state=read(base/'status.json')
assert state['pid']==handle['pid'] and state['create_time']==handle['create_time']
assert psutil.Process(handle['pid']).cmdline()==command
assert state['state']=='waiting_actual_full109_data_and_original_guard' and not state['actual_model_jobs_started']
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/original_full_raw873_execution_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
copies=dict(frozen_configuration=ROOT/s.CONFIG,original_execution_source=ROOT/(s.MODULE.replace('.','/')+'.py'),
    original_tests=ROOT/s.TEST,validation_32_checks=xml,controller_launch=base/'launch.json',status_at_registration=base/'status.json')
for name,path in copies.items():
    target=folder/(name+path.suffix);shutil.copyfile(path,target);assert sha(path)==sha(target)
rows=[];source_bindings=[]
for family in families:
    queue=read(ROOT/family['source_queue']);selected=s.selected(queue)
    assert len(selected)==family['required_cases']
    source_bindings.append(dict(family=family['name'],queue=family['source_queue'],queue_sha256=sha(ROOT/family['source_queue']),
        required_cases=len(selected),source_model_worker_unchanged=True))
    for job in selected:
        model=read(ROOT/job['model_config'])
        rows.append(dict(family=family['name'],source_job_id=job['id'],full_raw_job_id=job['id']+'__original_full109_v1',
            dataset=job['dataset'],seed=job['seed'],window=job['window'],device=job['parameter_overrides']['device'],
            model_configuration=job['model_config'],model_configuration_sha256=sha(ROOT/job['model_config']),
            source_total_training_epochs=job.get('total_training_epochs'),
            unchanged_source_model_parameters=json.dumps(model['parameters'],sort_keys=True),
            native_worker_python=queue['python'],unchanged_evaluation=json.dumps(queue['evaluation'],sort_keys=True)))
assert len(rows)==873 and len({r['full_raw_job_id'] for r in rows})==873
stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(folder/'all873_source_method_dataset_seed_bindings.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'source_queue_validation.json',source_bindings)
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),required_cases=873,cases_per_dataset=dict(smap=291,msl=291,smd=291),
    family_counts={f['name']:f['required_cases'] for f in families},tests_passed=32,actual_dispatcher_handle=handle,
    actual_data_preparation_complete=False,actual_new_numerical_model_jobs_started=0,
    queues_materialized_against_actual_full_data=False,source_numeric_workers_and_budgets_unchanged=True,
    independent_model_replay_complete=False,full_label_controls_and_TAB_metrics_complete=False,
    full_research_scope_reduced_to_this_queue=False,all_original_experiments_complete=False)
write(folder/'registration_summary.json',summary)
(folder/'README.md').write_text('''# 完整原始实体上的 873 项本地方法实验：自动衔接执行器已启动

本次把完整 109 实体的数据准备与现有七类本地实验相接。已登记 873 个来源任务，SMAP、MSL、SMD 各 291 项。逐算法、数据集、种子、上下文、设备、模型配置路径/哈希和完整参数列在 CSV，未以简化主方法名单替代原消融。

|来源任务族|完整原始三数据集任务数|
|---|---:|
|原流匹配目标|120|
|神经网络方法与消融|510|
|强基线及 USAD 符号对照|105|
|熵正则数值修复对照|45|
|迭代 reflow|60|
|TranAD float64 对照|15|
|MOMENT 三检查点 × 两上下文|18|

这 18 个 MOMENT 任务是原冻结权重的评估，原 seed42 保留，不冒充五个独立训练种子。其余任务保持来源种子、数据集专用模型配置、轮数、精度、设备和评价设置。MOMENT 继续使用原 512/100 上下文及短段右端复制、裁剪回原始点数的既有实现，不改检查点权重或分数公式。

32 项测试通过，包括全部 873 项物化、每种来源字段/数值设置不变、每数据集专用强基线配置、旧与新全部数据来源哈希保留、GPU 原阶段门槛保留、资源门槛不可降低、历史尝试不可覆盖和数据准备失败不能被隐藏为无限等待。测试只验证衔接与契约，没有原始数据上的新模型性能。

控制进程已按 PID、创建时间、完整命令核验。当前等待真实 109 实体准备报告及成功资源保护回执，尚未启动这 873 项数值模型。数据准备完成后，才会在 F 盘固定实际 manifest、完整数组及原始来源哈希，生成七类完整队列，并自动启动原入口的轻量控制进程。旧数值 worker、MOMENT 入口、CPU 原 10/12 GiB RSS/private 上限、物理/提交量门槛及锁、GPU 原内存保护和 imputation 阶段门槛保持。成功与失败原尝试均保留，不自动重试。

原训练/评分完成与独立检查点重推理、所有标签覆盖评价、五种子归约及完整 TAB 范围指标分别记账。不能把旧评分程序返回成功当作这些后续审计完成，也不能按测试指标选标签或预处理。NASA 来源预缩放、P-2 标注歧义和新短序列切分边界继续保留。

873 项是新增完整实体对照，不缩小此前研究任务。作者专用 MaelNet、Pi、CrossAD 等流程、Tiny/ICM 全预训练、其他数据集、周期异常检测和至少 45 篇论文/681 方法记录/273 数据任务及 702 标量目标继续保留。整体研究尚未完成。

来源：[Telemanom](https://github.com/khundman/telemanom)、[TranAD](https://github.com/imperial-qore/TranAD)、[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)、[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(**summary,files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/original_full_raw873_execution_registered.2026-10-16_v1.json'
write(ara,dict(required_cases=873,families=7,implementation_tests_passed=32,evidence=folder.relative_to(ROOT).as_posix(),
    actual_new_numerical_model_jobs_started=0,full_original_scope_remains_required=True,all_original_experiments_complete=False))
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/original_full_raw873_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(**summary,exact_files=len(paths))))
