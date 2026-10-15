"""Publish full five-seed analysis registration and its actual cold owner."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_objective_full440_statistics_v1 as runner
from scripts.flow_matching.register_objective_full440_statistics_v1 import CONFIG

ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
runtime=read(ROOT/CONFIG);source,cases,tasks=runner.verify(runtime)
base=ROOT/runtime['state_root'];launch=read(base/'launch.json');p=psutil.Process(launch['pid'])
assert p.create_time()==launch['create_time'] and runner.MODULE in p.cmdline()
assert launch['configuration_sha256']==sha(ROOT/CONFIG)
state=read(base/'status.json')
assert state['pid']==p.pid and state['create_time']==p.create_time()
assert state['required_cases']==440 and state['required_tasks']==206
assert state['completed_count']==0 and state['unestimable_count']==55 and state['failed_statistics']=={}
assert state['active_pid'] is None and not (base/'stderr.log').read_bytes()
for identity,negative in state['terminal_unestimable'].items():
    assert negative and set(negative).issubset(source['preserved_original_failures'])
    assert set(negative).issubset(tasks[identity]['cases'])
    assert all(source['preserved_original_failures'][i]==row for i,row in negative.items())
tests=ROOT/'tmp/Objective_full440_statistics_validation_20261015_v2.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==25
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/objective_full440_five_seed_statistics_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {
    'frozen_statistics_configuration.json':ROOT/CONFIG,
    'frozen_original_full440_replay_configuration.json':ROOT/runner.PRODUCER,
    'full_statistics_source.py':ROOT/'scripts/flow_matching/run_objective_full440_statistics_v1.py',
    'registration_source.py':ROOT/'scripts/flow_matching/register_objective_full440_statistics_v1.py',
    'validation_tests.py':ROOT/'tests/test_fm_objective_full440_statistics_v1.py',
    'validation.xml':tests,
    'actual_controller_launch.json':base/'launch.json',
}.items():shutil.copyfile(path,folder/name)
case_inventory=[]
for identity,case in cases.items():
    j=case['job'];case_inventory.append(dict(id=identity,family=case['family'],dataset=j['dataset'],seed=j['seed'],
        model_config=j['model_config'],queue=case['path'],source_result_exists=(ROOT/j['output_directory']/'result.json').exists(),
        original_terminal_failure=identity in source['preserved_original_failures'],
        supersedes_failed_numerics_job=j.get('supersedes_failed_numerics_job'),
        preprocessing_control_original_id=j.get('preprocessing_control_original_id')))
write(folder/'all440_source_case_inventory.json',dict(cases=case_inventory,
    source_result_existence_is_not_native_verification=True,all_original_experiments_complete=False))
write(folder/'all206_task_bindings.json',dict(tasks=list(tasks.values()),
    kind_counts=dict(Counter(t['kind'] for t in tasks.values())),
    contrast_counts=dict(Counter(t['contrast'] for t in tasks.values() if t['kind']=='paired'))))
write(folder/'original105_failure_outcomes_and_55_unestimable_analyses.json',dict(
    original_failures=source['preserved_original_failures'],affected_tasks=state['terminal_unestimable'],
    affected_kind_counts=dict(Counter(tasks[i]['kind'] for i in state['terminal_unestimable'])),
    original_failures_replaced_by_repairs=False,successful_new_statistics_tasks=0))
write(folder/'actual_metadata_owner_observation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    exact_live=True,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private,
    actual_state=state,new_numerical_model_completions=0,new_five_seed_performance_estimates=0,
    all_original_experiments_complete=False))
(folder/'README.md').write_text(
    '# 440个流匹配个案的完整五种子对比与消融统计\n\n'
    '本轮补齐训练及检查点复核之后的统计阶段，原训练、推理、数据及冻结输入不改动。'
    '范围是四条分别登记的本地异常检测适配轨道：原版280项、数值修正105项、SMAP常量特征预处理对照40项、'
    '常量特征对照中的数值修正15项。共440个不同个案、88个完整五种子队列，种子固定1103至1107。'
    '这些是本地时序适配器，不能据此宣称已复现原始图像/单细胞论文的算法或效果。\n\n'
    '共登记206项分析：88项方法/数据集/轨道队列统计；80项相对独立CFM的同种子配对；'
    '11项按已登记原始ID绑定的预处理对照；16项SF2M分数头开启减去flow-only的直接消融；'
    '11项七数据集宏平均。完整绑定见JSON，不是笛卡尔积补造的模型或新增训练。'
    '宏平均先在相同种子内等权平均PSM、SMAP、MSL、SMD、SWaT、SWaN和CICIDS，再对五个种子计算区间，'
    '不会把35个数据集/种子单元伪装成35个独立训练种子。\n\n'
    '每个正向结果必须先通过原440项复核中的完整native及独立审计，并重新核对源结果/检查点/数据/分数哈希、'
    '保护回执、完整命令及产物哈希。已有SB/SWaT证书继续走原登记复用路径，不归因为新的native推理。'
    '保存的源训练保护历史未证实时继续保留该限制，理想float64路径等价性未通过也继续保留，'
    '不会被新的种子平均或资源保护回执消除。\n\n'
    '主指标固定为验证集1%阈值、未作PA的点级micro F1。另行保留验证阈值0.1%/1%/5%的点级、分组F1、排序及事件指标，'
    '以及十点池化训练/测试分数阈值网格下未调整F1/PA F1和测试标签选优诊断。'
    '原单模型200次时序分块bootstrap区间不作为新的训练种子，也不拿区间上下界再冒充种子性能。'
    '配对要求相同种子和数据集，逐项匹配完整split覆盖、测试标签以及存在的时间戳/源行号/故障标签；'
    '每个模型仍使用自己完整的原验证校准。预处理对照允许输入特征值不同，但不允许样本支持集不同。\n\n'
    '五种子统计给出均值、样本标准差、标准误、未裁剪的df4 Student t 95%区间、中位数/范围及逐种子留一均值。'
    '它们只描述当前固定数据及训练种子的变化，不提供总体泛化保证、假设检验或多重比较显著性认证。'
    '任一指标含原始未定义值时保留全部五个位置，不删除该种子，不报告部分种子置信区间。\n\n'
    '原版105个已终止失败记录完整保留、未重跑，也没有用修正版覆盖。它们使55项分析无法形成完整估计：'
    '21项队列、31项配对和3项宏平均。这些是已知负面结果的传播，不是55项成功的性能实验；'
    '其余151项仍等待实际完整模型及复核证据，后续新失败也会保留并传播。\n\n'
    '25项检查通过，包括完整206/440绑定、分析数学、原失败隔离、指标协议分离、时间戳变更拒绝、'
    '实际产物哈希变更拒绝，以及队列/配对/宏平均的端到端数学与序列化fixture。'
    '这些fixture不是benchmark结果。新控制器PID7324及创建时间已核实，当前无数值子进程、错误日志为空。'
    '数值统计仍受原16/32/16GiB保护和10/12GiB上限约束，没有减少训练预算或放宽数值容差。\n\n'
    '本轮没有新增模型训练、推理或可发布的五种子性能数值。VUS/Affiliation积分独立复算、预处理数值重建、'
    '完整TAB及历史论文等价性不由此认证。原45篇/681方法记录/273数据任务的最低研究范围及其他未完成实验继续保留。\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=25,cases=440,tasks=206,original_failures_preserved=105,currently_unestimable_tasks=55,
    new_five_seed_performance_estimates=0,new_model_training_or_inference=0,
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),CONFIG,
    'scripts/flow_matching/run_objective_full440_statistics_v1.py',
    'scripts/flow_matching/register_objective_full440_statistics_v1.py',
    'tests/test_fm_objective_full440_statistics_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/Objective_full440_statistics_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=25,controller=p.pid,new_performance_estimates=0)))
