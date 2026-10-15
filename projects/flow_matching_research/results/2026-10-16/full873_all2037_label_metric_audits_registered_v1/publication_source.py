import csv,io,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as s

ROOT=s.ROOT
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert s.sha(path)==s.sha(dest)
c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time']
assert p.cmdline()==launch['command'] and state['state']=='waiting_actual_full873_source_results_and_successful_guards'
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG)
assert not state['failed_audits_preserved'] and not state['source_failures_preserved']
binding=s.source.ready_data(s.read(ROOT/c['source_execution']));assert binding is not None
validation=ROOT/'tmp/full873_all2037_label_metrics_final_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==41
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probes=[]
for executable in [c['controller_python'],next(r['worker_python'] for r in cases if r['family']=='moment')]:
    command=[str(ROOT/executable),'-B','-X','utf8','-m',s.MODULE,'--probe-only']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True)
    probe=json.loads(result.stdout)
    assert probe['passed'] and probe['cases']==873 and probe['label_views']==2037 and not probe['numpy_imported'] and not probe['torch_imported']
    probes.append(dict(command=command,result=probe))
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full873_all2037_label_metric_audits_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {
    'original_metric_audit_source.py':ROOT/(s.MODULE.replace('.','/')+'.py'),
    'original_metric_audit_tests.py':ROOT/s.TEST,'frozen_all873_all2037_configuration.json':ROOT/s.CONFIG,
    'validation41.xml':validation,'exact_controller_launch.json':ROOT/c['state_root']/'launch.json'}.items():copy(path,folder/name)
rows=[]
for case in cases:
    ds=binding['records'][case['dataset']]
    for policy in case['required_label_policies']:
        rows.append(dict(id=case['id'],family=case['family'],dataset=case['dataset'],seed=case['seed'],window=case['window'],
            label_policy=policy,is_primary_label_policy=policy==ds['primary_label_policy'],
            full_entities=len(ds['entities']),full_test_points=sum(e['test_points'] for e in ds['entities']),
            source_queue=case['queue_path'],source_guard=case['source_guard'],worker_python=case['worker_python'],
            registration_is_benchmark_performance=False))
assert len(rows)==2037 and len({(r['id'],r['label_policy']) for r in rows})==2037
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'registered_all2037_tasks_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'actual_controller_and_cross_environment_probes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    live_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),
        children=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline()) for x in p.children(recursive=True)]),
    state_snapshot=state,cold_native_probes=probes,full_prepared_data_binding=binding,
    all41tests_passed=True,registered_case_counts=dict(Counter(r['dataset'] for r in cases)),
    registered_label_view_counts=dict(Counter(r['dataset'] for r in rows)),
    original_model_native_reinference_certified_by_this_audit=False,full_TAB_or_original_author_equivalence_certified=False,
    all_original_experiments_complete=False))
failed=ROOT/'tmp/full873_all2037_label_metrics_validation_20261016_v1.xml'
preserved=Path('F:/aicoding/IIA_Data/experiments/original_full873_all2037_metric_validation_v1/initial_fixture_provenance_failures.xml')
copy(failed,preserved)
fs=list(ET.parse(failed).getroot().iter('testsuite'));assert sum(int(x.get('failures','0')) for x in fs)==7
write(folder/'preserved_pre_registration_fixture_failures.json',dict(path=str(preserved),sha256=s.sha(preserved),failed=7,passed=33,
    cause='Tiny unit fixture used placeholder strings where strict calibration requires lowercase64character SHA256',
    correction='Use valid synthetic provenance hashes;no real metric formula,data,threshold or source experiment changed',
    final_tests_passed=41,failed_real_benchmark_attempts_created=False))
(folder/'README.md').write_text('''# 873项实验的全部2037标签视图评估已实现、测试并启动

此前全量109实体的数据准备与873项队列已完成接入。本次补齐这些新实验完成后的只读指标评估：每个数据集291项，SMAP4种原始标签视图共1164组、MSL2种共582组、SMD1种共291组，合计2037组。完整任务清单是注册范围，不是已完成成绩；不缩减原45篇论文、681方法记录、273数据任务和702篇内指标的范围。

评估器仅在原任务成功退出且对应原CPU/GPU守卫、任务指纹、队列和模型配置均一致后处理结果。源端失败作为无指标的负面结果留存；成功返回却缺少完整结果会报错。模型、原始数据、准备后数据、标签覆盖、分数及原守卫均重查哈希。全部实体的TRAIN/验证/TEST分数长度、有限性及原标签逐数组检查，原指标、事件、200次实体重采样区间和完整诊断阈值网格必须精确重算相等。

之后每个预先声明的标签覆盖使用同一份预测和原验证阈值，保留所有原0.1/1/5验证分位设置、bootstrap次数/种子/分组以及全部诊断网格；另用独立混淆计数、排名、分段PA和分位阈值计算交叉检查。主标签覆盖必须精确恢复原指标。允许公开所有标签视图的差异，不允许按TEST成绩选择标签版本或分数方向。置信区间只条件于当前模型/阈值，五个训练种子的配对统计仍须另外完成。

41项测试通过，包含源任务失败/错误队列/预算变化/指纹错误/缺点/非有限值/错误标签/遗漏标签视图拒绝，以及各视图预测与验证阈值保持不变。Python3.12和作者Python3.10中的完整注册对象逐字段相同，两环境实际冷检查都未导入NumPy或Torch，未启动模型。原MOMENT分段padding与bootstrap块长设置沿用不变。小型测试夹具不代表benchmark成绩。

实际控制器PID、创建时间和命令已核验，正在等待完整源结果及成功守卫。它沿用只读CPU审计资源策略：16GiB物理、28GiB提交余量，16GiB紧急线，3/6GiB RSS/private上限和既有只读审计锁。没有降低资源或预算、改动活跃训练、自动重试失败，或发送外部通知。

本流程是保存分数的独立指标评估，不是从checkpoint重新加载模型并重新推理；完整模型回放仍须逐方法完成。完整TAB执行器、VUS/Affiliation底层积分、历史作者数据/预处理一致性也未由本次认证。CSV任务清单和控制器等待状态不得作为实验成绩或完成数。全部研究目标继续进行。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    tests_passed=41,completed_real_model_inferences_added=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full873_all2037_label_metric_audits_registered.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),required_model_dataset_seed_cases=873,required_label_views=2037,
    implemented_tested_and_controller_live=True,tests_passed=41,actual_new_native_model_replays=0,
    original_complete_scope_retained=c['original_scope'],all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full873_all2037_label_metrics_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),cases=873,views=2037,tests_passed=41,live_controller_pid=p.pid,
    actual_completed_label_results=state['completed_label_view_results'],new_model_reinferences=0)))
