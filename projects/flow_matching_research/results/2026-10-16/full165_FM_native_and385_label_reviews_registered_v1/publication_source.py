import csv,io,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full165_fm_objectives_v1 as s

ROOT=s.ROOT
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert s.sha(path)==s.sha(dest)
c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
launch=s.read(ROOT/c['state_root']/'launch.json');state=s.read(ROOT/c['state_root']/'status.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG) and not state['failed_replays_preserved']
validation=ROOT/'tmp/original_full165_fm_replays_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==40
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probe_cmd=[c['controller_python'],'-B','-X','utf8','-m',s.MODULE,'--probe-only']
probe=json.loads(subprocess.run(probe_cmd,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True).stdout)
assert probe['passed'] and probe['cases']==165 and probe['label_views']==385 and not probe['numpy_imported'] and not probe['torch_imported']
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full165_FM_native_and385_label_reviews_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {'original_FM_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),'original_FM_native_tests.py':ROOT/s.TEST,
    'frozen_all165_native_configuration.json':ROOT/s.CONFIG,'validation40.xml':validation,
    'exact_controller_launch.json':ROOT/c['state_root']/'launch.json'}.items():copy(path,folder/name)
rows=[];variants={}
for case in cases:
    queue,job=s.scalar.resolve_case(case);m=s.read(ROOT/job['model_config']);s.validate_model(c,case,queue,job,m)
    model_key=job['model_config'];variants[model_key]=dict(citation=m['citation'],parameters=m['parameters'],
        model_configuration=model_key,model_configuration_sha256=s.sha(ROOT/model_key))
    for label in case['required_label_policies']:
        rows.append(dict(id=case['id'],family=case['family'],dataset=case['dataset'],seed=case['seed'],label_policy=label,
            model_configuration=job['model_config'],objective=m['parameters']['objective'],score_weight=m['parameters']['score_weight'],
            source_epochs=m['parameters']['epochs'],hidden=m['parameters']['hidden_size'],batch=m['parameters']['batch_size'],
            MC_samples=m['parameters']['monte_carlo_samples'],score_times=json.dumps(m['parameters']['score_times']),
            CPU_threads=queue['evaluation']['cpu_threads'],source_queue=case['queue_path'],registration_is_benchmark_performance=False))
assert len(rows)==385 and len(variants)==11
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all385_registered_label_views_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'all11_frozen_variants_and_citations.json',dict(variants=variants,original_cases=120,separate_numeric_control_cases=45,
    no_original_failed_attempt_rewritten_or_automatically_retried=True))
write(folder/'actual_live_owner_and_pending_native_outcomes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    live_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])),
    status_snapshot=state,cold_probe=dict(command=probe_cmd,result=probe),registered_case_counts=dict(Counter(x['dataset'] for x in cases)),
    registered_family_counts=dict(Counter(x['family'] for x in cases)),actual_completed_FM_native_cases=len(state['completed_cases']),
    synthetic_fixture_training_is_benchmark_performance=False,all_original_experiments_complete=False))
other=s.read(ROOT/c['separate_TranAD_native_component']);osnap=s.read(ROOT/other['state_root']/'status.json')
op=psutil.Process(osnap['pid']);assert op.create_time()==osnap['create_time']
upstream=s.read(ROOT/c['upstream_scalar_configuration'])
allfamilies=Counter(x['family'] for x in upstream['cases'])
write(folder/'full873_native_component_coverage_and_remaining.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    all873_family_case_counts=dict(allfamilies),all873_required_cases=873,all_label_views=2037,
    implemented_native_components=[dict(families=['objectives','entropic'],registered_cases=165,required_views=385,
        config=s.CONFIG,actual_native_models_complete=state['completed_cases']),
        dict(families=['tranad64'],registered_cases=15,required_views=35,config=c['separate_TranAD_native_component'],
            actual_native_models_complete=osnap['completed_cases'])],
    remaining_method_specific_native_components={k:v for k,v in allfamilies.items() if k not in ['objectives','entropic','tranad64']},
    full_source_training_and_all_native_components_complete=False,original_scope=c['original_scope'],all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 全量原始数据轨道：165项FM原生回放与385标签视图复核已实现并启动

范围为120项原始目标函数实验及45项独立的稳定数值控制，3数据集各55项，全部5个原种子保留。SMAP4种/MSL2种/SMD1种预登记标签视图合计385组。涉及11个模型配置，包含独立CFM、OT、Rectified、SB、Target、VP及SF2M评分头/flow-only消融，原始与稳定数值版本分别保留。该组件不自动重试或改写旧失败记录，也不复用其他数据集/预处理的模型回放证书。

控制器只在真实原始全109实体数据准备、原训练/评分成功守卫和完整保存分数标签评估均通过后启动新模型推理。逐个恢复完整速度网络及可选评分头，检查每个参数/缓冲张量dtype、shape和内容，并确认推理前后不变。保持window100、hidden128、20epoch的原训练配置、完整20项有限损失历史、batch64、4CPU线程、4次Monte Carlo×5个时间点及原seed+1评分随机流。全窗口、尾窗、评分调用、每个MC概率路径及forward调用均须计数完整；所有原生分数/标签须逐字节匹配原文件并完成序列化回读，禁止拟合或优化器更新。

每个概率路径坐标用独立NumPy FP32源操作程序检查，整个MC残差用FP64独立累积检查，沿用既有2e-6相对/绝对容差。理想FP64概率路径的所有不匹配坐标计数及最大误差独立保存；它们不自动成为通过标准，也不被伪称为已证明高精度公式等价。速度网络仍使用原Torch算子，未声称重写了独立神经内核。数值路径诊断与原输出数组的零容差逐字节要求分别保留。

第二个新进程比较全部原始/原生数组、原始主标签和数据位置，重算全部原指标、事件、原200次实体bootstrap及完整诊断阈值网格，再核对所有标签覆盖。所有原验证分位、种子和分组不变，禁止按TEST挑选标签版本、阈值或分数方向。单模型条件区间不替代五训练种子的配对统计；原训练梯度更新计数若未提供，不据配置补造。

40项测试全部通过，包含4组真正执行完整20epoch训练、完整4×5 MC评分及另一个新进程复核的小型神经夹具，覆盖SF2M含评分头与flow-only两类。训练/窗口/采样/精度/线程/资源缩减及不完整观测会被拒绝，且理想FP64差异不会被误标为等价。小型夹具只验证流程，绝不计为真实数据集成绩或165项完成数。

正式控制器PID、创建时间、命令及冻结配置已核验，当前只在源任务就绪后按原策略排队。继续使用原FM全440模型回放的16GiB物理、32GiB提交余量、16GiB紧急线、10/12GiB RSS/private上限及原严格CPU锁，没有借用28GiB只读通道或缩小预算。模型权重和大数组均留在F盘。

整个新全量轨道仍是873项实验及2037组标签视图。本组件加上单独15项TranAD原生组件，共覆盖180项的原生处理实现；其中TranAD首项此前已在真实54实体上完整关闭，本包没有新增真实FM成绩。其他693项的新轨道方法特定原生组件、全部源训练、五种子统计、完整TAB/VUS/Affiliation与历史论文协议一致性仍须继续。原45论文、681方法记录、273数据任务和702篇内标量指标全部保留，研究目标未完成。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=40,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    new_real_FM_benchmark_results_claimed=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full165_FM_native_and385_labels_registered.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_cases=165,required_label_views=385,variants=11,tests_passed=40,
    implemented_and_native_controller_live=True,new_real_FM_results_claimed=0,original_scope=c['original_scope'],all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full165_FM_native_component_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),registered_native_cases=165,registered_views=385,variants=11,tests=40,
    native_controller_pid=p.pid,actual_FM_native_models_complete=len(state['completed_cases']),all_original_experiments_complete=False)))
