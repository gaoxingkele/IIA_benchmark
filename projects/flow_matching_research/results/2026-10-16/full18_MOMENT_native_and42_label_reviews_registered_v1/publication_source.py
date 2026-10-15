import csv,io,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full18_moment_v1 as s
ROOT=s.ROOT
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert s.sha(path)==s.sha(dest)
c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG) and not state['failed_replays_preserved']
validation=ROOT/'tmp/original_full18_MOMENT_replays_final_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==34
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probes=[]
for executable in [c['controller_python'],cases[0]['worker_python']]:
    command=[str(ROOT/executable),'-B','-X','utf8','-m',s.MODULE,'--probe-only']
    d=json.loads(subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True).stdout)
    assert d['passed'] and d['cases']==18 and d['label_views']==42 and not d['numpy_imported'] and not d['torch_imported']
    probes.append(dict(command=command,result=d))
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full18_MOMENT_native_and42_label_reviews_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {'original_MOMENT_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),'original_MOMENT_native_tests.py':ROOT/s.TEST,
    'frozen_all18_native_configuration.json':ROOT/s.CONFIG,'validation34.xml':validation,
    'exact_controller_launch.json':ROOT/c['state_root']/'launch.json'}.items():copy(path,folder/name)
rows=[];models={}
for case in cases:
    q,j=s.scalar.resolve_case(case);m=s.read(ROOT/j['model_config']);s.validate_model(c,case,q,j,m)
    length=s.registered_length(m);assert length==512
    models[j['model_config']]=dict(model_configuration=j['model_config'],sha256=s.sha(ROOT/j['model_config']),
        checkpoint_citation=m['checkpoint_citation'],paper_citation=m['citation'],parameters=m['parameters'],
        actual_official_checkpoint_length=length)
    for label in case['required_label_policies']:
        rows.append(dict(id=case['id'],dataset=case['dataset'],official_checkpoint_seed=42,external_context=case['window'],
            official_checkpoint_length=length,label_policy=label,model_configuration=j['model_config'],
            checkpoint_SHA256=m['parameters']['checkpoint_sha256'],padding=m['parameters']['padding_policy'],
            epochs=0,batch_size=1,CPU_threads=2,producer_python=case['worker_python'],registration_is_benchmark_performance=False))
assert len(rows)==42 and len(models)==6
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all42_registered_label_views_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'all6_checkpoint_context_configurations.json',dict(models=models,three_official_sizes=True,
    configured_checkpoint_length512_for_both_external_contexts=True,new_training_seeds=0))
write(folder/'actual_live_controller_and_cross_environment_probes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])),
    state_snapshot=state,cold_probes=probes,registered_case_counts=dict(Counter(x['dataset'] for x in cases)),
    actual_newraw_MOMENT_models_native_complete=len(state['completed_cases']),
    real_official_Small_weights_tiny_input_tests_not_benchmark_performance=True,all_original_experiments_complete=False))
failed=ROOT/'tmp/original_full18_MOMENT_replays_validation_20261016_v1.xml'
destination=Path('F:/aicoding/IIA_Data/experiments/original_full18_MOMENT_native_validation_v1/initial_fixture_queue_metadata_failure.xml')
copy(failed,destination)
fs=list(ET.parse(failed).getroot().iter('testsuite'));assert sum(int(x.get('failures','0')) for x in fs)==2
write(folder/'preserved_pre_registration_fixture_metadata_failure.json',dict(path=str(destination),sha256=s.sha(destination),
    failed_tests=2,passed_tests=32,cause='Unit fixture omitted required queue_path metadata before the new body could enter native inference',
    correction='Add the real case-schema field to the isolated fixture;all34tests pass before production registration/launch',
    changed_real_source_model_or_window_or_data=False,real_benchmark_retry_performed=False))
upstream=s.read(ROOT/c['upstream_scalar_configuration'])
allfamilies=Counter(x['family'] for x in upstream['cases'])
components=[]
for config in [
    'configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full165_objective_checkpoint_replays.2026-10-16_v1.json',s.CONFIG]:
    item=s.read(ROOT/config);status=s.read(ROOT/item['state_root']/'status.json');owner=psutil.Process(status['pid'])
    assert owner.create_time()==status['create_time']
    components.append(dict(configuration=config,required_native_cases=item['required_family_native_cases'],
        required_label_views=item['required_family_label_views'],current_complete_native_cases=status['completed_cases'],
        owner_pid=owner.pid,owner_create_time=owner.create_time()))
assert sum(x['required_native_cases'] for x in components)==198
write(folder/'all873_native_implementation_components_and_remaining.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    required_source_cases=873,required_scalar_label_views=2037,all_source_family_counts=dict(allfamilies),
    implemented_native_case_components=components,implemented_native_cases=198,
    remaining_method_specific_native_components={x:allfamilies[x] for x in ['neural','baselines','reflow']},
    remaining_method_specific_native_case_count=675,original_scope=c['original_scope'],all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 新原始数据轨道：MOMENT全部18项原生回放与42种标签视图复核已接入

覆盖官方Small/Base/Large三个固定权重、外部100/512两个窗口及完整SMAP54/MSL27/SMD28三数据集，共18项和42组标签视图。仍是seed42的固定权重零样本任务，不把重复推理当作独立训练种子。正式控制器PID、创建时间、命令、冻结配置及两个解释器的冷检查均已核验，当前完成数以真实源端成功守卫和原生两阶段回执为准，不计小型测试为真实成绩。

模型配置seq_len=512同时适用于外部100和512窗口；验证器按已冻结官方config.json哈希核验512，保留外部窗口与原padding。没有再次使用旧的“配置长度必须等于外部100”错误断言，也没有改变检查点或将窗口改成512。新原始数据案例有独立身份和输出，不是重试旧失败推理，旧12项前置元数据失败和原39项恢复队列继续保留。

原全78项MOMENT回放的native_worker代码对象和逐窗口推理、显式全观测mask、分组、padding、batch1及2CPU线程保持不变。新进程私有提供经过实际全109数据准备、源训练/评分守卫、全标签指标检查的源绑定，并增加被动权重观察。每个冻结命名参数以1MiB块完整哈希，确认原生推理前后完全不变；检查点文件/配置也重新核验，完整严格加载且无随机新头、下载、微调或参数估计。fit()在此仅加载官方固定权重，单独记为1次加载调用，参数估计和优化器更新为0，不混淆为训练。

全部原生角色、窗口、原始源点和tail必须覆盖，重新推理的每个分数/标签数组保持dtype、shape与内容逐字节一致并完整序列化回读。每次forward继续执行原2e-6相对/绝对容差的独立FP64窗口MSE检查，保留绝对误差，不据相对检查声称近零绝对误差。另一个新进程比较全部原始/原生数组，精确重算全部原指标及所有标签覆盖的事件、原200次bootstrap和完整阈值网格/PA/排名控制；源端MOMENT有效bootstrap块长规则也保留。

34项测试通过，包含真实官方35341512参数Small权重在100和512两窗口的小型输入原生推理与独立新进程复核，并覆盖全部冻结参数的字节校验。512窗口测试保留短序列右边缘复制，100窗口保留TAB额外patch；两种源码与状态都未改。小型数据不是原论文/真实全量数据集成绩。早期2项测试仅因夹具遗漏queue_path元数据而失败，原XML完整留在F盘并归档哈希，生产注册前已修正并全部通过；没有重试真实任务或改变模型数值。

保持原MOMENT回放的16GiB物理、32GiB提交余量、16GiB紧急线、10/12GiB RSS/private上限和原严格CPU锁，未借用更低门禁通道。所有大权重、数组和失败原始记录均留在F盘。

连同15项TranAD和165项FM目标函数，新增全量轨道已有198项原生处理实现；这些是已实现和接入的范围，绝非198项已完成。其他675项（510神经方法、105强基线、60Reflow）的新轨道原生组件仍需逐项完成，整个873项/2037组标签评估范围及原45论文、681方法记录、273数据任务、702篇内标量指标继续保留。官方当前权重零样本任务不替代完整Tiny/ICM预训练、作者微调、历史权重或完整TAB协议。五训练种子不确定性、历史预处理/协议对齐和VUS/Affiliation底层积分仍未由此认证。

来源：[MOMENT论文](https://arxiv.org/abs/2402.03885)、本包6项配置中的官方Small/Base/Large权重链接及冻结数据引用。整体目标继续进行。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=34,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    new_real_benchmark_results_claimed=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full18_MOMENT_native_and42_labels_registered.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_native_cases=18,registered_label_views=42,tests_passed=34,
    newraw_native_implementation_coverage=198,new_real_benchmark_results_claimed=0,original_scope=c['original_scope'],all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full18_MOMENT_native_component_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),registered_native_cases=18,label_views=42,tests_passed=34,
    actual_newraw_MOMENT_native_complete=len(state['completed_cases']),live_pid=p.pid,all_original_experiments_complete=False)))
