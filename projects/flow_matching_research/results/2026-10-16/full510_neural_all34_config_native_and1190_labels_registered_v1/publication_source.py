import csv,io,json,shutil,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full510_neural_v1 as s
ROOT=s.ROOT

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
status=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
owner=psutil.Process(launch['pid']);assert owner.create_time()==launch['create_time']==status['create_time'] and owner.cmdline()==launch['command']
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG) and not status['failed_replays_preserved']
validation=ROOT/'tmp/full510_neural_native_validation_20261016_v2.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==70
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probes=[]
for executable in sorted({c['controller_python']}|{case['worker_python'] for case in cases}):
    command=[str(ROOT/executable),'-B','-X','utf8','-m',s.MODULE,'--probe-only']
    result=json.loads(subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True).stdout)
    assert result['passed'] and result['cases']==510 and result['label_views']==1190 and not result['numpy_imported'] and not result['torch_imported']
    probes.append(dict(command=command,result=result))
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full510_neural_all34_config_native_and1190_labels_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
sources={
    'original_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),
    'original_complete_state_restorer.py':ROOT/'scripts/flow_matching/neural_full510_checkpoint_state_v1.py',
    'original_comparator_score_math.py':ROOT/'scripts/flow_matching/comparator_checkpoint_score_math_v1.py',
    'original_DFM_density_math.py':ROOT/'scripts/flow_matching/dfm_checkpoint_density_math_v1.py',
    'original_GRASP_graph_path_math.py':ROOT/'scripts/flow_matching/grasp_native_checkpoint_math_v1.py',
    'original_native_tests.py':ROOT/s.TEST,
    'original_checkpoint_math_tests.py':ROOT/'tests/test_fm_neural_full510_checkpoint_math_v1.py',
    'frozen_all510_native_configuration.json':ROOT/s.CONFIG,
    'validation70.xml':validation,'exact_controller_launch.json':ROOT/c['state_root']/'launch.json'}
for name,path in sources.items():copy(path,folder/name)
rows=[];methods=Counter()
for case in cases:
    queue,job=s.scalar.resolve_case(case);model=s.read(ROOT/job['model_config']);s.validate_model(c,case,queue,job,model)
    p=model['parameters'];method=p.get('method','grasp' if model['entrypoint'].endswith(':GRASPDetector') else 'dfm');methods[method]+=1
    for label in case['required_label_policies']:
        rows.append(dict(id=case['id'],dataset=case['dataset'],seed=case['seed'],label_policy=label,method=method,
            model_configuration=job['model_config'],epochs=p['epochs'],window=case['window'],batch=p['batch_size'],
            device=job['parameter_overrides']['device'],source_CPU_threads=4,
            GRASP_source_samples=p.get('source_samples'),GRASP_flow_evaluations=p.get('flow_evaluations'),
            DFM_steps=p.get('steps'),DFM_trace=p.get('trace'),registration_is_benchmark_performance=False))
assert len(rows)==1190 and dict(methods)==dict(dt_la=165,shcl_transformer=195,kgl=60,dfm=30,grasp=60)
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all1190_registered_label_views_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
units=[]
unitroot=Path('F:/aicoding/IIA_Data/experiments/full510_neural_native_units_v2')
for path in sorted(unitroot.glob('test_full_original_epoch_capac*/fixture_metadata.json')):
    m=s.read(path);base=path.parent;native=base/'replay'/m['case']['id']/'native/native_replay.json'
    review=native.parent.parent/'review/all_label_view_independent_review.json';n,r=s.read(native),s.read(review)
    source=s.read(base/'source/result.json')
    assert n['passed'] and r['passed'] and n['all_array_dtype_shape_content_bytes_identical'] and n['all_model_tensors_identical_before_and_after']
    assert n['actual_producer_environment']['device']=='cpu' and not n['CUDA_context_initialized']
    assert r['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
    units.append(dict(model_id=m['model']['id'],original_full_parameters=m['model']['parameters'],
        actual_source_epochs=m['model']['parameters']['epochs'],source_training_seconds=source['training_seconds'],
        synthetic_fixture_only=True,actual_source_device='cpu',CUDA_replay_certified=False,benchmark_performance_claimed=False,
        source_result_path=str(base/'source/result.json'),source_result_sha256=s.sha(base/'source/result.json'),
        native_path=str(native),native_sha256=s.sha(native),independent_review_path=str(review),independent_review_sha256=s.sha(review),
        score_observation=n['all_source_forward_and_FP64_formula_observation']))
assert len(units)==34 and {x['model_id'] for x in units}=={m['id'] for m in c['source_model_configurations'].values()}
write(folder/'all34_full_budget_tiny_CPU_tests_not_CUDA_or_benchmark.json',dict(records=units,
    actual_full_original_model_capacities_and_epochs_retained=True,real_CUDA_benchmark_results_claimed=0))
write(folder/'actual_live_owner_and_cold_probes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),status_snapshot=status,
    actual_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),cold_probes=probes,
    current_actual_neural_CUDA_native_cases_complete=len(status['completed_cases']),registered_cases=510,
    original_GPU_policy=c['GPU_settings'],original_CPU_review_policy=c['settings'],
    original_phase_gates=c['gpu_prerequisite_queues'],legacy_mutex=c['legacy_gpu_queue_lock'],all_original_experiments_complete=False))
failure=ROOT/'tmp/full510_neural_native_validation_20261016_v1.xml'
fs=list(ET.parse(failure).getroot().iter('testsuite'));assert sum(int(x.get('errors','0')) for x in fs)==1
target=Path('F:/aicoding/IIA_Data/experiments/full510_neural_native_validation_preserved_v1/initial_test_collection_error.xml');copy(failure,target)
write(folder/'preserved_initial_test_collection_error.json',dict(path=str(target),sha256=s.sha(target),collection_errors=1,
    cause='Glob found34canonical executable models plus8separate MaelNet/Pi author-runner schemas;not the frozen neural source queue membership',
    correction='Select all34models directly from the authoritative frozen510case source queue;other author queues remain separate original obligations',
    source_model_or_real_experiment_modified=False,real_experiment_retried=False))
components=[]
for configuration in [
    'configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full165_objective_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full18_moment_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full60_reflow_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full105_baselines_checkpoint_replays.2026-10-16_v1.json',s.CONFIG]:
    item=s.read(ROOT/configuration);state=s.read(ROOT/item['state_root']/'status.json');p=psutil.Process(state['pid'])
    assert p.create_time()==state['create_time']
    components.append(dict(configuration=configuration,registered_native_cases=item['required_family_native_cases'],
        actual_completed_native_cases=state['completed_cases'],live_pid=p.pid,create_time=p.create_time()))
assert sum(x['registered_native_cases'] for x in components)==873
write(folder/'all873_native_implementation_components_not_completed_results.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    components=components,implemented_registered_native_case_count=873,remaining_method_specific_components_in_this873track=0,
    original_scope=c['original_scope'],known11fixed_window_incompatible_cases_still_separate=True,
    all873_actual_native_results_complete=False,all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 全510项神经方法与消融的原生回放组件已接入

覆盖完整SMAP54、MSL27、SMD28和5个种子，510项任务及1190组标签复核：DT-LA165、SHCL195、KGL60、GRASP60、DFM30。全部34份模型配置保持原容量、100/1500等epoch预算、分支、掩码、损失、评分模式、图、MC、积分步数与种子；原有作者专用MaelNet/Pi/CrossAD等队列继续保留。这些是已接入的回放任务，不代表510项已完成。

正式原生回放保持源CUDA:0设备和4数值CPU线程，恢复全部模块和缓冲区，包含GRASP实际保存的特征基和邻接、DFM正反两个场；不拟合，不重训，不用CPU结果冒充GPU原生回放。保留原参数requires_grad标志。所有源/原生分数及标签要求dtype、shape、内容字节一致且序列化回读，另一个新进程复核全部标签、指标、事件、bootstrap200与阈值网格。

比较模型保留原Python评分帧并只读核验逐步骤FP64数学：DT-LA输入/潜空间平方和及softmax，SHCL每个互补掩码、分支、Gaussian KL或重构误差，KGL下一步误差和首点零。完整理想FP64分数差异仍单独记录，不能把源精度逐步骤核验当作理想FP64端到端等价，也不声称独立重写所有神经网络内核。

DFM以独立FP64 TimeMLP解析公式核验速度及散度，观察全部Jacobian对角/Hutchinson梯度和4步原Euler状态、修正与标准高斯先验；Hutchinson全部Rademacher坐标按原生成器重现。没有宣称重演另一条全程理想FP64 ODE轨迹。原固定高斯/TimeMLP/四步Euler及双向余弦目标的论文差异继续披露，未冒称原U-Net或学习先验复现。

GRASP从完整训练窗口重算原Pearson/等边数ER图，保留实际已保存的特征基，独立核对图对角化、路径及全部5个源×10个时刻的谱残差。保留原局部窗口分数映射；不等同原作者逐实体协议或已完成原9120项GRASP任务。

70项验证通过，包含全部34份原配置的完整原epoch/容量预算小型CPU训练、真实检查点重载和独立进程指标复核。这些只验证流程与数值，不能认证CUDA执行或真实数据成绩。首轮测试收集误用了文件glob，混入8份另有作者队列的MaelNet/Pi运行配置；已改为冻结源队列的34份精确成员，原失败XML和哈希保留。

GPU回放保持原18GiB物理/32GiB提交余量/16GiB紧急线、14GiB显存准入/4GiB紧急线、原UUID、35秒让出、插补前置阶段及旧GPU队列锁和共享GPU锁。CPU复核保持16/32/16GiB门槛、10/12GiB RSS/private上限与严格CPU锁。没有降低门槛或停止科学任务。

新增完整原数据轨道的873项原生组件现已全部实现和接入；实际完成数必须看成功源守卫、原生与复核回执，不是873项均已完成。11项已发现的固定窗口不兼容仍独立保留，不能缩窗、补造数据或丢实体来冒称通过。全部873项/2037组标签及原45篇论文、681方法记录、273数据任务、702篇内标量目标继续保留；完整作者协议、全TAB范围指标和五训练种子统计尚需完成。

来源：冻结34份配置中各论文、代码、数据引用和[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=70,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    new_real_CUDA_benchmark_results_claimed=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full510_neural_all34_models_and1190_labels_registered.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_native_cases=510,registered_label_views=1190,
    tests_passed=70,newraw_native_implementation_coverage=873,remaining_method_specific_components_in_this873track=0,
    original_scope=c['original_scope'],new_real_CUDA_benchmark_results_claimed=0,all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,'tests/test_fm_neural_full510_checkpoint_math_v1.py',
    'scripts/flow_matching/neural_full510_checkpoint_state_v1.py','scripts/flow_matching/comparator_checkpoint_score_math_v1.py',
    'scripts/flow_matching/dfm_checkpoint_density_math_v1.py','scripts/flow_matching/grasp_native_checkpoint_math_v1.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full510_neural_native_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=70,registered_native_cases=510,label_views=1190,
    actual_complete=len(status['completed_cases']),live_pid=owner.pid,all_original_experiments_complete=False)))
