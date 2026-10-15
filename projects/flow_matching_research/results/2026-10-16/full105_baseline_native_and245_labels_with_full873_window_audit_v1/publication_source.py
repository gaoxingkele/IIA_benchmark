import csv,io,json,shutil,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full105_baselines_v1 as s
ROOT=s.ROOT

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
owner=psutil.Process(launch['pid']);assert owner.create_time()==launch['create_time']==state['create_time'] and owner.cmdline()==launch['command']
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG) and not state['failed_replays_preserved']
validation=ROOT/'tmp/full105_baseline_native_final_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==67
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probes=[]
for executable in sorted({c['controller_python']}|{case['worker_python'] for case in cases}):
    command=[str(ROOT/executable),'-B','-X','utf8','-m',s.MODULE,'--probe-only']
    result=json.loads(subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True).stdout)
    assert result['passed'] and result['cases']==105 and result['label_views']==245 and not result['numpy_imported'] and not result['torch_imported']
    probes.append(dict(command=command,result=result))
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full105_baseline_native_and245_labels_with_full873_window_audit_v1'
folder.mkdir(parents=True,exist_ok=False)
sources={
    'original_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),
    'original_checkpoint_math.py':ROOT/'scripts/flow_matching/legacy_baseline_checkpoint_math_v1.py',
    'original_native_tests.py':ROOT/s.TEST,
    'original_checkpoint_math_tests.py':ROOT/'tests/test_fm_legacy_baseline_checkpoint_math_v1.py',
    'original_window_audit.py':ROOT/'scripts/flow_matching/audit_full873_window_geometry_v1.py',
    'original_window_audit_tests.py':ROOT/'tests/test_fm_full873_window_geometry_v1.py',
    'frozen_all105_native_configuration.json':ROOT/s.CONFIG,
    'validation67.xml':validation,'exact_controller_launch.json':ROOT/c['state_root']/'launch.json',
    'all873_real_array_header_window_audit.json':Path('F:/aicoding/IIA_Data/experiments/original_full873_window_geometry_audit_20261016_v1.json'),
    'tiny_fixture_requires_grad_flag_numerical_probe.json':Path('F:/aicoding/IIA_Data/experiments/full105_baseline_native_validation_preserved_v1/requires_grad_flag_numerical_probe.json')}
for name,path in sources.items():copy(path,folder/name)
rows=[]
for case in cases:
    queue,job=s.scalar.resolve_case(case);model=s.read(ROOT/job['model_config']);s.validate_model(c,case,queue,job,model);p=model['parameters']
    for label in case['required_label_policies']:
        rows.append(dict(id=case['id'],dataset=case['dataset'],seed=case['seed'],label_policy=label,
            model_configuration=job['model_config'],method=p['method'],epochs=p['epochs'],window=case['window'],
            training_batch=p['parameters']['batch_size'],source_score_batch=512 if p['method']=='usad' else p['parameters']['batch_size'],
            USAD_adversarial=p['parameters'].get('adversarial'),CPU_threads=2,registration_is_benchmark_performance=False))
assert len(rows)==245
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all245_registered_label_views_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'actual_live_owner_and_cold_probes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),state_snapshot=state,
    actual_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),cold_probes=probes,
    current_actual_complete_cases=len(state['completed_cases']),registered_cases=105,registered_label_views=245,
    tiny_fixture_tests_are_benchmark_performance=False,all_original_experiments_complete=False))
failures=[]
causes={1:'Registration receipt used plural baselines instead of the actual baseline test filename;no production launch occurred',
    2:'New verifier incorrectly averaged AT layers,treated idealFP64attention as sourceFP32equivalence,and disabled original requires_grad flags causing iTransformer byte differences',
    3:'New analytic layer-reduction test used saturated softmax values and its difference assertion was ineffective;test temperature changed to1,source model temperatures remain50'}
for version,expected in [(1,30),(2,5),(3,1)]:
    source=ROOT/f'tmp/full105_baseline_native_validation_20261016_v{version}.xml'
    fs=list(ET.parse(source).getroot().iter('testsuite'));count=sum(int(x.get('failures','0')) for x in fs);assert count==expected
    target=Path(f'F:/aicoding/IIA_Data/experiments/full105_baseline_native_validation_preserved_v1/pre_registration_validation_v{version}.xml');copy(source,target)
    failures.append(dict(path=str(target),sha256=s.sha(target),failures=count,cause=causes[version]))
write(folder/'preserved_pre_registration_failures_and_corrections.json',dict(attempts=failures,
    unchanged_native_byte_equality_gate=True,unchanged_2e_6_primitive_tolerance=True,
    ideal_FP64_end_to_end_mismatch_counts_and_errors_preserved=True,
    production_source_model_or_data_or_budget_modified=False,real_experiment_retried=False))
components=[]
for configuration in [
    'configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full165_objective_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full18_moment_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full60_reflow_checkpoint_replays.2026-10-16_v1.json',s.CONFIG]:
    item=s.read(ROOT/configuration);status=s.read(ROOT/item['state_root']/'status.json');p=psutil.Process(status['pid'])
    assert p.create_time()==status['create_time']
    components.append(dict(configuration=configuration,registered_native_cases=item['required_family_native_cases'],
        actual_completed_native_cases=status['completed_cases'],live_pid=p.pid,create_time=p.create_time()))
assert sum(x['registered_native_cases'] for x in components)==363
write(folder/'full873_native_implementation_coverage_not_completed_performance.json',dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),components=components,implemented_registered_native_cases=363,
    remaining_method_specific_native_components=dict(neural=510),remaining_native_case_count=510,
    original_scope=c['original_scope'],all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 强基线完整数据回放与全 873 项窗口审计

105 项强基线原生回放及 245 组标签复核已接入，覆盖完整 SMAP54、MSL27、SMD28 和 5 个种子。包含 AT、DCdetector、iTransformer、TimesNet、FP32 TranAD、USAD 非对抗和原符号损失两个版本。全部 21 份原数据集配置随冻结配置归档，保留原窗口、容量、epoch、patch、损失符号和 batch；这些是已注册的局部复现配置，不能宣称与论文原配置或 SOTA 成绩等价。实际完成数见进程与回执快照，任务注册和小型测试不计作 benchmark 成绩。

恢复所有检查点参数和缓冲区，包括 USAD 的编码器和两个解码器，禁止拟合；源模型 requires_grad 标志也保持，评分仍使用原 no_grad。实际小型检查点诊断证实：改变该标志可令 iTransformer 分数产生约 9.5e-7 差异，保留后恢复逐字节一致。未修改源模型、权重、原训练预算或数值门禁。

保持原始前向与评分，逐批核验完整输入、输出和所有窗口/尾部。逐头 KL、先验归一化及 softmax 用独立 NumPy FP64 核验；温度和层归约另用独立源 FP32 程序检查。AT 原代码累加层，DC 原代码平均层。重构、USAD 加权误差、TranAD 末点误差继续按 2e-6 原容差核验。完整理想 FP64 注意力评分的失败坐标数和最大误差单列保留，不能把逐步骤核验通过等同于全程理想 FP64 等价。神经网络沿用源内核，未声称独立神经网络实现或完整优化器轨迹复演。

全部源端/原生分数和标签必须保留 dtype、shape、全部内容字节与序列化回读；另一个新进程复核所有标签的指标、事件、200 次 bootstrap、完整阈值网格和 PA/排名控制。TranAD/USAD 原窗口分数广播机制保持，未冒称作者逐时间戳或 POT 协议。

67 项验证通过，含 11 份代表性配置的完整原 epoch/容量/patch 预算小型训练、真实检查点恢复及独立进程复核；包含三种 DC patch 配置、三种 TimesNet 容量和两个 USAD 损失版本。小型数据只用于流程和数值验证。全部注册前失败 XML 和修正原因保留；没有重试真实科学任务。

全 873 项窗口审计读取了全部 436 个实际 NPZ 数组头，并核对 109 个实体的训练、验证、测试长度与分组区间。11 项存在固定窗口不兼容：SMAP 的 5 个 DC105 任务验证长度只有 100；MOMENT 三个尺寸在 SMAP/MSL 的 6 个外部 512 任务含短训练/验证段。前者可能在完成训练后才因评分拒绝，后者训练前即不满足窗口门禁。报告是窗口门禁预测，不冒称已观察到训练终止失败；没有缩窗、补造样本、丢实体、取消或跳过源任务，也未把数组头审计冒称完整数据值/校验和审计。实际数据完整性与源守卫仍分别要求。

继续使用 2 CPU 线程、16 GiB 物理、32 GiB 提交余量、16 GiB 紧急线、10/12 GiB RSS/private 上限和原严格 CPU 锁。新增完整数据轨道原生组件覆盖由 258 增至 363 项，另 510 项神经方法仍需原生组件；这不表示 363 项均已完成。全部 873 项/2037 组标签及原 45 篇论文、681 项方法记录、273 项数据任务和 702 项篇内标量目标继续保留。完整 TAB、作者历史协议、配对训练种子不确定性及其他原实验仍未完成。

来源：冻结配置中各方法的论文、代码和数据引用；[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=67,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    new_real_benchmark_results_claimed=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full105_baselines_native_and245_labels_window873_audit.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_native_cases=105,registered_label_views=245,
    tests_passed=67,newraw_native_implementation_coverage=363,remaining_native_implementation_cases=510,
    predicted_window_incompatible_cases=11,original_scope=c['original_scope'],new_real_benchmark_results_claimed=0,
    all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py','scripts/flow_matching/legacy_baseline_checkpoint_math_v1.py',s.TEST,
    'tests/test_fm_legacy_baseline_checkpoint_math_v1.py','scripts/flow_matching/audit_full873_window_geometry_v1.py',
    'tests/test_fm_full873_window_geometry_v1.py',s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full105_baseline_native_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=67,registered_native_cases=105,label_views=245,
    actual_complete=len(state['completed_cases']),live_pid=owner.pid,all_original_experiments_complete=False)))
