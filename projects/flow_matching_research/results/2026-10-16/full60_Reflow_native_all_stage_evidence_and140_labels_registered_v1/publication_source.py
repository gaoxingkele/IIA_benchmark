import csv,io,json,os,shutil,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full60_reflow_v1 as s
ROOT=s.ROOT

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert launch['configuration_sha256']==s.sha(ROOT/s.CONFIG) and not state['failed_replays_preserved']
validation=ROOT/'tmp/full60_reflow_native_final_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==56
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
probes=[]
for executable in sorted({c['controller_python']}|{case['worker_python'] for case in cases}):
    command=[str(ROOT/executable),'-B','-X','utf8','-m',s.MODULE,'--probe-only']
    result=json.loads(subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True).stdout)
    assert result['passed'] and result['cases']==60 and result['label_views']==140 and not result['numpy_imported'] and not result['torch_imported']
    probes.append(dict(command=command,result=result))
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full60_Reflow_native_all_stage_evidence_and140_labels_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
sources={
    'original_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),
    'original_stage_evidence_verifier.py':ROOT/'scripts/flow_matching/reflow_checkpoint_evidence_v1.py',
    'original_native_tests.py':ROOT/s.TEST,
    'original_stage_evidence_tests.py':ROOT/'tests/test_fm_reflow_checkpoint_evidence_v1.py',
    'frozen_all60_native_configuration.json':ROOT/s.CONFIG,
    'validation56.xml':validation,'exact_controller_launch.json':ROOT/c['state_root']/'launch.json'}
for name,path in sources.items():copy(path,folder/name)
rows=[];models={}
for case in cases:
    queue,job=s.scalar.resolve_case(case);model=s.read(ROOT/job['model_config']);s.validate_model(c,case,queue,job,model)
    models[job['model_config']]=dict(configuration=model,sha256=s.sha(ROOT/job['model_config']))
    for label in case['required_label_policies']:
        rows.append(dict(id=case['id'],dataset=case['dataset'],seed=case['seed'],label_policy=label,
            model_configuration=job['model_config'],total_training_epochs=job['total_training_epochs'],
            additional_teacher_pair_generation_compute=job['additional_pair_generation_compute'],
            stages=model['parameters'].get('reflow_stages',1),batch64=True,hidden128=True,window100=True,
            CPU_threads=2,all_source_pair_Heun_steps=50 if job['additional_pair_generation_compute'] else 0,
            registration_is_benchmark_performance=False))
assert len(models)==4 and len(rows)==140
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all140_registered_label_views_not_performance.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
write(folder/'all4_source_model_configurations.json',models)
write(folder/'actual_live_owner_and_cold_probes.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),state_snapshot=state,
    actual_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),cold_probes=probes,
    current_actual_complete_cases=len(state['completed_cases']),registered_cases=60,registered_label_views=140,
    new_unit_test_results_are_benchmark_performance=False,all_original_experiments_complete=False))
failure=ROOT/'tmp/full60_reflow_native_validation_20261016_v1.xml'
failed_suites=list(ET.parse(failure).getroot().iter('testsuite'))
assert sum(int(x.get('failures','0')) for x in failed_suites)==1
target=Path('F:/aicoding/IIA_Data/experiments/full60_reflow_native_validation_preserved_v1/noncanonical_JSON_fixture_initial_failure.xml')
copy(failure,target)
write(folder/'preserved_initial_metadata_fixture_failure.json',dict(path=str(target),sha256=s.sha(target),failed=1,passed=55,
    cause='json.dumps on decoded sorted-key metadata reproduced the original canonical bytes;the test did not actually corrupt anything',
    correction='Use indented JSON in this isolated corruption fixture;the existing strict canonical-byte verifier correctly rejects it',
    production_model_or_data_or_numerical_budget_changed=False,real_experiment_retried=False))
components=[]
for configuration in [
    'configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full165_objective_checkpoint_replays.2026-10-16_v1.json',
    'configs/reproducibility/fm_original_full18_moment_checkpoint_replays.2026-10-16_v1.json',s.CONFIG]:
    item=s.read(ROOT/configuration);status=s.read(ROOT/item['state_root']/'status.json');owner=psutil.Process(status['pid'])
    assert owner.create_time()==status['create_time']
    components.append(dict(configuration=configuration,registered_native_cases=item['required_family_native_cases'],
        actual_completed_native_cases=status['completed_cases'],live_pid=owner.pid,create_time=owner.create_time()))
assert sum(x['registered_native_cases'] for x in components)==258
write(folder/'full873_native_implementation_coverage_not_completed_performance.json',dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),components=components,implemented_registered_native_cases=258,
    remaining_method_specific_native_components=dict(neural=510,baselines=105),remaining_native_case_count=615,
    original_scope=c['original_scope'],all_original_experiments_complete=False))
(folder/'README.md').write_text('''# Reflow 全部阶段证据与完整数据回放已接入

新增完整 SMAP54、MSL27、SMD28 数据上的 60 项原生回放及 140 组标签复核；覆盖 2/3 阶段 Reflow 和独立训练 40/60 epoch 对照，全部 5 个种子。正式完成数见本目录的真实进程快照，注册数与单元测试均不算 benchmark 成绩。其他原论文、原有队列及完整 TAB 对比继续保留。

严格恢复最终网络、每阶段教师权重、全部有序起点与终点、规范 JSON 阶段记录；重新生成所有起点并以原 50 步 Heun 完整重积分全部训练端点对，核对逐字节结果。独立状态程序检查每次教师调用输入、预测与校正递推、离散 action 和位移；教师神经网络沿用原 Torch 内核，并未独立重写神经网络或重演优化器轨迹。逐阶段 loss 与全部 40/60 epoch 历史保持一致，不能仅凭记录中的 warm start 声明认证完整训练轨迹。

随后对所有训练、验证、测试窗口与尾部执行原 MC4×5 时刻残差评分；分数和标签的 dtype、shape、全部内容字节及序列化回读必须一致。独立 FP32 路径程序及 FP64 残差计算保留原 2e-6 容差，理想 FP64 路径误差仍单列诊断。第二个新进程重算所有标签下的原指标、事件、200 次 bootstrap、完整阈值网格与 PA/排名对照。

56 项测试通过，包含四种完整原 epoch/容量/ODE 预算的小型数据训练、真实多模块检查点恢复、完整端点积分和独立进程复核；这些是数值与流程测试。首轮唯一失败是破坏性测试夹具未实际改变 JSON 字节，已修正为缩进 JSON；生产校验代码无需放宽，原失败 XML 与哈希已保留。

保持 2 CPU 线程、100 窗口、128 隐层、64 batch、所有种子及原随机数约定。继续使用 16 GiB 物理、32 GiB 提交余量、16 GiB 紧急线、10/12 GiB RSS/private 上限与原严格 CPU 锁。Reflow 额外生成端点对，因此 epoch 匹配不等于计算量匹配。没有重启科学任务、重试失败任务或修改冻结原配置。

新增完整数据轨道的原生实现覆盖由 198 增至 258 项，另 615 项（神经方法 510、强基线 105）仍需相应原生组件；这不表示 258 项均已完成。全部 873 项/2037 组标签及原 45 篇论文、681 项方法记录、273 项数据任务和 702 项篇内标量目标继续保留。原图像实验、一步蒸馏、完整 TAB、训练种子不确定性及作者历史协议一致性均未由本次测试认证。

来源：[Rectified Flow 论文](https://arxiv.org/abs/2209.03003)、冻结作者仓库快照 `gnobitab/RectifiedFlow`、[TAB](https://github.com/decisionintelligence/TAB)，以及冻结配置中的原始数据引用。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=56,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    new_real_benchmark_results_claimed=0,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full60_Reflow_all_stage_evidence_and140_labels_registered.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_native_cases=60,registered_label_views=140,
    tests_passed=56,newraw_native_implementation_coverage=258,remaining_native_implementation_cases=615,
    original_scope=c['original_scope'],new_real_benchmark_results_claimed=0,all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py','scripts/flow_matching/reflow_checkpoint_evidence_v1.py',s.TEST,
    'tests/test_fm_reflow_checkpoint_evidence_v1.py',s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full60_Reflow_native_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=56,registered_native_cases=60,label_views=140,
    actual_complete=len(state['completed_cases']),live_pid=p.pid,all_original_experiments_complete=False)))
