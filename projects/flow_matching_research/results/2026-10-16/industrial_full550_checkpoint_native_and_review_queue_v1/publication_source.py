"""Publish the guarded full industrial checkpoint queue, without result credit."""
from collections import Counter
import csv
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import replay_industrial_full550_checkpoints_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);q,manifest,jobs,executions=s.verify(c)
base=ROOT/c['state_root'];state=read(base/'status.json');launch=read(base/'launch.json')
p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']==launch['create_time']
assert p.cmdline()==launch['command'] and launch['configuration_sha256']==sha(ROOT/s.CONFIG)
assert state['state']=='waiting_for_memory' and state['active_pid'] is None
assert state['verified_ids']==state['native_only_ids']==[] and not state['failed_preserved']
assert not [x for x in p.children(recursive=True) if x.name().lower()=='python.exe']
assert state['commit_available_bytes']<c['settings']['minimum_commit_headroom_bytes']==32*2**30
assert not state['shared_lock_owned']
assert not Path(c['artifact_root']).exists()
prior=ROOT/s.PRIOR;bindings=read(prior/'all466_proof_source_and_guard_bindings.json')['records']
assert len(bindings)==466
rows=[];models={};counts=Counter()
for cid,original in jobs.items():
    job=executions[cid];configuration=read(ROOT/job['model_config']);family=s.ENTRIES[configuration['entrypoint']]
    models.setdefault(original['model_config'],dict(configuration=configuration,configuration_sha256=sha(ROOT/original['model_config'])))
    source_path=ROOT/job['output_directory']/'result.json';available=cid in bindings
    assert source_path.is_file()==available
    if available:
        binding=bindings[cid]
        assert binding['actual_execution_id']==job['id'] and sha(source_path)==binding['source_result_sha256']
    counts[family,available]+=1
    rows.append(dict(canonical_id=cid,actual_execution_id=job['id'],dataset=original['dataset'],seed=original['seed'],
        model_config=original['model_config'],family=family,original_window=original['window'],original_cpu_threads=2,
        original_training_epochs=configuration['parameters']['epochs'],source_complete_and_point_audited=available,
        exact_recovery_mapping=cid!=job['id'],native_replay_complete=False,separate_independent_review_complete=False,
        status='waiting_unchanged32GiB_admission' if available else 'waiting_full_original_source_and_independent_point_audit'))
assert len(rows)==550 and len(models)==22 and sum(r['source_complete_and_point_audited'] for r in rows)==466
assert Counter(r['original_window'] for r in rows)=={100:500,60:25,10:25}
assert sum(r['exact_recovery_mapping'] for r in rows)==3
first=s.source_binding(c,jobs,executions,state['active_case']);assert first is not None
tests=ROOT/'tmp/industrial_full550_native_replays_shared_validation_20261016_v3.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==106
assert all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
first_tests=ROOT/'tmp/industrial_full550_native_replays_validation_20261016_v1.xml'
old=list(ET.parse(first_tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in old)==32 and sum(int(v.get('failures',0)) for v in old)==5
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_full550_checkpoint_native_and_review_queue_v1'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
with (out/'all550_original_checkpoint_replay_and_review_matrix.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name,path in dict(frozen_registration=ROOT/s.CONFIG,actual_launch=base/'launch.json',actual_waiting_state=base/'status.json',
    fresh106_validation=tests,preserved_initial_fixture_validation_failures=first_tests,
    corrected32_unit_fixture_validation=ROOT/'tmp/industrial_full550_native_replays_validation_20261016_v2.xml').items():
    shutil.copyfile(path,out/(name+path.suffix))
write(out/'all22_original_model_configs_parameters_and_citations.json',dict(models=models,
    original_industrial_source_and_data_citations=read(ROOT/s.point.CONFIG)['citations']))
write(out/'first_actual_source_ready_binding.json',first[-1])
write(out/'source_case_counts_by_family.json',dict(counts=[dict(family=family,source_complete=available,cases=n) for (family,available),n in sorted(counts.items())]))
handles=[]
for pid in [16360,30032,31444,33876,10880,36108,41992,18292,27336,13636,39444,30800]:
    try:
        process=psutil.Process(pid);handles.append(dict(pid=pid,create_time=process.create_time(),command=process.cmdline(),
            user_seconds=process.cpu_times().user,actual_python_descendants=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline())
                for x in process.children(recursive=True) if x.name().lower()=='python.exe']))
    except psutil.NoSuchProcess:handles.append(dict(pid=pid,handle_missing_at_capture=True))
write(out/'actual_other_training_and_review_handles_preserved.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles,
    processes_stopped_or_modified_by_this_iteration=False))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    original_CPU_conditions=550,original_GPU_conditions=850,original_model_configs=22,source_complete_cases=466,missing_original_sources=84,
    actual_controller_live=True,numerical_children_started=0,full_native_cases_completed=0,full_independent_reviews_completed=0,
    new_optimizer_updates=0,new_training_seed_slots=0,new_complete_model_training_results=0,
    first_original_case_waiting=state['active_case'],original_CPU_threads=2,unchanged_admission_commit_bytes=32*2**30,
    unchanged_serial_resource_lock=c['resource_lock'],original_window_counts={str(k):v for k,v in Counter(r['original_window'] for r in rows).items()},
    seven_original_legacy_presets_source_checkpoint_scores_passed_unit_roundtrips=True,
    objective_auxiliary_head_and_Reflow_evidence_roundtrips_passed=True,
    source_train_validation_test_metadata_and_tails_roundtrips_passed=True,
    unit_tests_are_not_benchmark_performance=True,independent_neural_kernels=False,
    independent_historical_training_updates=False,source_historical_resource_gate_certified=False,
    independent_bootstrap=False,independent_VUS=False,independent_affiliation=False,
    full_TAB_harness_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False))
(out/'README.md').write_text('''# 工业CPU全部550条件：完整检查点重放与独立复核队列

范围为原550个CPU条件、22个模型配置、五个工业数据任务和五个种子，不缩小为当前已有的466项结果。GPU850项和全45篇/681方法记录/273数据任务/702论文指标范围保持独立。全部550行原任务身份、配置、窗口、源完成状态及两阶段复核状态见 `all550_original_checkpoint_replay_and_review_matrix.csv`。

队列已实际启动，捕获时PID30276仍存活，首项为FM independent／TEP／seed1103。源训练和独立评分证明已就绪，但提交余量约25.5GiB低于原32GiB门槛，当前没有数值子进程。完整重放完成0项，独立复核完成0项；队列启动不计作实验成绩。84个原源条件尚缺完整结果及独立评分证明；数值修正后的不同配置、不同输出不自动替代原条件。

每个条件顺序执行两阶段，各阶段重新获得既有串行CPU锁并通过原16GiB物理/32GiB提交/16GiB紧急门槛和10/12GiB RSS/private上限：

1. 恢复全部原权重和缓冲区，不调用fit。用原2个CPU线程、原精度、原评分函数和原批划分重推全部独立训练段、验证段和测试实体。逐字节核对全部评分、标签、时间戳、源行和故障标签，覆盖末尾窗口。FM核验每个MC路径和前向的评分公式；Reflow恢复所有教师/端点证据并重放全部原ODE步；七个基线预设独立核验各自评分公式。独立程序复用原神经网络算子，不称独立神经网络实现。
2. 新进程重新读取两份完整NPZ，严格比较每项数组的dtype、shape及全部内容字节，并独立复算原0.1%、1%、5%验证阈值的点级、实体与事件指标。校准仍只使用验证分数，不使用测试标签选阈值。保持既有2e-6数学残差容差，并保留理想FP64的所有残差失败诊断；评分数组仍要求逐字节一致。失败、部分输出及守卫回执保留，不自动重试，也不回写原数据或检查点。

窗口逐模型沿用配置：500条件为100，DCdetector的25条件为60，TranAD的25条件为10。完整角色/尾窗/标量窗口分数和逐点窗口分数均单独验证。106项相关测试通过，包括七个原基线配置、SF2M辅助头、两阶段Reflow证据恢复和完整元数据/尾窗破坏拒绝。测试中的小型训练只验证程序和恢复契约，不构成真实数据集指标。首轮五项单元夹具设置失败已保留；修正了夹具错误参数/窗口和过晚捕获预期异常，并让新队列按各模型原窗口核验，未改变原模型、数据、训练预算或评分实现。

此层验证不能追溯补出原训练未保存的优化器更新轨迹，不回溯认证历史资源门槛，也不独立重算bootstrap、VUS或Affiliation积分；这些限制单列。各模型/数据引用和冻结参数见 `all22_original_model_configs_parameters_and_citations.json`。既有工业点级和区间指标证据、主指标表v69保持原快照。整体任务尚未完成，其他正在训练的数值进程未修改或停止。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial550_full_checkpoint_replay_queue.2026-10-16_v1.json'
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),frozen_configuration=s.CONFIG,required_CPU_cases=550,source_completed_cases=466,
    native_cases_completed=0,independent_review_cases_completed=0,controller_live=True,actual_state='waiting_for_memory',
    unchanged32GiB_admission=True,original_global_scope=c['original_global_scope'],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8');assert out.name not in text
text+='\n`工业CPU全550条件接入完整检查点重放及独立复核 <results/2026-10-16/'+out.name+'/README.md>`_\n'
text+='22个原配置、原2线程和各模型100/60/10窗口保持；七基线、SF2M辅助头、Reflow教师证据及完整角色/元数据/尾窗的106项相关验证通过。真实控制器已启动，当前按原32GiB提交门槛等待；新增完整重放及模型训练成绩均0。源已有466项、仍缺84项，全部原CPU550/GPU850及全局研究范围不变。\n'
page.write_text(text,encoding='utf-8',newline='\n')
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths.extend(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_full550_native_queue_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(packet=str(out),files=len(paths),required_cases=550,native_cases_completed=0,tests=106))
