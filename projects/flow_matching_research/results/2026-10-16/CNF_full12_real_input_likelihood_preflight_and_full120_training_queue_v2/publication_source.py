from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_source_cnf_full120_v2 as s
from scripts.flow_matching import audit_source_cnf_full120_v2 as audit
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

q=read(ROOT/s.QUEUE);s.verify(q)
c=read(ROOT/audit.CONFIG);audit.verify(c)
binding=s.prerequisite(q);assert binding is not None
report=read(binding['report_path']);assert len(report['records'])==12
data={case['dataset']:case['data'] for case in read(ROOT/s.preflight.CONFIG)['cases']}
assert len(data)==5
for item in data.values():assert sha(ROOT/item['path'])==item['sha256']
for record in report['records']:
    key=record['id'].split('__leftout')[0]
    assert record['full_data_cells']==data[key]['cells'] and record['full_timepoint_counts']==data[key]['days']
    assert record['parameters']==9093 and record['dimensions']==5 and record['training_batch']==1024
    assert record['requested_leftout_gradient_zero'] and record['all_retained_timepoint_gradients_nonzero_finite']
    assert record['optimizer_updates']==0 and not record['benchmark_performance_claim']
tests=Path('F:/aicoding/IIA_Data/experiments/CNF_full120_training_and_replay_20261016_v3_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==27
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CNF_full12_real_input_likelihood_preflight_and_full120_training_queue_v2'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
shutil.copyfile(binding['report_path'],packet/'actual_full12_native_likelihood_preflight.json')
shutil.copyfile(binding['guard_path'],packet/'actual_native_preflight_process_guard.json')
pref=read(ROOT/s.preflight.CONFIG)
for label,path in dict(preflight_status=ROOT/pref['state_root']/'status.json',
    complete_progress=ROOT/pref['state_root']/'progress.json').items():shutil.copyfile(path,packet/(label+'.json'))
for name in ['worker.stdout.log','worker.stderr.log']:
    shutil.copyfile(ROOT/pref['state_root']/name,packet/('actual_preflight_'+name))
shutil.copyfile(tests,packet/'validation_27_checks.xml')
shutil.copyfile(__file__,packet/'publication_source.py')
for name in ['launch_source_CNF_full120_20261016_v1.py','handoff_source_CNF_metadata_v2_20261016.py','handoff_source_CNF_metadata_v2b_20261016.py']:
    shutil.copyfile(ROOT/'tmp'/name,packet/name)
launch=Path('F:/aicoding/IIA_Data/experiments/source_CNF_full120_v2b_launch')
for path in launch.iterdir():
    if path.is_file():shutil.copyfile(path,packet/('actual_v2_'+path.name))
retired=read(launch/'verified_metadata_only_predecessors.json')
assert not retired['numerical_workers_stopped'] and len(retired['records'])==2
for row in retired['records']:
    assert row['numerical_descendants']==[] and row['shared_numerical_lock_unowned']
    try:p=psutil.Process(row['owner']['pid'])
    except psutil.NoSuchProcess:pass
    else:assert p.create_time()!=row['owner']['create_time']
write(packet/'preserved_metadata_review_findings.json',dict(
    initial_v1_validation_inclusion_flag_incorrect=True,
    actual_CNF_validation_excludes_requested_leftout=True,
    v2_changes_numerical_recipe=False,v1_original_registration_and_files_preserved=True,
    initial_handoff_safely_rejected_because_native_conhost_descendants=True,
    successful_handoff_accepts_only_exact_native_conhost_image_command_identity=True,
    numerical_workers_stopped=False,actual_training_attempts_retried=0))
rows=[]
for r in report['records']:
    rows.append(dict(id=r['id'],full_data_cells=r['full_data_cells'],dimensions=r['dimensions'],
        parameters=r['parameters'],training_batch=r['training_batch'],validation_batch=r['validation_batch'],
        requested_leftout=r['requested_leftout'],training_timepoint_indices=json.dumps(r['actual_training_timepoint_indices']),
        initial_batch_objective=r['objective'],initial_validation_objective=r['validation_objective'],
        optimizer_updates=0,benchmark_performance_claim=False))
with (packet/'all12_real_input_likelihood_and_gradient_checks_NOT_benchmark_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write(packet/'all5_complete_input_data_hashes.json',dict(records=data,untouched_full_raw_inputs_rehashed=True))
handles=[]
for label,config in [('source',q),('replay',c)]:
    state_path=ROOT/config['state_root']/'status.json';state=read(state_path)
    assert not state.get('failed_preserved')
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(label=label,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private))
    shutil.copyfile(state_path,packet/('actual_v2_'+label+'_state.json'))
for pid,ct in [(19444,1792131765.9271505),(40468,1792135612.9762442),
    (36912,1792097263.1796665),(41992,1791948094.8603492)]:
    p=psutil.Process(pid)
    # The exact latest CrossAD creation time comes from the live process tree;
    # do not impose an approximated timestamp on a numerical handle.
    if pid!=40468:assert p.create_time()==ct
    assert p.is_running()
    handles.append(dict(label='unmodified_original_numerical_worker',pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])))
write(packet/'actual_live_original_workers_and_new_controllers.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],actual_real_input_preflight_cases_passed=12,
    required_real_input_preflight_cases=12,new_full_CNF_models_completed=0,required_full_CNF_conditions=120,
    required_five_seed_cohorts=24,full_source_and_checkpoint_replay_implemented=True,
    native_framework_lifecycle_fixture_not_yet_complete=True,full_training_waiting_original_32GiB_commit_gate=True,
    original_training_resource_gates_lowered=False,original_numeric_workers_stopped=False,
    historical_paper_equivalence_certified=False,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# CNF 全12条件真实输入预检与全120条件训练/重放队列

本轮真正完成的是12/12个原始完整数据输入条件的似然与梯度预检，进程退出0且无资源中止；尚无新的完整CNF训练模型或论文指标。
覆盖5个数据轨道：公开donor13176、发布donor32606、发布EB、发布CITE、代码引用EB100；全部原始输入再次核对SHA256，未覆盖或改写原始数据。
全部12个内部留出分支保留原定1024训练批量、5个PC、9093参数、精确迹、dopri5及1e-5容差。
每个条件完整累加全部保留时间点的似然梯度，实际检查请求留出点梯度为零、其余时间点梯度有限且非零。此前v1形状失败和v2内存中止保留，成功的是独立v3预检。
CSV中的初始化目标仅为连通性检查，不能当作训练后性能、Wasserstein结果或论文重现结果。

完整训练和独立检查点重放已实现并启动调度：12分支×2套源码参数×5训练种子=120条件、24个完整五种子组。
两套参数是模型默认AdamW weight_decay=0.01和实验YAML覆盖1e-5；学习率0.001、1024批量、最大1000轮、每10轮验证、原默认100次验证耐心，分别冻结。
缺失构造绑定明确采用dopri5/autograd/atol=rtol1e-5，不冒充历史作者实验组合，也不替代独立TrajectoryNet和Reg.CNF基线。
原始加法目标逐个端点反传，全部端点反传完成后才进行一次AdamW更新；不降低批量、不省略端点、不缩短训练。
似然的输入自动微分要求no_grad评估上下文，显式关闭Lightning inference_mode。精确似然公式不变。
源码CNF在训练和验证似然中都排除修正后的留出索引，完整测试分布仍保留。最初v1沿用了CFM的验证包含标志，复核后在隔离v2纠正元数据，未改动已冻结v1文件。
两条v1控制器仅等待内存，零模型结果、零数值后代、共享数值锁未持有，经PID/创建时间/完整命令和原生conhost身份逐项确认后交接；没有停止训练、评估或重放数值进程。

完整训练前还要求锁定原环境中的Lightning框架集成验证，测试合成1024批量、两套衰减和四/五时间点的每批一次更新、检查点计数/优化器矩及评估自动微分。该框架验证尚在资源准入等待，不能记作完整训练结果。
后续完整重放恢复验证选择权重及全部AdamW矩，重跑原完整测试分布，所有标量要求2e-6一致，并使用SciPy完整双精度距离矩阵/POT最优LP独立计算W1；不抽样测试细胞。
两条新控制器均已验证存活；源调度等待原32GiB提交内存余量门槛，原16GiB紧急线及6/8GiB进程RSS/private上限保持。未降低资源守卫。

27项相关检查通过，包括真实预检报告绑定、全120条件/种子/参数范围、完整加法目标与单次AdamW更新等价、只修正优化器断言的重放AST，以及真实源码验证留出行为。
全部45篇论文、681条方法/基线/消融、273条原始数据任务、702条标量目标及原论文/TAB/新数据轨道仍在范围内，整体实验尚未完成。
''',encoding='utf-8')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CNF_full12_preflight_full120_training.2026-10-16_v2.json'
assert not ara.exists()
write(ara,dict(source_configuration=s.QUEUE,replay_configuration=audit.CONFIG,result_packet=packet.relative_to(ROOT).as_posix(),
    real_input_native_preflight_completed=12,required_full_CNF_models=120,full_CNF_models_completed=0,
    original_scope=q['original_scope'],all_original_experiments_complete=False))
paths=[s.QUEUE,s.MODEL,s.MODULE.replace('.','/')+'.py',s.TEST,audit.CONFIG,audit.MODULE.replace('.','/')+'.py',
    s.previous.QUEUE,s.previous.MODEL,s.previous.MODULE.replace('.','/')+'.py',s.previous.TEST,
    'configs/reproducibility/fm_source_cnf_full120_checkpoint_replay.v1.json',
    'scripts/flow_matching/audit_source_cnf_full120_v1.py',
    'src/iia_benchmark/models/otcfm_cnf_manual_training_v1.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/CNF_full12_full120_publication_manifest_20261016_v2.json'
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),exact_paths=len(paths),real_input_preflight_passed=12,full_training_models_completed=0)))
