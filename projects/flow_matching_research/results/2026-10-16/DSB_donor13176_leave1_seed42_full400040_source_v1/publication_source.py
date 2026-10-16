"""Publish the real full training completion while keeping native review pending."""
from datetime import datetime,timezone
import csv
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_dsb_first_full_source_receipt_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);s.verify(c);q=read(ROOT/s.dsb.QUEUE);job=c['source_job'];source=Path(job['artifact_directory'])
base=ROOT/c['state_root'];state=read(base/'status.json');audit_guard=read(base/'receipt.json')
proof_path=Path(c['artifact_root'])/'report.json';proof=read(proof_path)
assert state['state']=='complete' and state['active_pid'] is None and state['passed']
assert audit_guard['passed'] and audit_guard['exit_code']==0 and not audit_guard['resource_guard_aborted']
assert audit_guard['proof_sha256']==sha(proof_path) and proof['passed']
assert proof['full_trained_source_conditions']==1 and not proof['full_stochastic_native_replay_complete']
assert proof['actual_source_optimizer_updates']==400040 and proof['complete_original_IPF_passes']==40
assert proof['all209_original_trained_artifacts_unchanged']
training_guard=read(c['source_guard']);result=read(source/'source_result.json');phases=read(source/'phase_progress.json')
assert training_guard['passed'] and training_guard['exit_code']==0 and not training_guard['resource_guard_aborted']
assert training_guard['report_sha256']==sha(source/'source_result.json')==proof['source_result_sha256']
s.verify_phases(phases)
dsb_state=read(ROOT/q['state_root']/'status.json');owner=psutil.Process(dsb_state['pid'])
assert owner.is_running() and owner.create_time()==dsb_state['create_time']
assert not dsb_state['completed_ids'] and not dsb_state['failed_preserved']
assert not (source/'native_replay/report.json').exists()
assert sum((Path(j['artifact_directory'])/'source_result.json').exists() for j in q['jobs'])==1
xml=ROOT/'tmp/DSB_first_full_source_shared_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==37
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/DSB_donor13176_leave1_seed42_full400040_source_v1'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,path in dict(source_result=source/'source_result.json',source_training_guard=Path(c['source_guard']),
    actual40_phase_progress=source/'phase_progress.json',independent_source_and_moment_audit=proof_path,
    independent_audit_guard=base/'receipt.json',actual_audit_complete_state=base/'status.json',actual_audit_launch=base/'launch.json',
    frozen_audit_registration=ROOT/s.CONFIG,frozen_native70_queue=ROOT/s.dsb.QUEUE,
    frozen_original_source_profile=ROOT/s.dsb.MODEL,fresh37_validation=xml).items():shutil.copyfile(path,out/(name+path.suffix))
metric_rows=[]
for metric,value in result['metrics'].items():
    independent=proof['independent_six_moment_metrics'].get(metric)
    metric_rows.append(dict(canonical_id=s.IDENTITY,metric=metric,full_source_value=value,independent_saved_population_value=independent,
        absolute_difference=abs(value-independent) if independent is not None else None,
        status='complete_source_native_LP_OPTIMAL_independent_LP_pending' if independent is None else 'complete_source_and_independent_full_population_moment_verified',
        complete_model_training_seeds=1,required_training_seeds=5,complete_native_checkpoint_replays=0,
        historical_paper_equivalence_certified=False))
with (out/'all8_full_population_source_and_independent_moment_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
phase_rows=[dict(phase=r['phase'],direction=r['direction'],actual_phase_updates=r['actual_phase_updates'],
    cumulative_forward_updates=r['cumulative_updates']['f'],cumulative_backward_updates=r['cumulative_updates']['b']) for r in phases['phases']]
with (out/'all40_actual_optimizer_phase_records.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(phase_rows[0]));writer.writeheader();writer.writerows(phase_rows)
matrix=[]
for j in q['jobs']+q['deferred_jobs']:
    trained=j['id']==s.IDENTITY
    matrix.append(dict(canonical_id=j['id'],dataset=j['dataset'],leftout_timepoint=j['leftout_timepoint'],seed=j['seed'],
        real_full_training_complete=trained,all400040_optimizer_updates_observed=trained,full_population_source_metrics_complete=trained,
        full_stochastic_checkpoint_replay_complete=False,independent_full_W1_complete=False,
        state='full_source_complete_native_replay_pending' if trained else 'pending_full_training' if j['data'] else 'deferred_full_GEO_preprocessing'))
with (out/'all70_original_DSB_condition_completion_matrix.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
write(out/'actual_original70_controller_handle_and_state.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),state=dsb_state,
    actual_handle=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline(),
        numerical_children=[dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()) for p in owner.children(recursive=True) if p.name().lower()=='python.exe']),
    controller_modified_or_restarted=False))
write(out/'completion_boundary.json',dict(original_scope=q['original_scope'],source_complete_conditions=1,required_original_DSB_conditions=70,
    complete_stochastic_native_replays=0,independent_full_W1_conditions=0,complete_five_seed_groups=0,
    source40_passes_complete=True,actual_source_updates=400040,total_source_parameters=28522,
    original_full_LP_entries_each=85579890,original_W1=result['metrics']['1-Wasserstein'],
    original_W2=result['metrics']['2-Wasserstein'],independent_all_six_population_moment_metrics=True,
    source_gamma_to_paper_sigma_mapping_certified=False,historical_paper_equivalence_certified=False,strict_TAB_result=False,
    existing_source32GiB_training_and_replay_gate_unchanged=True,
    additional_neural_forwards_or_training_updates_by_this_closure=0,additional_LP_solves_by_this_closure=0,
    master_metric_catalog_unchanged='v69',all_original_experiments_complete=False))
(out/'README.md').write_text('''# DSB donor13176／留出索引1／seed42：400040次完整源训练更新

第一个真实DSB sourceE4条件已完成源训练及全量分布评估。原20个IPF轮次、每轮backward/forward两阶段，共40个阶段，每阶段原包含端点的10001次优化器更新，总400040次。两网络合计28522参数；全10000缓存粒子、512批量、20步Langevin和原缓存刷新100保持。真实训练守卫退出0、无资源中止，训练本身约5567秒。

全9546×5预测及8965×5真值均保留；W1和W2原线性规划各使用完整9546×8965成本矩阵，即85579890个条目，均返回OPTIMAL。源W1为0.897232965089，W2为0.958097653820。全部八项源指标见 `all8_full_population_source_and_independent_moment_metrics.csv`。

另一个受限纯Python进程检查了全部209项训练产物哈希、40份完整网络/优化器/RNG检查点的存在及完整阶段计数，并从两份完整NPY独立重算全部六项均值/中位数距离，均满足已冻结2e-6容差。中位数按原torch.median的偶数总体下中位数定义，不能改成两个中间值平均。此次审计没有神经网络前向、训练更新或LP重求解；37项相关测试通过。

当前源训练完成1/70，完整随机检查点重放0/70、独立全量W1复核0/70、完整五种子组0。原控制器仍存活并排队执行后续native阶段，原32GiB训练和重放门槛未修改。单条件指标不汇总为论文行或训练种子区间；主指标总表v69不提前填入这项尚未完整原生复核的源成绩。所有70原任务及60可用/10待完整GEO预处理范围见 `all70_original_DSB_condition_completion_matrix.csv`。

该实现使用冻结原IPFSequential的训练/缓存/播种/Langevin方法及原Accelerate0.2.1，通过已披露的端点、论文网络与记录适配运行。Py310/Torch1.12兼容环境不认证原Py38/Torch1.8环境；源gamma0.001与论文sigma1的映射尚未认证。本公共donor13176原始输入不认证历史论文预处理资产字节等价。源训练完成不代表历史论文结果重现或严格TAB成绩。原配置是执行前冻结登记，实际完成状态以上述源结果、守卫和70条件矩阵为准，不改写冻结配置的旧描述字段。

原45篇/681方法记录/273数据任务/702论文指标范围保持；全部训练状态仍未完成。本次只接收已经自然结束且守卫成功的源任务，没有停止或重启数值进程。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/DSB_first_source400040.2026-10-16_v1.json'
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),canonical_id=s.IDENTITY,source_full_training_complete=True,
    source_conditions_complete=1,required_conditions=70,actual_source_updates=400040,full_native_replay_complete=False,
    independent_full_W1_complete=False,original_scope=q['original_scope'],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8');assert out.name not in text
text+='\n`DSB首个真实sourceE4条件完成400040次更新及完整总体源指标 <results/2026-10-16/'+out.name+'/README.md>`_\n'
text+='donor13176／留出索引1／seed42完成40阶段、28522参数、全9546预测/8965真值；两原完整85579890条目LP均OPTIMAL，W1=0.897232965089、W2=0.958097653820。209产物与全部阶段计数、六项总体矩统计独立核验通过，37项测试通过。源1/70，完整随机检查点重放/独立全量W1仍0/70，不计算部分组或提前填主表；原32GiB门槛和全部兼容性/历史等价边界保留，整体未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths.extend(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/DSB_first400040_source_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(packet=str(out),files=len(paths),source_complete=1,full_native_complete=0,tests=37))
