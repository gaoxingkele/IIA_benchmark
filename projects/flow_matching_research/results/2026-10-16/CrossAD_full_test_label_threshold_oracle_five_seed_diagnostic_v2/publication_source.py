from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_crossad_full_test_label_oracle_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);q,upstream=s.verify(c)
state_path=ROOT/c['state_root']/'status.json';state=read(state_path)
expected={f'crossad_author__fresh_train__gecco__seed{i}' for i in s.source.groups.SEEDS}
assert expected.issubset(state['verified_ids']) and not state['failed_preserved']
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']
tests=Path('F:/aicoding/IIA_Data/experiments/CrossAD_full_test_label_oracle_20261016_v2_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==15
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
reports={};bindings=[]
for identity in sorted(expected):
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
    path=Path(c['artifact_root'])/'conditions'/(identity+'.json');report=read(path)
    assert report==s.audit(c,q,upstream,job)
    guard=read(ROOT/c['state_root']/'guards'/(identity+'.json'))
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(path)
    # Check every oracle improves or equals all four fixed-policy controls;
    # optimization is over every real global threshold,not over model seeds.
    raw=report['full_test_label_selected_upper_bounds']['raw']
    pa=report['full_test_label_selected_upper_bounds']['PA']
    for budget,group in report['protocol_factorial'].items():
        for protocol,record in group['protocols'].items():
            best=pa if protocol.endswith('_PA') else raw
            assert best['f1']+1e-15>=record['f1']
    assert raw['evaluated_full_test_points']==pa['evaluated_full_test_points']==69261
    assert pa['f1']>=raw['f1'] and raw['test_labels_used_to_select_threshold'] and pa['test_labels_used_to_select_threshold']
    reports[identity]=report
    bindings.append(dict(canonical_id=identity,seed=job['seed'],report_sha256=sha(path),
        raw_sha256=job['raw_sha256'],complete_score_archive_sha256=report['complete_score_archive_sha256'],
        source_independent_metric_proof_sha256=report['source_independent_metric_proof_sha256'],
        all_score_label_and_timestamp_payload_sha256=report['all_score_label_and_timestamp_payload_sha256']))
summary=s.cohort_summary(q,reports)
assert summary==read(Path(c['artifact_root'])/'all22_oracle_diagnostic_cohorts.json')
assert summary['complete_primary_cohorts']==1 and len(summary['cohorts'])==22
primary=summary['cohorts']['GECCO/main']
assert primary['verified_training_seeds']==s.source.groups.SEEDS

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_full_test_label_threshold_oracle_five_seed_diagnostic_v2'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.log -text -whitespace\n*.py -text\n',encoding='utf-8')
for identity in sorted(expected):
    out=packet/'all5_full_test_diagnostics'/identity;out.mkdir(parents=True)
    shutil.copyfile(Path(c['artifact_root'])/'conditions'/(identity+'.json'),out/'full_test_oracle_report.json')
    shutil.copyfile(ROOT/c['state_root']/'guards'/(identity+'.json'),out/'process_guard.json')
write(packet/'all22_complete_or_partial_cohorts.json',summary)
write(packet/'all5_source_and_complete_score_bindings.json',dict(records=bindings,new_neural_forwards=0,new_model_training_updates=0))
shutil.copyfile(state_path,packet/'actual_controller_state.json')
shutil.copyfile(tests,packet/'validation_15_checks.xml')
shutil.copyfile(__file__,packet/'publication_source.py')
shutil.copyfile(ROOT/'tmp/launch_crossad_full_test_label_oracle_20261016_v1.py',packet/'launch_source.py')
launch=Path('F:/aicoding/IIA_Data/experiments/crossad_full_test_label_oracle_v1_launch')
for name in ['launch.json','stdout.log','stderr.log']:shutil.copyfile(launch/name,packet/('actual_launch_'+name))
rows=[]
for identity,r in sorted(reports.items()):
    for protocol,values in r['full_test_label_selected_upper_bounds'].items():
        rows.append(dict(canonical_id=identity,training_seed=r['seed'],protocol='test_label_selected_'+protocol,**values))
with (packet/'all5_models_10_full_test_label_selected_upper_bounds_NOT_leaderboard.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
table_rows=[]
for name,values in primary['by_alarm_budget']['1']['protocol_F1'].items():
    table_rows.append(dict(protocol=name,source='fixed_1percent_alarm_budget',**values))
for name,values in primary['test_label_selected_F1_upper_bounds'].items():
    table_rows.append(dict(protocol='test_label_selected_'+name,source='optimistic_posthoc_global_threshold_upper_bound',**values))
with (packet/'five_seed_protocol_comparison_NOT_algorithm_leaderboard.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(table_rows[0]));writer.writeheader();writer.writerows(table_rows)
table=['| 评价规则 | 五种子平均F1 | 描述性均值t95区间 |','|---|---:|---|']
for row in table_rows:
    low,high=row['mean_t95_interval'];table.append(f'| {row["protocol"]} | {row["mean"]:.6f} | [{low:.6f}, {high:.6f}] |')
delta=primary['paired_oracle_PA_minus_validation_1percent_raw']
(packet/'README.md').write_text('''# CrossAD GECCO：测试标签选阈值的完整评分诊断上界

本轮对原来已完整训练/重放的5个主要种子1103..1107分别完成全69261测试点的最佳全局单阈值穷举，另计算含PA的同一上界；未增加模型训练或神经网络推理。
全部原始训练、验证、测试评分，标签、原始行号和尾部覆盖均绑定此前完整源执行与独立指标证明的哈希，不修改任何原始结果。

算法完整枚举全部不同评分的并列组及不报任何警的集合，严格采用score>threshold，并用整数F1分子分母交叉相乘选最优，精确同分时取更保守的最高阈值。
PA上界先把每段真实异常事件的分数替换为该段最大分数，正常点分数保留；这与任意阈值下把命中的真实异常段补齐完全相同。
选出的阈值再次应用到全部原始评分，然后独立重构全体预测/PA及TP/FP/FN/F1，要求与排序扫描结果完全一致。
测试标签既参与最优阈值选择，又在PA条件参与事件补全；这是乐观的事后诊断上界，绝不能写入严格验证集基准成绩。

固定1%告警预算的四项控制与两个事后上界，共用同一组完整评分：

'''+ '\n'.join(table)+f'''

测试标签选择阈值加PA，相比验证集校准/无PA，配对F1平均增加{delta['mean']:.6f}，五种子描述性t95为{delta['mean_t95_interval']}。
即使采用这个乐观上界，本组GECCO也约70%，没有达到90%以上。该结果不能推广到其他数据集或论文，也不能把论文高分一概归咎于校准或PA。
上界仅覆盖原始拼接时间线上的全局单阈值和此处明确的PA规则；不覆盖按实体选阈值、其他后处理、其他数据版本/划分或模型结构。
五种子区间只描述同一固定文件上的训练随机性，并不提供未知实体泛化、算法优越性或完整实验族显著性证据。

新队列保留全部139原始条件、22个主要五种子组，当前诊断完成5/139、完整组1/22。控制种子2025及发布检查不混入主要均值。
15项检查通过，包括对全部小型二值标签组合及并列评分模式进行暴力穷举对照、PA变换等价、精确边界和明确标注测试标签参与。
未执行的最初注册草案仍保留；正式v2配置把测试标签参与标志明确设为true，同时单独标注四项固定控制不使用测试标签选阈值。草案没有启动数值进程。

当前新增完整训练模型数为0，新增完整评分诊断为5。原始CPU/GPU训练、完整TAB、新数据与其他方法/消融实验继续执行，全部45/681/273/702范围保持，整体目标仍未完成。
''',encoding='utf-8')
handles=[]
for pid,ct in [(state['pid'],state['create_time']),(36292,1792136499.03037),
    (34576,1792136313.8145552),(1784,1792136313.8267477),(36912,1792097263.1796665),(41992,1791948094.8603492)]:
    p=psutil.Process(pid);assert p.is_running() and p.create_time()==ct
    handles.append(dict(pid=pid,create_time=ct,command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])))
write(packet/'actual_live_original_workers_and_diagnostic_controller.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],complete_full_test_diagnostic_conditions=5,
    required_conditions=139,complete_primary_cohorts=1,required_primary_cohorts=22,new_model_training_updates=0,
    new_neural_forwards=0,new_original_full_training_models=0,oracle_scores_enter_strict_leaderboards=False,
    original_numerical_workers_stopped=False,all_original_experiments_complete=False))
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_full_test_label_oracle.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(configuration=s.CONFIG,result_packet=packet.relative_to(ROOT).as_posix(),complete_diagnostic_conditions=5,
    required_conditions=139,new_full_training_models=0,strict_TAB_result=False,original_scope=q['original_scope'],all_original_experiments_complete=False))
paths=[s.CONFIG,c['superseded_unexecuted_metadata_draft'],s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/CrossAD_full_test_oracle_publication_manifest_20261016_v1.json'
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),exact_files=len(paths),primary_oracle_F1=primary['test_label_selected_F1_upper_bounds'],paired_delta=delta)))
