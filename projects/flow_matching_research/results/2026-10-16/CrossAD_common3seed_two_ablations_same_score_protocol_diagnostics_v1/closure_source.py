"""Nine fixed-score diagnostic cases; no partial-cohort means or benchmark oracles."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_crossad_full_test_label_oracle_v1 as oracle
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/oracle.CONFIG);q,upstream=oracle.verify(c)
state=read(ROOT/c['state_root']/'status.json')
assert not state['failed_preserved']
p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
sources={};rows=[];case_rows=[]
for seed in [1103,1104,1105]:
    for architecture,cid in [('full',f'crossad_author__fresh_train__gecco__seed{seed}'),
            ('row1',f'crossad_ablation__row1__gecco__seed{seed}'),('row2',f'crossad_ablation__row2__gecco__seed{seed}')]:
        assert cid in state['verified_ids']
        path=Path(c['artifact_root'])/'conditions'/(cid+'.json');r=read(path)
        j=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==cid)
        assert oracle.audit(c,q,upstream,j)==r
        guard_path=ROOT/c['state_root']/'guards'/(cid+'.json');guard=read(guard_path)
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(path)
        assert r['complete_points']==dict(train=55408,validation=13852,test=69261)
        assert r['new_model_training_updates']==0 and not r['new_neural_inference_executed']
        assert r['posthoc_test_label_selected_diagnostic'] and r['oracle_scores_never_enter_strict_leaderboards']
        metrics_path=Path(upstream['artifact_root'])/'conditions'/(cid+'.json');strict=read(metrics_path)
        assert sha(metrics_path)==r['source_independent_metric_proof_sha256']
        controls=r['protocol_factorial']['1']['protocols']
        bounds=r['full_test_label_selected_upper_bounds']
        for protocol,value in list(controls.items())+[(f'test_label_selected_{k}',v) for k,v in bounds.items()]:
            test_labels=protocol.startswith('test_label_selected_')
            pa=protocol.endswith('_PA')
            if test_labels:assert value['test_labels_used_to_select_threshold'] and value['diagnostic_only'] and not value['deployable_validation_only_result']
            rows.append(dict(canonical_id=cid,architecture=architecture,training_seed=seed,protocol=protocol,
                F1=value['f1'],precision=value['precision'],recall=value['recall'],threshold=value['threshold'],
                TP=value['true_positives'],FP=value['false_positives'],FN=value['false_negatives'],
                test_scores_used_in_calibration=protocol.startswith('pooled_train_test'),test_labels_used_to_select_threshold=test_labels,
                test_labels_used_in_event_postprocessing=pa,diagnostic_only=protocol!='validation_raw',
                new_training_updates=0,new_neural_forwards=0,full_five_seed_cohort_in_this_packet=False))
        v=strict['independent_strict_metrics']['1']['micro']
        case_rows.append(dict(canonical_id=cid,architecture=architecture,training_seed=seed,
            validation_raw_F1=v['f1'],AUROC=v['auroc'],AP=v['average_precision'],
            pooled_PA_F1=controls['pooled_train_test_PA']['f1'],test_label_selected_raw_F1=bounds['raw']['f1'],test_label_selected_PA_F1=bounds['PA']['f1']))
        sources[cid]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),
            full_score_archive_sha256=r['complete_score_archive_sha256'],payload_sha256=r['all_score_label_and_timestamp_payload_sha256'],
            strict_metric_path=str(metrics_path),strict_metric_sha256=sha(metrics_path))
assert len(rows)==54 and len(case_rows)==9
for seed in [1103,1104,1105]:
    ids=[r['canonical_id'] for r in case_rows if r['training_seed']==seed]
    for field in ['labels','source_rows']:
        assert len({sources[cid]['payload_sha256'][field] for cid in ids})==1
xml=ROOT/'tmp/CrossAD_partial3seed_protocol_diagnostics_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert all(int(s.get('failures',0))==int(s.get('errors',0))==int(s.get('skipped',0))==0 for s in suites)
test_count=sum(int(s.get('tests',0)) for s in suites);assert test_count>=10
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_common3seed_two_ablations_same_score_protocol_diagnostics_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for name,data in [('all54_fixed_score_protocol_diagnostics.csv',rows),('all9_case_ranking_and_calibration_diagnostics.csv',case_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write(packet/'all9_actual_source_bindings.json',dict(records=sources,all9_reports_recomputed_exactly=True,
    actual_live_collector=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),
    common_training_seeds=[1103,1104,1105],required_original_primary_seeds=list(range(1103,1108)),
    partial_training_seed_means_or_intervals_computed=False,new_training_models=0,new_neural_forwards=0,
    original_scope=c['original_scope'],all_original_experiments_complete=False))
for cid,b in sources.items():
    shutil.copyfile(b['report_path'],packet/(cid+'.json'))
    shutil.copyfile(b['guard_path'],packet/(cid+'.guard.json'))
shutil.copyfile(xml,packet/'relevant_fresh_validation.xml')
table='\n'.join(f"| {r['training_seed']} | {r['architecture']} | {r['validation_raw_F1']:.6f} | {r['AUROC']:.6f} | {r['AP']:.6f} | {r['pooled_PA_F1']:.6f} | {r['test_label_selected_PA_F1']:.6f} |" for r in case_rows)
(packet/'README.md').write_text(f'''# CrossAD 两个消融的共同三种子：排序与评价协议诊断

已完整训练、检查点重放和独立数组核验的主模型、row1、row2，在共同种子1103、1104、1105上的9个固定模型，逐一重算全部完整69261测试点。
原始数据、标签与坐标哈希一致。每个模型交叉验证集或train+test评分校准、无PA或PA，再计算测试标签选阈值的raw/PA完整穷举上界，共54行。
所有阈值是严格大于，原始全部并列评分和13点尾部保留。已有自动诊断守卫均退出0、无资源中止；本轮重新计算与保存报告完全一致。

| 种子 | 模型 | 验证集1%无PA F1 | AUROC | AP | train+test评分校准+PA F1 | 测试标签选阈值+PA上界 |
|---|---|---:|---:|---:|---:|---:|
{table}

表中两项替代协议F1的规则必须明确区分：train+test评分校准使用测试评分，PA使用真实异常事件标签；测试标签最优阈值还直接使用测试答案。
测试标签上界只作为事后诊断，绝不并入严格验证集基准成绩。AUROC/AP描述排序，冻结阈值F1还取决于校准和阳性率，因此某个阈值F1接近不代表完整排序质量接近。
同一模型不同规则之间的差值隔离评价协议影响；不同结构模型仍有参数量、实际早停更新量和重建实现差异，不能称纯组件因果结论。
row1关闭全部四组件，row2仅保留multi-scale，均为paper_reconstructed_author_harness，作者精确消融实现未认证。

共同三种子只是5种子计划的3/5子集，不计算部分均值或区间，也不声称两个消融完整完成。
主模型此前完整五种子结果仍见独立五种子报告，本表只为三种子逐项配对而列三行，不替换已有五种子统计。
{test_count}项相关协议与精确穷举检查通过。新增训练模型0、神经网络前向0；重复指标计算不能增加完整训练实验数量。
全部139条件、22主要组及45/681/273/702研究范围保持；不认证历史论文或完整TAB等价，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_common3seed_protocol_diagnostics.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),fixed_models=9,protocol_rows=54,
    new_complete_training_models=0,partial_cohort_means_computed=False,all_original_experiments_complete=False))
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
files=[ara.relative_to(ROOT).as_posix()]+[x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()]
write(ROOT/'tmp/CrossAD_common3seed_protocol_diagnostics_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(packet),cases=9,protocol_rows=54,ranking_table=case_rows)))
