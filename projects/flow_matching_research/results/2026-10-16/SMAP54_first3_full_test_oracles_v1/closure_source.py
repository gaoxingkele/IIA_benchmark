from pathlib import Path
from datetime import datetime,timezone
import csv
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
import numpy as np
from scripts.flow_matching import audit_common105_test_label_oracle_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import write
ROOT,read,sha=s.ROOT,s.read,s.sha
c=read(ROOT/s.CONFIG);s.verify(c);base=ROOT/c['state_root']
ids=['fm_reflow_2stage__smap__seed1103__common105_v1',
    'fm_objective_sb__smap__seed1103__stable_numerics_v2__common105_v1',
    'fm_strict_tranad__smap__reference_float64__seed1103__common105_v1']
state=read(base/'status.json');assert set(ids).issubset(state['completed_cases']) and not state['failed_audits_preserved']
launch=read(base/'launch.json');actor=psutil.Process(launch['pid'])
assert actor.is_running() and actor.create_time()==launch['create_time'] and actor.cmdline()==launch['command']
tests=Path(launch['validation_path']);assert sha(tests)==launch['validation_sha256']
suites=list(ET.parse(tests).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==13
assert not any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped'])
reports={};bindings={};rows=[];summaries={};data_all=[];files={}
for index,identity in enumerate(ids):
    case=next(v for v in c['cases'] if v['id']==identity)
    basis=s.upstream.ready_case(case);queue,job,result,data,binding=s.upstream.upstream.source_binding(c,case)
    path=Path(c['artifact_root'])/identity/s.REPORT;guard_path=base/'attempts'/identity/'guard_receipt.json'
    r,g=read(path),read(guard_path)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(path)
    expected=[str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',s.MODULE,'--job-id',identity]
    assert g['exact_child_handle']['command']==expected
    assert r['passed'] and r['case']==case and r['configuration_sha256']==sha(ROOT/s.CONFIG)
    assert r['source_binding']==binding and r['upstream_scalar_proof']==basis
    assert r['test_labels_used_to_select_oracle_threshold'] and r['oracle_scores_never_enter_strict_leaderboards']
    assert not r['model_training_or_reinference_performed'] and not r['independent_model_reinference_certified']
    assert len(data['entities'])==54 and data['primary_label_policy']=='source_first_exclusive'
    assert set(r['label_view_results'])==set(case['required_label_policies'])
    scalar=read(basis['upstream_report']);data_all.append(data)
    with np.load(Path(binding['result_path']).parent/'scores.npz',allow_pickle=False) as scores:
        for policy,v in r['label_view_results'].items():
            assert v['all_test_points']==435826 and not v['per_entity_threshold_optimization']
            assert not v['label_policy_selected_by_TEST_performance']
            original=scalar['label_view_results'][policy]
            assert v['original_validation_1percent_noPA']==original['metrics']['strict']['1']['micro']
            assert v['original_same_validation_threshold_entitywise_PA']==original['primary_protocol_diagnostics']['same_validation_threshold_entitywise_PA_diagnostic']
            overlay=data['labels_only_controls'][policy];assert sha(overlay['path'])==overlay['sha256']
            with np.load(overlay['path'],allow_pickle=False) as labels:
                for protocol,uses_pa in [('raw',False),('PA',True)]:
                    value=v[protocol];counts=[0,0,0];population=0
                    for entity in sorted(data['entities'],key=lambda e:e['entity_id']):
                        prefix=entity['array_prefix'];y=labels[prefix+'_labels'].astype(bool);x=scores[prefix+'_test']
                        assert x.ndim==1 and x.shape==y.shape and np.isfinite(x).all()
                        pred=x>value['threshold']
                        if uses_pa:
                            padded=np.r_[False,y,False].astype(np.int8);edges=np.diff(padded)
                            for first,last in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                                if np.any(pred[first:last]):pred[first:last]=True
                        counts[0]+=int(np.count_nonzero(y&pred));counts[1]+=int(np.count_nonzero(~y&pred));counts[2]+=int(np.count_nonzero(y&~pred));population+=len(y)
                    assert population==435826 and counts==[value[k] for k in ['true_positives','false_positives','false_negatives']]
                    tp,fp,fn=counts;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                    assert f1==value['f1'] and value['independent_entity_groups']==54
                    assert value['test_labels_used_to_select_threshold'] and value['diagnostic_only']
                    assert value['point_adjustment']==uses_pa and not value['deployable_validation_only_result']
                    rows.append(dict(case_id=identity,seed=case['seed'],label_policy=policy,protocol='test_label_oracle_'+protocol,
                        F1=f1,TP=tp,FP=fp,FN=fn,threshold=value['threshold'],point_adjustment=uses_pa,
                        test_labels_used_to_select_threshold=True,strict_leaderboard_result=False))
    primary=r['label_view_results'][data['primary_label_policy']]
    summaries[identity]=dict(validation_raw_F1=primary['original_validation_1percent_noPA']['f1'],
        validation_PA_F1=primary['original_same_validation_threshold_entitywise_PA']['f1'],
        exact_test_label_raw_oracle_F1=primary['raw']['f1'],exact_test_label_PA_oracle_F1=primary['PA']['f1'],
        seed=1103,full_test_points=435826,full_entities=54,complete_five_training_seed_comparison=False)
    reports[identity]=r;bindings[identity]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),source_binding=binding,
        all4_full_label_view_oracle_predictions_independently_reconstructed=True)
    files[f'model{index}_oracle_report.json']=path;files[f'model{index}_oracle_guard.json']=guard_path
assert data_all[0]==data_all[1]==data_all[2]
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/SMAP54_first3_full_test_oracles_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.csv -text\n*.py -text\n',encoding='utf-8')
for name,p in files.items():shutil.copyfile(p,packet/name)
for name in ['launch.json','status.json']:shutil.copyfile(base/name,packet/('actual_'+name))
shutil.copyfile(tests,packet/'validation_13_cases.xml')
write(packet/'all3_full_model_oracle_bindings.json',bindings);write(packet/'three_model_primary_label_summary.json',summaries)
with (packet/'all3_models_4_label_views_2_oracles.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],complete_oracle_source_models=3,required_source_models=577,
    complete_label_views=12,required_label_views=1726,new_model_training_updates=0,new_model_reinference=0,
    source_native_TAB_gates_changed=False,oracle_values_enter_strict_leaderboard=False,all_original_experiments_complete=False))
lines=['# SMAP54：三个完整固定评分的测试标签阈值上界诊断','',
 'Reflow、SB流匹配适配、TranAD FP64的seed1103使用相同冻结输入、54实体及435826测试点；全部四种标签视图均保留，不选择有利标签。',
 '对所有真实分数并列组穷举严格大于阈值，以整数交叉乘积选择精确最大F1。每个标签视图只允许一个全局阈值，PA严格在原实体内部进行，实体边界异常段不会合并。', '',
 '| 方法 | 验证阈值无PA | 同验证阈值+PA | 测试标签选阈值无PA | 测试标签选阈值+PA |',
 '|---|---:|---:|---:|---:|']
for identity,name in zip(ids,['Reflow','SB FM adaptation','TranAD FP64']):
    v=summaries[identity];lines.append('| '+name+' | '+' | '.join(f'{v[k]:.6f}' for k in ['validation_raw_F1','validation_PA_F1','exact_test_label_raw_oracle_F1','exact_test_label_PA_oracle_F1'])+' |')
lines += ['', '测试标签选阈值的两列是明确的事后诊断，使用了测试标签；PA列也使用真实异常区间。它们不是可部署的验证集成绩，不进入严格排行榜，不称前瞻预注册。',
 '该对照说明固定评分下评价协议可能显著改变F1；不能据此认定具体论文有泄漏，也不能跨数据集或单种子判定算法总体优劣。没有新增模型训练、重推理或训练种子。',
 '每个最终阈值在所有原实体/测试点上重新构造预测，独立事件边界实现及混淆计数复核通过。13项相关测试通过，包括64种二元标签组合的穷举比较、并列分数、实体边界、非有限值和范围约束。',
 '自动诊断保留全部577任务/1726标签视图，当前3任务/12视图实际完成。采用既有轻量纯评分审计资源配置，原训练/原生重放/TAB门槛不变；每个完整诊断后让出共享锁20秒。',
 'NASA原来源文件已做TEST最小最大值缩放的限制保留；本包不认证历史作者实验、全流水线归纳性、完整TAB或尚未结束的原生重推理。',
 '全部45论文/681方法记录/273数据任务/702指标及873项公共比较/2037标签视图、原论文/TAB/新数据实验目标保持，整体未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile('tmp/launch_common105_test_oracle_20261016_v1.py',packet/'launch_source.py');shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/SMAP54_first3_full_test_oracles.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),complete_diagnostic_cases=3,required_cases=577,
    complete_label_views=12,required_label_views=1726,test_label_selected_values_are_strict_leaderboard=False,all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/SMAP54_first3_test_oracles_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),primary_label_metrics=summaries)))
