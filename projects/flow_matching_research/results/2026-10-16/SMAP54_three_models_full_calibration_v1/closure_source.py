"""Audit every full fixed prediction and label view; publish three matched models."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import shutil
import subprocess
import numpy as np
import psutil
from scripts.flow_matching import audit_common105_validation_calibration_v1 as d

ROOT,read,sha,write=d.ROOT,d.read,d.sha,d.write
c=read(ROOT/d.CONFIG);d.verify(c)
ids=['fm_reflow_2stage__smap__seed1103__common105_v1',
 'fm_objective_sb__smap__seed1103__stable_numerics_v2__common105_v1',
 'fm_strict_tranad__smap__reference_float64__seed1103__common105_v1']
state=read(ROOT/c['state_root']/'status.json')
assert set(ids).issubset(state['completed_cases']) and not state['failed_audits_preserved']
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']
records={};rows=[];summaries={};datasets=[];checks=[];paths_to_copy={}
def count(y,p):
    y=y.astype(bool)
    return [int(np.sum(y&p)),int(np.sum(~y&p)),int(np.sum(y&~p)),int(np.sum(~y&~p))]
def metrics(v):
    tp,fp,fn,tn=map(int,v);pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.
    return dict(precision=pr,recall=re,f1=2*pr*re/(pr+re) if pr+re else 0.,TP=tp,FP=fp,FN=fn,TN=tn)
for index,identity in enumerate(ids):
    case=next(x for x in c['cases'] if x['id']==identity)
    basis=d.ready_case(case);assert basis and basis['kind']=='source_complete'
    queue,job,result,data,binding=d.upstream.source_binding(c,case)
    report_path=Path(c['artifact_root'])/identity/d.REPORT
    guard_path=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json'
    report,guard=read(report_path),read(guard_path)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(report_path)
    assert report['passed'] and report['case']==case and report['source_binding']==binding and report['upstream_scalar_proof']==basis
    assert report['configuration_sha256']==sha(ROOT/d.CONFIG)
    assert report['original_primary_global_metrics_reproduced_exactly'] and report['all_entities_retained']
    assert not report['any_test_label_or_score_used_to_fit_threshold'] and not report['model_training_or_reinference_performed']
    assert not report['independent_model_reinference_certified_by_this_diagnostic'] and not report['full_TAB_or_historical_author_equivalence_certified']
    assert sha(report['predictions_path'])==report['predictions_sha256']
    assert len(data['entities'])==54 and data['grouped_splits_verified']
    assert data['primary_label_policy']=='source_first_exclusive'
    datasets.append(data)
    scalar=read(basis['upstream_report'])
    with np.load(Path(binding['result_path']).parent/'scores.npz',allow_pickle=False) as scores,np.load(report['predictions_path'],allow_pickle=False) as predictions:
        entities=sorted(data['entities'],key=lambda x:x['entity_id'])
        validation={e['entity_id']:np.asarray(scores[e['array_prefix']+'_validation'],dtype=np.float64) for e in entities}
        threshold=float(np.quantile(np.concatenate(list(validation.values())),.99,method='linear'))
        assert threshold==report['thresholds']['global_threshold']
        assert sum(len(scores[e['array_prefix']+'_test']) for e in entities)==435826
        for e in entities:
            name,prefix=e['entity_id'],e['array_prefix'];test=scores[prefix+'_test'];v=validation[name]
            assert np.isfinite(test).all() and np.isfinite(v).all()
            own=float(np.quantile(v,.99,method='linear'))
            assert own==report['thresholds']['per_entity_thresholds'][name]
            assert np.array_equal(test>threshold,predictions[prefix+'_global_prediction'])
            assert np.array_equal(test>own,predictions[prefix+'_entity_prediction'])
        for policy,label in data['labels_only_controls'].items():
            assert sha(label['path'])==label['sha256']
            view=report['label_view_results'][policy]
            assert not view['point_adjustment'] and not view['label_policy_selected_by_TEST_performance']
            a=[];b=[]
            with np.load(label['path'],allow_pickle=False) as labels:
                for e,entry in zip(entities,view['per_entity']):
                    name,prefix=e['entity_id'],e['array_prefix'];assert name==entry['entity']
                    y=labels[prefix+'_labels'];assert np.isin(y,[0,1]).all()
                    ca=count(y,predictions[prefix+'_global_prediction']);cb=count(y,predictions[prefix+'_entity_prediction'])
                    assert metrics(ca)==entry['global_policy'] and metrics(cb)==entry['entity_policy']
                    a.append(ca);b.append(cb)
            a,b=np.array(a),np.array(b)
            assert metrics(a.sum(axis=0))==view['global_policy'] and metrics(b.sum(axis=0))==view['entity_policy']
            assert int(a.sum())==int(b.sum())==435826
            rng=np.random.default_rng(1103);diffs=[]
            for _ in range(200):
                chosen=rng.integers(0,54,54)
                diffs.append(metrics(b[chosen].sum(axis=0))['f1']-metrics(a[chosen].sum(axis=0))['f1'])
            u=view['paired_entity_uncertainty']
            assert u['seed']==1103 and u['resamples']==200 and u['independent_entity_groups']==54
            assert diffs==u['paired_F1_difference_samples']
            ci=np.quantile(diffs,[.025,.975],method='linear').tolist()
            assert all(math.isclose(x,y,rel_tol=1e-14,abs_tol=1e-14) for x,y in zip(ci,u['entity_bootstrap_F1_difference_ci95']))
            for name in ['global_policy','entity_policy']:
                rows.append(dict(case=identity,label_policy=policy,validation_calibration=name,**view[name],
                    paired_entity_minus_global_F1=u['observed_F1_difference'],paired_entity_ci95_low=ci[0],paired_entity_ci95_high=ci[1]))
    primary=report['label_view_results'][data['primary_label_policy']]
    summaries[identity]=dict(parameters=result['parameter_count'],source_global_validation=primary['global_policy'],
        source_entity_validation=primary['entity_policy'],paired_entity_difference_CI95=primary['paired_entity_uncertainty']['entity_bootstrap_F1_difference_ci95'],
        ROC_AUC=result['metrics']['strict']['1']['micro']['auroc'],average_precision=result['metrics']['strict']['1']['micro']['average_precision'],
        observed_training_loss_count=len(result.get('training_losses') or []),
        source_model_configuration=job['model_config'],frozen_source_job=job,
        source_model_reinference_certified=False,complete_five_training_seed_comparison=False)
    records[identity]=dict(report_path=str(report_path),report_sha256=sha(report_path),guard_path=str(guard_path),guard_sha256=sha(guard_path),
        source_binding=binding,full_scalar_report_sha256=sha(basis['upstream_report']),predictions_sha256=report['predictions_sha256'])
    checks.append(dict(case=identity,all54_validation_thresholds_and_full_predictions_independently_recomputed=True,
        all4_full_label_views_confusion_and_200_paired_entity_resamples_independently_recomputed=True))
    if index>0:
        for name,path in dict(source_result=Path(binding['result_path']),source_guard=Path(case['source_guard']),
            scalar_report=Path(basis['upstream_report']),calibration_report=report_path,calibration_guard=guard_path,
            frozen_model=ROOT/job['model_config']).items():paths_to_copy[f'model{index}_{name}.json']=path
assert datasets[0]==datasets[1]==datasets[2]
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/common105_Reflow_SMAP54_seed1103_validation_calibration_v1/complete_calibration_diagnostic.json'
assert sha(prior)==records[ids[0]]['report_sha256']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/SMAP54_three_models_full_calibration_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
for name,path in paths_to_copy.items():shutil.copyfile(path,packet/name)
write(packet/'three_model_actual_source_summary.json',summaries)
write(packet/'full_prediction_recomputation_and_hash_bindings.json',dict(records=records,checks=checks,
    identical_frozen_input_and_full_entity_split_metadata=True,original_Reflow_report_unchanged=True,
    captured_utc=datetime.now(timezone.utc).isoformat()))
write(packet/'full54_entity_input_binding.json',datasets[0])
with (packet/'all3_models_4_labels_2_calibrations.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],completed_fixed_model_diagnostics=3,
    required_diagnostic_cases=577,completed_label_views=12,required_label_views=1726,
    model_training_or_reinference_performed_by_this_analysis=False,new_original_model_training_credit=0,
    full_native_model_replay_certified=False,full_TAB_certified=False,all_original_experiments_complete=False))
lines=['# SMAP54：三个完整源模型的固定验证校准诊断','',
 'Reflow两阶段、SB流匹配适配、TranAD FP64的seed1103源评分全部完成。三者冻结输入SHA256、全部54实体切分与预处理元数据完全相同，完整435826测试点保留。',
 '独立重算54个实体的验证99百分位、全部固定预测、四种标签视图混淆计数及每视图200次配对实体bootstrap，与实际诊断报告一致。', '',
 '| 方法 | 统一验证阈值F1 | 逐实体验证阈值F1 | 差值95%实体区间 | ROC AUC | AP |',
 '|---|---:|---:|---|---:|---:|']
for identity,name in zip(ids,['Reflow 2-stage','SB FM adaptation','TranAD FP64']):
    x=summaries[identity];ci=x['paired_entity_difference_CI95']
    lines.append(f"| {name} | {x['source_global_validation']['f1']:.6f} | {x['source_entity_validation']['f1']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {x['ROC_AUC']:.6f} | {x['average_precision']:.6f} |")
lines += ['', '表中固定主标签为source_first_exclusive、验证1%报警预算、无PA。全部四种标签结果均保留，不按测试表现选择标签或阈值。',
 '逐实体校准的差值区间在本次TranAD模型上为正；两种流匹配适配的区间均跨0。该观察提示校准影响依赖评分模型，不能把所有低分归因于统一阈值。',
 '这是看到源聚合结果后明确登记的事后固定评分诊断。阈值只拟合验证评分，不拟合测试评分或标签；标签只用于计算结果。训练预算和参数量不同，不作等算力结论。',
 '区间条件于一个固定训练模型、54实体，不是五训练种子区间，也不是跨数据集结论。尚不判断算法总体优劣。Reflow和SB为本地时间序列适配；不代表原论文作者实验认证。',
 'NASA原始来源文件已使用TEST最小最大值预缩放的限制保留，因此不认证整个原始数据生成流程的全局归纳性。',
 '本包不执行新训练或模型重推理，完整原生检查点重推理与TAB仍由原队列继续，不能用这次固定评分核验抵扣。',
 '已完成诊断3/577、标签视图12/1726；原45论文/681方法记录/273数据任务/702指标及所有原论文/TAB/新数据未完成实验仍保留。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/SMAP54_three_models_full_calibration.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),full_fixed_model_diagnostics=3,
    required_cases=577,full_label_views=12,required_label_views=1726,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/SMAP54_three_model_calibration_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),exact_files=len(paths),metrics={k:{'global_F1':v['source_global_validation']['f1'],'entity_F1':v['source_entity_validation']['f1'],'ci':v['paired_entity_difference_CI95']} for k,v in summaries.items()})))
