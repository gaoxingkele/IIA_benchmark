from pathlib import Path
from datetime import datetime,timezone
import array
import csv
import json
import math
import shutil
import subprocess
import numpy as np
from scripts.flow_matching import diagnose_timesnet1105_full_replay_v1 as s
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as independent

ROOT,read,sha=s.ROOT,s.read,s.sha
write=s.old.exclusive
c=read(ROOT/s.CONFIG);stage=s.verify(c)
base=ROOT/c['state_root'];out=Path(c['artifact_root'])
report_path=out/'full_replay_diagnostic.json';r=read(report_path);guard=read(base/'guard_receipt.json')
assert guard['diagnostic_execution_complete'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert not guard['strict_native_reproduction_certified'] and guard['report_sha256']==sha(report_path)
assert r['diagnostic_execution_complete'] and not r['strict_native_reproduction_certified']
assert r['strict_historical_failure_preserved'] and not r['numerical_backend_cause_proven']
assert r['new_optimizer_updates']==r['new_training_seed_slots']==0
assert s.preserved_failure(stage)==c['preserved_failure']
for artifact in r['arrays']:assert sha(artifact['path'])==artifact['sha256']
for pin in c['source_receipts']:assert sha(ROOT/pin['path'])==pin['sha256']
queue=read(ROOT/stage['queue']);job=next(j for j in queue['jobs'] if j['id']==stage['job_id'])
source=ROOT/job['output_directory'];original=read(source/'result.json')
dataset=next(d for d in read(ROOT/job['dataset_manifest_path'])['datasets'] if d['id']==job['dataset'])
streams=[];old_tests=[];new_tests=[];labels=[];old_validation=[];new_validation=[]
with np.load(source/'scores.npz',allow_pickle=False) as old,np.load(out/'aligned_scores.npz',allow_pickle=False) as replay,\
        np.load(out/'raw_window_scores_and_starts.npz',allow_pickle=False) as raw,\
        np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs:
    for entry in dataset['entities']:
        prefix=entry['array_prefix']
        assert np.array_equal(replay[prefix+'_labels'],old[prefix+'_labels'])
        assert np.array_equal(replay[prefix+'_labels'],inputs[prefix+'_labels'])
        for split in ['train','validation','test']:
            key=prefix+'_'+split;a,b=old[key],replay[key];length=len(inputs[key])
            assert a.shape==b.shape==(length,) and np.isfinite(a).all() and np.isfinite(b).all()
            windows=raw[key+'__window_scores'];starts=raw[key+'__window_starts']
            expected=list(range(0,length-99,100))
            if expected[-1]!=length-100:expected.append(length-100)
            assert np.array_equal(starts,np.asarray(expected,dtype=np.int64))
            assert windows.shape==(len(starts),100) and np.isfinite(windows).all()
            indices=(starts[:,None]+np.arange(100)).reshape(-1)
            counts=np.bincount(indices,minlength=length)
            reconstructed=np.bincount(indices,weights=windows.astype(np.float64).reshape(-1),minlength=length)/counts
            assert np.all(counts>0) and np.array_equal(reconstructed,b)
            delta=b-a;comparison=s.compare_scores(b,a)
            recorded=next(x for x in r['streams'] if x['key']==key)
            assert all(recorded[k]==v for k,v in comparison.items())
            index=int(np.argmax(np.abs(delta)))
            streams.append(dict(entity=entry['entity_id'],split=split,key=key,**comparison,
                maximum_scaled_difference=float(np.max(np.abs(delta)/(1+np.abs(a)))),
                largest_absolute_difference_row=index,original_value_at_largest_difference=float(a[index]),
                replay_value_at_largest_difference=float(b[index])))
            if split=='test':old_tests.append(a);new_tests.append(b);labels.append(old[prefix+'_labels'])
            if split=='validation':old_validation.append(a);new_validation.append(b)
assert len(streams)==153 and len(labels)==51
ys=np.concatenate(labels).astype(np.uint8);a=np.concatenate(old_tests);b=np.concatenate(new_tests)
va=np.concatenate(old_validation);vb=np.concatenate(new_validation)
assert len(ys)==408160 and len(va)==len(vb)==26684
label_vector=array.array('B',ys.tolist())
rankings={}
for tag,scores,metrics in [('original',a,r['original_saved_metrics']),('replayed',b,r['replayed_metrics'])]:
    value=independent.ranking(array.array('d',scores.tolist()),label_vector)
    expected=metrics['strict']['1']['micro']
    for name,result in value.items():assert math.isclose(result,expected[name],rel_tol=1e-12,abs_tol=1e-12)
    rankings[tag]=value
controls={};rows=[]
for ratio in ['0.1','1','5']:
    t0=float(np.percentile(va,100-float(ratio),method='linear'))
    t1=float(np.percentile(vb,100-float(ratio),method='linear'))
    assert t0==r['original_saved_metrics']['strict'][ratio]['calibration']['threshold']
    assert t1==r['replayed_metrics']['strict'][ratio]['calibration']['threshold']
    values={}
    for name,scores,threshold in [('original',a,t0),('replay_original_threshold',b,t0),('replay_own_validation_threshold',b,t1)]:
        micro=independent.at_threshold(array.array('d',scores.tolist()),label_vector,threshold)['micro']
        values[name]=micro
    saved=r['fixed_original_threshold_controls'][ratio]
    assert values['original']['f1']==saved['original']['F1']
    assert values['replay_original_threshold']['f1']==saved['replay_at_same_original_threshold']['F1']
    n0=int(np.count_nonzero((a>t0)!=(b>t0)));n1=int(np.count_nonzero((a>t0)!=(b>t1)))
    assert n0==saved['prediction_disagreements']==0 and n1==0
    assert values['original']==values['replay_original_threshold']==values['replay_own_validation_threshold']
    for name,expected in [('original',r['original_saved_metrics']['strict'][ratio]['micro']),
                          ('replay_own_validation_threshold',r['replayed_metrics']['strict'][ratio]['micro'])]:
        assert all(values[name][k]==expected[k] for k in values[name])
    controls[ratio]=dict(original_threshold=t0,replayed_validation_threshold=t1,
        all_three_full_confusion_matrices=values,same_original_threshold_prediction_disagreements=n0,
        own_replay_validation_threshold_prediction_disagreements=n1)
    rows.append(dict(validation_alarm_percent=ratio,original_threshold=t0,replay_validation_threshold=t1,
        original_F1=values['original']['f1'],replay_F1=values['replay_own_validation_threshold']['f1'],
        original_precision=values['original']['precision'],replay_precision=values['replay_own_validation_threshold']['precision'],
        original_recall=values['original']['recall'],replay_recall=values['replay_own_validation_threshold']['recall'],
        original_AUROC=rankings['original']['auroc'],replay_AUROC=rankings['replayed']['auroc'],
        original_AP=rankings['original']['average_precision'],replay_AP=rankings['replayed']['average_precision'],
        same_threshold_prediction_disagreements=n0,own_validation_threshold_prediction_disagreements=n1,
        independent_training_models=1,strict_native_reproduction_certified=False))
aggregate={}
for split in ['train','validation','test']:
    entries=[x for x in streams if x['split']==split]
    maximum=max(entries,key=lambda x:x['maximum_absolute_difference'])
    aggregate[split]=dict(points=sum(x['points'] for x in entries),changed_points=sum(x['changed_points'] for x in entries),
        maximum_absolute_difference=maximum['maximum_absolute_difference'],
        original_value_at_maximum=maximum['original_value_at_largest_difference'],
        replay_value_at_maximum=maximum['replay_value_at_largest_difference'],
        maximum_scaled_difference=max(x['maximum_scaled_difference'] for x in entries),
        maximum_difference_entity=maximum['entity'])
assert sum(x['points'] for x in aggregate.values())==536409
assert sum(x['changed_points'] for x in aggregate.values())==415618
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet1105_full_diagnostic_complete_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
shutil.copyfile(report_path,packet/'full_native_diagnostic_report.json')
shutil.copyfile(base/'guard_receipt.json',packet/'actual_full_diagnostic_guard.json')
write(packet/'independent_full_array_and_metric_verification.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    diagnostic_report_sha256=sha(report_path),all153_raw_window_to_timestamp_alignments_independently_reconstructed=True,
    all536409_saved_and_replayed_points_independently_compared=True,all51_original_labels_and_rows_verified=True,
    score_scale_and_changed_point_summary=aggregate,full408160_test_ranking_metrics_independently_recomputed=rankings,
    all_three_threshold_controls=controls,old_strict_failure_unchanged=s.preserved_failure(stage),
    strict_native_reproduction_certified=False,numerical_backend_cause_proven=False,new_training_updates=0,
    all_original_experiments_complete=False))
for name,data in [('all153_stream_discrepancies.csv',streams),('all_three_full_test_metric_comparisons.csv',rows)]:
    with (packet/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write(packet/'immutable_original_and_diagnostic_artifact_bindings.json',dict(
    original_checkpoint=next(x for x in c['source_receipts'] if x['path'].endswith('model.pt')),
    original_scores=next(x for x in c['source_receipts'] if x['path'].endswith('scores.npz')),
    arrays=r['arrays'],diagnostic_configuration=s.CONFIG,diagnostic_configuration_sha256=sha(ROOT/s.CONFIG),
    original_failed_native_guard=c['preserved_failure'],original_model_state_unchanged=r['original_model_state_unchanged']))
(packet/'README.md').write_text(f'''# TimesNet seed1105：全51实体重放诊断完成

原28132633参数CPU float32检查点按原2线程、batch128、window100完成全部51实体、153个训练/验证/测试流重放，原28GiB提交内存门槛和3/6GiB进程上限保留。进程退出0、无资源中止；没有新增训练、权重更新或独立训练种子。

全部536409个对齐分数中415618个数值不同，原逐字节失败得到全范围量化，不能改成原严格证书通过。独立从逐窗口分数与窗口起点使用bincount重建全部153个时间戳序列，精确匹配诊断保存数组；原数据、标签、检查点和失败记录哈希保持。

对408160个完整测试点，用独立标准库重新计算精确混淆计数、F1、tie-aware AUROC及AP。验证集0.1%、1%、5%三种告警率下，使用同一原验证阈值或各自完整验证分数重算阈值，预测不一致点数都为0，precision/recall/F1完全相同。1%阈值F1仍为{rows[1]['original_F1']:.12f}。

AUROC从{rankings['original']['auroc']:.15f}变为{rankings['replayed']['auroc']:.15f}，AP从{rankings['original']['average_precision']:.15f}变为{rankings['replayed']['average_precision']:.15f}。这些排序指标的微小变化已原值保留，没有四舍五入成完全相同。

最大绝对测试分数差为{aggregate['test']['maximum_absolute_difference']:.9g}，对应原分数{aggregate['test']['original_value_at_maximum']:.9g}；全测试最大abs(delta)/(1+abs(original))为{aggregate['test']['maximum_scaled_difference']:.9g}。同时给出尺度，不能只凭绝对差判断影响。

这份同一训练模型的对照说明：本次重放差异没有改变这三种阈值规则下的F1，因此不能用该差异解释这个注册模型的低F1或它与论文指标的差距。该结论仅限当前51实体注册子集、单一训练种子与固定评分规则；不是全部NASA54/109实体、五种子稳定性或作者论文实验的认证。原训练只保留配置3 epoch预算，实际epoch/update轨迹未保存。

原逐字节失败、原未通过状态和所有历史原件保留；未放宽容差，未证明内存布局/后端/历史运行过程的因果解释。全量诊断完成不增加严格benchmark通过数，不自动解锁依赖原严格证书的队列。81项相关检查已在原诊断启动前通过，本次另外执行了全数组对齐、逐点差异、完整独立排名及阈值计数核验。

原论文/TAB/新数据、45论文/681方法记录/273数据任务/702既有标量目标及全部未完成项继续保留，整体目标未完成。
''',encoding='utf-8')
write(packet/'iteration_status.json',dict(diagnostic_execution_complete=True,full_entities=51,full_streams=153,
    full_test_points=408160,independent_test_rankings_and_thresholds_verified=True,
    new_complete_positive_benchmark_conditions=0,new_training_seed_slots=0,
    strict_native_reproduction_certified=False,all_original_experiments_complete=False))
shutil.copyfile(__file__,packet/'independent_closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=x.relative_to(packet).as_posix(),sha256=sha(x))
    for x in sorted(packet.rglob('*')) if x.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet1105_full_diagnostic_complete.2026-10-16_v1.json'
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),diagnostic_execution_complete=True,
    strict_native_reproduction_certified=False,new_complete_positive_benchmark_conditions=0,
    all_original_experiments_complete=False))
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()]
write(ROOT/'tmp/TimesNet1105_full_diagnostic_complete_manifest_20261016_v1.json',dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),aggregate=aggregate,
    rankings=rankings,all_fixed_rule_prediction_disagreements=0,strict_native_reproduction_certified=False)))
