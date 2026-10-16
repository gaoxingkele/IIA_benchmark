"""Close exact guarded fixed-checkpoint cases; retain all78 tasks and failures."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import psutil
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as native
from scripts.flow_matching import run_moment_TAB100_metadata_recovery_v2 as correction
from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical,require_same_metrics,binary_counts
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

config_path=correction.ORIGINAL
runtime=read(ROOT/config_path);cases=native.verify(runtime);runtime['_config_path']=config_path
recovery=read(ROOT/correction.CONFIG);recovered_runtime,recovered_cases=correction.verify(recovery)
recovered_runtime['_config_path']=correction.CONFIG
private=correction.corrected_private_module((ROOT/correction.SOURCE).read_text(encoding='utf-8'))
state_bytes=(ROOT/runtime['state_root']/'status.json').read_bytes();state=json.loads(state_bytes)
recover_bytes=(ROOT/recovery['state_root']/'status.json').read_bytes();recover_state=json.loads(recover_bytes)
assert not recover_state['failed_preserved'] and len(state['failed_preserved'])==13
ids=set(state['verified_ids']);recovered=set(recover_state['verified_ids'])
assert len(ids)==19 and len(recovered)==3 and not ids&recovered
complete=ids|recovered;assert len(complete)==22
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/MOMENT_small_TAB100_SMAP_full_native_and_raw51_reconstruction_v1'
for pin in read(prior/'packet_integrity.json')['files']:assert sha(prior/pin['path'])==pin['sha256']
old=set(read(prior/'original78_collector_snapshot.json')['verified_ids'])|set(read(prior/'corrected39_collector_snapshot.json')['verified_ids'])
expected_new={'fm_moment_pretrained_base__release512__tep_classic__seed42','fm_moment_pretrained_base__release512__skab__seed42','fm_moment_pretrained_small__TAB100__msl__seed42'}
assert len(old)==19 and complete==old|expected_new and not old&expected_new
owners=[]
for s in [state,recover_state]:
    p=psutil.Process(s['pid']);assert p.is_running() and p.create_time()==s['create_time']
    owners.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
bindings={};archive_sources={};records=[];pair_refs={};selected=[]
for cid in sorted(complete):
    is_recovered=cid in recovered
    actual_runtime=recovered_runtime if is_recovered else runtime
    config=correction.CONFIG if is_recovered else config_path
    module=correction.MODULE if is_recovered else native.MODULE
    case=cases[cid];job=case['job']
    if is_recovered:
        admission=correction.metadata_failure(case,recovered_runtime)
        assert admission and admission==recover_state['diagnosed_admissions'][cid]
        source,data,binding=private.check_source(actual_runtime,case)
    else:source,data,binding=native.check_source(actual_runtime,case)
    directory=Path(actual_runtime['artifact_root'])/cid
    np_path=directory/'native/native_replay.json';rp=directory/'independent/review.json'
    n,r=read(np_path),read(rp)
    assert n['passed'] and r['passed'] and n['job']==r['job']==job
    assert n['source_binding']==r['source_binding']==binding and n['runtime_sha256']==sha(ROOT/config)
    assert r['native_replay_sha256']==sha(np_path) and n['native_coverage']==source['native_coverage']
    assert n['all_original_groups_reinferred'] and n['all_array_dtype_shape_content_identical']
    assert r['all_original_arrays_independently_checked'] and r['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert n['training_updates_performed']==0 and not n['new_training_seed'] and not r['new_training_seed']
    assert not n['historical_paper_equivalence_certified'] and not n['full_TAB_harness_certified']
    assert sha(n['scores_path'])==n['scores_sha256']
    assert n['inline_full_window_MSE_check']['windows']==sum(x['windows'] for x in n['native_coverage'])
    selected_files=dict(native=np_path,review=rp)
    for stage,proof in [('native',np_path),('independent',rp)]:
        base=ROOT/actual_runtime['state_root']/'attempts'/cid/stage
        gp=base/'guard_receipt.json';guard=read(gp)
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
        assert guard['proof_sha256']==sha(proof)
        expected=[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',module,'--config',config,'--stage',stage,'--job-id',cid]
        assert guard['exact_child_handle']['command']==expected and guard['exact_child_handle']==read(base/'launch.json')
        selected_files[stage+'_guard']=gp
    counts={role:sum(x['native_points'] for x in n['native_coverage'] if x['role']==role) for role in ['train','validation','test']}
    controls=r['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    labels=[];entity_labels=[]
    with np.load(n['scores_path'],allow_pickle=False) as replay,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        assert set(replay.files)==set(saved.files)==set(n['arrays'])
        for key in replay.files:
            actual=replay[key];require_identical(actual,saved[key]);assert array_digest(actual)==n['arrays'][key]
        assert len(n['native_coverage'])==len(native.groups(job,data))
        for coverage,(role,group,key) in zip(n['native_coverage'],native.groups(job,data)):
            assert coverage['role']==role and coverage['source_array']==key
            values=replay[group['array_prefix']+'_'+role]
            assert len(values)==coverage['native_points'] and np.isfinite(values).all()
            if role=='test':
                y=replay[group['array_prefix']+'_labels'];assert y.ndim==1 and len(y)==len(values) and set(np.unique(y)).issubset({0,1})
                labels.append(y);entity_labels.append(y)
        y=np.concatenate(labels);assert len(y)==counts['test']
        positive=int(y.sum());negative=len(y)-positive
        constant=binary_counts(y,np.ones(len(y),dtype=bool))
        if data['id'] in ['swan','tep_classic','skab'] and job['window']==512:
            selected.append(cid)
            pair_refs[cid]=dict(input_sha256=binding['input_sha256'],dataset=data['id'],label_arrays_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                test_points=len(y),positive_points=positive,normal_points=negative,positive_prevalence=positive/len(y),
                all_alert_reference_micro_F1=constant['f1'],all_alert_reference_macro_F1=float(np.mean([binary_counts(a,np.ones(len(a),dtype=bool))['f1'] for a in entity_labels])),
                constant_score_AP_reference=positive/len(y),reference_alert_fraction=1.,reference_is_not_same_alert_budget=True)
    for ratio,v in controls.items():
        o=source['metrics']['strict'][ratio];require_same_metrics(v['micro'],o['micro'])
        assert v['threshold']==o['calibration']['threshold'] and v['macro_f1']==o['macro_f1']
        if cid in selected:
            ci=o['f1_ci95'];events=o['event_metrics']
            records.append(dict(canonical_id=cid,dataset=data['id'],size=Path(case['config']['parameters']['checkpoint_directory']).name,
                backbone=n['checkpoint_load_audit']['pretrained_size'],
                parameters=n['checkpoint_load_audit']['parameter_count'],context=512,seed=42,validation_upper_tail_percent=ratio,
                threshold=v['threshold'],micro_F1=v['micro']['f1'],macro_F1=v['macro_f1'],precision=v['micro']['precision'],recall=v['micro']['recall'],
                AUROC=v['micro']['auroc'],AP=v['micro']['average_precision'],same_threshold_PA_F1=v['same_validation_threshold_PA_diagnostic']['f1'],
                positive_prevalence=positive/counts['test'],test_normal_false_positive_rate=v['micro']['false_positives']/negative,
                always_alert_reference_F1=constant['f1'],F1_CI95_low=ci['low'],F1_CI95_high=ci['high'],bootstrap_unit=ci['unit'],
                bootstrap_resamples=ci['resamples'],bootstrap_seed=ci['seed'],bootstrap_block_points=ci['block_points'],
                event_recall=events['event_recall'],false_alarms_per1000_normal=events['false_alarms_per_1000_normal_points'],
                full_TAB_certified=False,training_seed_uncertainty=False))
    bindings[cid]=dict(source_binding=binding,configuration=config,native_path=str(np_path),native_sha256=sha(np_path),review_path=str(rp),review_sha256=sha(rp),
        score_archive_path=n['scores_path'],score_archive_sha256=n['scores_sha256'],all_array_payloads_independently_rechecked_now=True,
        native_role_points=counts,array_count=len(n['arrays']),group_role_count=len(n['native_coverage']),
        explicit_timestamp_or_row_arrays=[k for k in n['arrays'] if k.endswith(('_timestamps','_source_rows'))],
        full_window_MSE_check=n['inline_full_window_MSE_check'],fixed_MSE_tolerance=actual_runtime['mse_tolerance'],
        original_source_execution_guard_proven=binding['source_guard_history']['original_execution_guard_history_proven'],
        metadata_recovery=is_recovered,historical_paper_or_full_TAB_equivalence_certified=False)
    archive_sources[cid]=selected_files
assert len(bindings)==22 and len(selected)==6 and len(records)==18
for dataset in ['swan','tep_classic','skab']:
    pair=[x for x in pair_refs.values() if x['dataset']==dataset];assert len(pair)==2 and pair[0]==pair[1]
sources={}
for name in ['MOMENT_full19_industrial_native_closure_validation_20261016_v1.xml','MOMENT_full22_corrected_metadata_closure_validation_20261016_v1.xml']:
    file=ROOT/'tmp'/name;suites=list(ET.parse(file).getroot().iter('testsuite'))
    assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
    sources[name]=dict(path=file,tests=sum(int(x.get('tests',0)) for x in suites))
assert sum(x['tests'] for x in sources.values())==31
out=ROOT/'projects/flow_matching_research/results/2026-10-16/MOMENT_full22_union_and_three_industrial_matched_controls_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for cid,paths in archive_sources.items():
    folder=out/'cases'/cid;folder.mkdir(parents=True)
    for label,p in paths.items():shutil.copyfile(p,folder/(label+'.json'))
    if cid in selected or cid in expected_new:
        shutil.copyfile(ROOT/cases[cid]['job']['output_directory']/'result.json',folder/'source_result.json')
        shutil.copyfile(ROOT/cases[cid]['job']['model_config'],folder/'model_configuration.json')
    if cid in recovered:
        admission=correction.metadata_failure(cases[cid],recovered_runtime);write(folder/'preserved_metadata_rejection_binding.json',admission)
        gp=Path(admission['original_guard_path']);shutil.copyfile(gp,folder/'preserved_original_native_rejection_guard.json')
(out/'frozen_original78_collector_snapshot.json').write_bytes(state_bytes)
(out/'frozen_corrected39_collector_snapshot.json').write_bytes(recover_bytes)
write(out/'all22_complete_source_native_review_bindings.json',dict(records=bindings,previous_union19_packet=prior.relative_to(ROOT).as_posix(),
    previous_union19_packet_integrity_sha256=sha(prior/'packet_integrity.json'),previous_verified_ids=sorted(old),newly_closed_ids=sorted(expected_new),
    complete_unique_original_cases=22,required_original_cases=78,original_historical_failure_count=13,
    those_historical_failures_with_complete_isolated_recovery=len(recovered),original_failures_not_removed=True,
    all_original_raw_data_and_frozen_source_preserved=True,additional_training_updates=0,additional_neural_forwards_by_this_closure=0,
    real_existing_collectors=owners,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
    historical_pretraining_finetuning_full_TAB_and_complete_scope_remain_unproven=True,all_original_experiments_complete=False))
write(out/'all6_model_label_prevalence_and_unconstrained_alert_references.json',dict(records=pair_refs,
    references_are_diagnostics_not_registered_competing_methods=True,always_alert_uses100percent_test_alarm_budget=True,
    positive_prevalence_is_a_constant_score_AP_reference=True))
with (out/'all18_fixed_validation_metrics_and_PA_controls.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
ledger=[]
for cid,case in cases.items():
    rp=ROOT/case['job']['output_directory']/'result.json';source_complete=rp.exists() and read(rp).get('status')=='completed'
    ledger.append(dict(canonical_id=cid,dataset=case['job']['dataset'],context=case['job']['window'],family=case['family'],
        source_result_completed=source_complete,original_native_review_complete=cid in ids,original_failure_preserved=cid in state['failed_preserved'],
        isolated_metadata_recovery_complete=cid in recovered,union_full_native_and_review_complete=cid in complete,
        original_failure=state['failed_preserved'].get(cid,''),new_training_seed=False))
with (out/'all78_original_case_completion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
for name,value in sources.items():shutil.copyfile(value['path'],out/name)
for name,p in [('original78_runtime',ROOT/config_path),('corrected39_runtime',ROOT/correction.CONFIG)]:shutil.copyfile(p,out/(name+'.json'))
primary=[x for x in records if float(x['validation_upper_tail_percent'])==1]
table='\n'.join(f"| {x['dataset']} | {x['size']} | {x['micro_F1']:.6f} | {x['macro_F1']:.6f} | {x['same_threshold_PA_F1']:.6f} | {x['AUROC']:.6f} | {x['AP']:.6f} | {x['test_normal_false_positive_rate']:.4%} |" for x in primary)
(out/'README.md').write_text(f'''# MOMENT：22/78完整原生案例与三个工业数据集同数据对照

原78控制器19项通过，加独立纠正元数据轨道3项，按原始canonical ID去重为22/78。
相对此前已发表19项快照，新增Base/context512在TEP classic、SKAB，以及Small/context100在本地26实体MSL的完整重放与独立复核三项；SWAN比较为已完成的复用对照，不重复计数。
逐项重验22个实际源结果、权重/输入/代码哈希、原生与独立证明、全部两个阶段守卫退出0且无资源中止，并重新比较每个完整数组的dtype/shape/内容及其SHA256。
新包提供全部78案例完成矩阵。原13项失败保持，3项已在独立恢复路径补齐，原失败记录不删除或覆盖。

三个工业数据集的Small35,341,512参数与Base109,641,608参数分别使用完全相同的输入及标签哈希、context512、batch1、CPU两线程、seed42和冻结padding。
SWAN有15637/3935/72000训练/验证/测试点，每个模型180次窗口forward；TEP classic为500/960/20160点、45次；SKAB为9405/745/36656点、116次。
所有角色与尾部保持；数组中实际存在的标签/时间/行号均被核对，未额外生成缺失时间戳。旧Small源执行guard缺口仍为缺口，当前两阶段成功不能补写历史guard。

固定验证集1%上尾阈值，无按测试标签选阈值：

| 数据集 | 官方权重 | 无PA micro F1 | 无PA entity macro F1 | 同阈值PA F1 | AUROC | AP | 测试正常点误报率 |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

TEP测试的异常点为16800/20160，即83.3333%；常量分数AP参考值为0.833333，100%全点告警的micro F1参考值为0.909091。
Base无PA F1约0.594258、同分数同阈值PA后约0.964353，AUROC约0.566316。高PA F1、异常阳性率与整体排序质量必须一起解释。
全点告警参考使用100%告警预算，只是标签比例诊断，不能充当同预算竞争方法的成绩。验证1%分位数也不保证迁移测试正常点的误报率等于1%；实际数值已完整列出。
三个验证分位数0.1/1/5全部保留；每个原200次bootstrap、事件指标及10个pooled诊断比例已由原独立进程复算，本次保留其全部证明和哈希。
区间描述固定模型与阈值下的实体/时间块不确定性，不是训练种子区间。相同输入对照也不提供完整模型规模显著性或跨数据集优越性结论。

恢复轨道只纠正官方配置seq_len512与外部100窗口被错误要求相等的前推理元数据断言；模型数值主体、窗口100、原额外patch和原全部资源门槛不变。
新增MSL恢复仍是原26实体/71418测试点的本地轨道，不等同新增27实体轨道。所有历史失败、guard缺口及原始数据保持。
逐窗口FP64 MSE检查保留原rtol=atol=2e-6，最大绝对差与原分数尺度单列，接受相对检查不表示绝对差近零；未裁剪、翻转评分或改变容差。

31项原生及纠正元数据检查通过。此归档追加训练更新0、神经网络前向0；实际完整零样本推理来自已有控制器，固定官方权重不能充作新增训练种子。
官方当前发布权重zero-shot评估与历史论文微调、Tiny/ICM全PILE预训练、完整TAB训练策略分别验收。
全量预处理值独立重建、底层VUS/Affiliation积分及历史论文等价不由这个包认证；其他54/27/28实体新增轨道、全部45论文/681方法/273数据任务/702标量目标继续保留，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'closure_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_full22_union_industrial_controls.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),complete_unique_cases=22,required_original_cases=78,
    newly_closed_vs_previous19=sorted(expected_new),new_training_seeds=0,all_original_experiments_complete=False))
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
files=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
write(ROOT/'tmp/MOMENT_full22_union_industrial_controls_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(out),files=len(files),full_unique_cases=22,newly_closed_ids=sorted(expected_new),primary=primary)))
