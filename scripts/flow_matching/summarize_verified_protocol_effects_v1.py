"""Evidence-bound protocol contrasts from four closed AD cases;no new inference.

Within-case fixed-score comparisons are not an inter-model leaderboard. Both
Pi cases are branches of one family/seed; MOMENT evidence is saved-array-only.
"""
from datetime import datetime,timezone
import argparse
import json
import math
from pathlib import Path
import subprocess

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT,read,sha,write

CONFIG='configs/reproducibility/fm_verified_protocol_effects.2026-10-15_v1.json'
PI_SOURCE='projects/flow_matching_research/results/2026-10-15/pi_source1107_actual_complete_model_replay_closure_v1'
PI_CAUSAL='projects/flow_matching_research/results/2026-10-15/pi_causal1107_actual_complete_model_replay_closure_v1'
MOMENT='projects/flow_matching_research/results/2026-10-15/moment_base_release512_SMAP_constant_fit_full_saved_scores_v1'
SB='projects/flow_matching_research/results/2026-10-14/local_SB_SWaT_full_checkpoint_scores_v1'


def configuration():
    packets=dict(pi_source=PI_SOURCE,pi_causal=PI_CAUSAL,moment=MOMENT,sb=SB)
    files=[]
    for kind,folder in packets.items():
        names=(['archive_validation.json','one_complete_original_model_and_all_fifteen_streams_closure.json',
                'independent_full_stream_proof.json','completed_native_report.json','native_guarded_receipt.json','independent_guarded_receipt.json']
               if kind.startswith('pi_') else ['closure_validation.json','independent_audit_guard.json','original_source_guard.json','original_result.json'])
        if kind=='moment':names+=['same_scores_protocol_effect_and_complete_metrics.json','verification/independent_full_saved_score_verification.json']
        if kind=='sb':names+=['complete_metrics_and_score_shift_diagnostics.json','verification/independent_full_model_score_verification.json']
        files.extend(folder+'/'+name for name in names)
    code=['scripts/flow_matching/summarize_verified_protocol_effects_v1.py','tests/test_fm_verified_protocol_effects_v1.py']
    return dict(schema_version=1,packets=packets,evidence=[dict(path=p,sha256=sha(ROOT/p)) for p in files],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in code],
        metric_tolerance=2e-12,scope='four_closed_cases_three_method_families_three_datasets_no_cross_model_ranking',
        primary_contrast='same_scores_same_threshold_point_adjustment_only',
        sb_secondary_contrast='test_label_selected_frozen_ten_ratio_grid_diagnostic_only',
        output_directory='projects/flow_matching_research/results/2026-10-15/verified_AD_protocol_effects_v1',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        boundary='Metadata analysis of previously closed native/independent evidence,not new model inference, '
            'training,threshold optimization,raw-array audit,original-paper certification or a leaderboard. '
            'All model/dataset/seed/context/calibration distinctions retained. No aggregate across datasets '
            'or pseudo-independent Pi branches,confidence intervals or p-values are introduced.')


def verify(config,root=ROOT):
    if (config['packets']!=dict(pi_source=PI_SOURCE,pi_causal=PI_CAUSAL,moment=MOMENT,sb=SB)
        or config['metric_tolerance']!=2e-12
        or config['primary_contrast']!='same_scores_same_threshold_point_adjustment_only'
        or config['scope']!='four_closed_cases_three_method_families_three_datasets_no_cross_model_ranking'):
        raise ValueError('Retain actual case selection,within-case contrast and evidence scope')
    for record in config['evidence']+config['source_receipts']:
        if sha(root/record['path'])!=record['sha256']:raise ValueError('Closed source evidence/code changed: '+record['path'])


def guard_ok(guard):
    if guard['exit_code']!=0 or guard['resource_guard_aborted']:
        raise ValueError('A completed passed actual review guard is required')


def counts(value,total):
    def take(long,short):return value[long] if long in value else value[short]
    tp,fp,fn=[float(take(a,b)) for a,b in [('true_positives','tp'),('false_positives','fp'),('false_negatives','fn')]]
    tn=float(value.get('tn',total-tp-fp-fn))
    if any(v<0 or not v.is_integer() for v in [tp,fp,fn,tn]) or tp+fp+fn+tn!=total:
        raise ValueError('Exact nonnegative same-support confusion counts required')
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))


def f1_from_counts(c):
    denominator=2*c['tp']+c['fp']+c['fn']
    return 2*c['tp']/denominator if denominator else 0.


def fixed_threshold_effect(before,after,total,before_f1,after_f1):
    a,b=counts(before,total),counts(after,total)
    if (a['tp']+a['fn']!=b['tp']+b['fn'] or a['fp']!=b['fp'] or a['tn']!=b['tn']
        or b['tp']<a['tp']):
        raise ValueError('This is not a PA-only contrast on unchanged truth/support/normal predictions')
    if abs(f1_from_counts(a)-before_f1)>2e-12 or abs(f1_from_counts(b)-after_f1)>2e-12:
        raise ValueError('Reported F1 does not match its complete confusion counts')
    return dict(no_PA_f1=before_f1,PA_f1=after_f1,F1_increase_percentage_points=100*(after_f1-before_f1),
        no_PA_confusion=a,PA_confusion=b,newly_credited_anomaly_points=b['tp']-a['tp'],
        fixed_score_arrays=True,fixed_threshold=True,fixed_test_support=True,
        ground_truth_anomaly_events_used_by_PA=True,model_changed=False)


def pi_case(config,kind,root):
    folder=root/config['packets'][kind]
    closure=read(folder/'one_complete_original_model_and_all_fifteen_streams_closure.json')
    independent=read(folder/'independent_full_stream_proof.json');native=read(folder/'completed_native_report.json')
    seal=read(folder/'archive_validation.json')['artifact_files_sha256']
    for name in ['one_complete_original_model_and_all_fifteen_streams_closure.json','independent_full_stream_proof.json',
                 'completed_native_report.json','native_guarded_receipt.json','independent_guarded_receipt.json']:
        if sha(folder/name)!=seal[name]:raise ValueError('Pi sealed full-model evidence changed')
    ng=read(folder/'native_guarded_receipt.json');ig=read(folder/'independent_guarded_receipt.json');guard_ok(ng);guard_ok(ig)
    if (not closure['full_native_checkpoint_reinference_and_independent_stream_metric_review_complete']
        or not closure['all_streams_bitwise_identical_to_original_saved_arrays'] or not independent['passed']
        or not native['full_native_checkpoint_reinference_passed']
        or independent['completed_native_replay_report_sha256']!=sha(folder/'completed_native_report.json')
        or independent['canonical_id']!=closure['canonical_id'] or closure['independently_recomputed_metrics']!=independent['independently_recomputed_metrics']
        or not ng['full_native_checkpoint_reinference_passed'] or not ig['passed']):
        raise ValueError('Complete canonical Pi native+independent evidence required')
    metric=closure['independently_recomputed_metrics'];plain=metric['pointwise_no_PA_same_author_threshold']
    effect=fixed_threshold_effect(closure['confusion_no_PA'],closure['confusion_author_PA'],closure['scored_original_test_points'],plain['f1'],metric['f1'])
    return dict(case_id=closure['canonical_id'],label='Pi source / PSM' if kind=='pi_source' else 'Pi time-axis control / PSM',
        family='Pi-Transformer',dataset='PSM',seed=independent['seed'],test_points=closure['scored_original_test_points'],
        omitted_test_tail_points=closure['original_dropped_tail_points'],parameters=closure['full_model_parameters'],
        threshold=closure['source_threshold'],calibration='source pooled train/test scores; source test checkpoint selection retained',
        native_model_reinference_proven=True,independent_saved_stream_and_metric_review_proven=True,
        original_training_guard_proven=closure['original_training_guard_verified'],historical_paper_equivalence=False,strict_TAB=False,
        AUROC=plain['auroc'],average_precision=plain['average_precision'],same_threshold_effect=effect,
        evidence_packet=config['packets'][kind],between_branch_independent_study_claim=False,fully_causal_deployment_certified=False)


def moment_case(config,root):
    folder=root/config['packets']['moment'];s=read(folder/'same_scores_protocol_effect_and_complete_metrics.json')
    proof=read(folder/'verification/independent_full_saved_score_verification.json');closure=read(folder/'closure_validation.json')
    guard=read(folder/'independent_audit_guard.json');guard_ok(guard)
    original_guard=read(folder/'original_source_guard.json');guard_ok(original_guard)
    if (not closure['passed'] or not proof['passed'] or not guard['full_saved_score_review_passed']
        or closure['native_proof_sha256']!=sha(folder/'verification/independent_full_saved_score_verification.json')
        or closure['native_audit_guard_sha256']!=sha(folder/'independent_audit_guard.json')
        or proof['model_reinference_performed'] or s['model_reinference_in_this_review']
        or proof['full_saved_score_protocol_controls']!=s['same_scores_protocol_controls']
        or proof['primary_metrics']!=s['primary_validation_only_no_PA']
        or proof['original_result_sha256']!=sha(folder/'original_result.json')):
        raise ValueError('MOMENT full saved-score-only proof must retain its actual evidence strength')
    control=s['same_scores_protocol_controls'];a=control['validation_only_no_PA'];b=control['validation_only_same_threshold_with_PA']
    if not control['all_controls_use_identical_frozen_score_arrays'] or control['primary_threshold_selected_using_test_features_or_labels']:
        raise ValueError('Actual unchanged-score validation-threshold PA control required')
    effect=fixed_threshold_effect(a,b,s['native_test_points'],a['f1'],b['f1']);primary=s['primary_validation_only_no_PA']['micro']
    return dict(case_id=s['canonical_id'],label='MOMENT base / SMAP',family='MOMENT',dataset='SMAP',seed=read(folder/'original_result.json')['seed'],
        test_points=s['native_test_points'],parameters=s['model_parameters'],threshold=control['validation_threshold'],
        calibration='validation-only percentile; constant-fit processed data; 512 context',
        native_model_reinference_proven=False,independent_saved_stream_and_metric_review_proven=True,
        original_execution_guard_proven=bool(original_guard['full_registered_budget_retained']),
        historical_paper_equivalence=False,strict_TAB=False,AUROC=primary['auroc'],average_precision=primary['average_precision'],
        same_threshold_effect=effect,evidence_packet=config['packets']['moment'],
        test_pooled_ratio1_control=control['test_pool_calibration_ratio1'],
        best_of_test_label_selected_grid_PA_f1=control['best_test_label_selected_PA_grid_F1'])


def sb_case(config,root):
    folder=root/config['packets']['sb'];s=read(folder/'complete_metrics_and_score_shift_diagnostics.json')
    proof=read(folder/'verification/independent_full_model_score_verification.json');closure=read(folder/'closure_validation.json')
    guard=read(folder/'independent_audit_guard.json');guard_ok(guard)
    original_guard=read(folder/'original_source_guard.json');guard_ok(original_guard)
    if (not closure['passed'] or not proof['passed'] or not guard['full_cold_model_replay_passed']
        or closure['proof_sha256']!=sha(folder/'verification/independent_full_model_score_verification.json')
        or closure['audit_guard_sha256']!=sha(folder/'independent_audit_guard.json')
        or not proof['original_train_validation_test_native_scores_reinferred_bitwise']
        or not proof['complete_checkpoint_tensors_restored_bitwise']
        or proof['source_result_sha256']!=sha(folder/'original_result.json')
        or proof['tab_core_diagnostic']!=s['original_full_core_diagnostic']
        or proof['primary_metrics']!=s['primary_validation_only_no_PA']):raise ValueError('Complete frozen SB native evidence required')
    grid=s['original_full_core_diagnostic']['ratio_grid'];ratio1=next(g for g in grid if g['ratio_percent']==1)
    primary=s['primary_validation_only_no_PA']['micro'];c=counts(primary,s['native_test_points'])
    false_alarm=c['fp']/(c['fp']+c['tn'])
    if abs(false_alarm-s['measured_test_normal_false_alarm_fraction'])>2e-12:raise ValueError('Recorded false-alarm shift differs from full confusion counts')
    effect=dict(no_PA_f1=ratio1['f_score'],PA_f1=ratio1['adjust_f_score'],F1_increase_percentage_points=100*(ratio1['adjust_f_score']-ratio1['f_score']),
        fixed_score_arrays=True,fixed_threshold=True,fixed_test_support=True,ground_truth_anomaly_events_used_by_PA=True,model_changed=False,
        confusion_counts_not_archived_in_this_grid_record=True)
    return dict(case_id=s['canonical_id'],label='Local SB / SWaT (pooled 1%)',family='local SB objective adapter',dataset='SWaT',seed=1103,
        test_points=s['native_test_points'],parameters=s['parameter_count'],threshold=ratio1['threshold'],
        calibration='same source pooled train/test ratio1 threshold for the PA-only contrast',
        native_model_reinference_proven=True,independent_saved_stream_and_metric_review_proven=True,
        original_execution_guard_proven=bool(original_guard['full_registered_budget_retained']),
        historical_paper_equivalence=False,strict_TAB=False,AUROC=primary['auroc'],average_precision=primary['average_precision'],
        same_threshold_effect=effect,evidence_packet=config['packets']['sb'],
        primary_validation_only_no_PA_f1=primary['f1'],primary_validation_threshold=s['frozen_validation_threshold'],
        primary_test_normal_false_alarm_fraction=false_alarm,primary_validation_target_false_alarm_percent=1,
        diagnostic_test_pooled_grid=grid,best_test_label_selected_plain_f1=max(g['f_score'] for g in grid),
        best_test_label_selected_PA_f1=max(g['adjust_f_score'] for g in grid),
        model_changed_between_grid_controls=False,source_only_single_seed_not_complete_five_seed_cohort=True)


def build_report(config,root=ROOT):
    verify(config,root)
    cases=[pi_case(config,'pi_source',root),pi_case(config,'pi_causal',root),moment_case(config,root),sb_case(config,root)]
    sb=cases[-1]
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),scope=config['scope'],case_count=4,method_families=3,datasets=3,cases=cases,
        comparisons_are_within_case_not_cross_model_ranking=True,independent_study_count_not_inferred_from_four_rows=True,
        original_paper_target_scores_verified_by_this_report=False,new_model_inference=0,new_training_updates=0,
        raw_score_or_checkpoint_hashes_recomputed_by_this_report=False,closed_archival_proof_hashes_rechecked=True,
        new_seed_aggregate_confidence_intervals_or_pvalues=False,source_evidence=config['evidence'],
        observations_zh=[
            dict(type='measured',text='两个Pi/PSM分支在固定分数、固定阈值和固定测试点下，仅加入PA就将F1从约2%提高至约98%；AUROC仍约0.49。两个分支来自同一家族和同一随机种子，不是两项独立研究。'),
            dict(type='measured',text='MOMENT/SMAP同一验证阈值下的PA对照也明显提高F1；该个案只有完整保存分数审计，本报告不把它升级为新检查点推理证据。'),
            dict(type='measured',text='本地SB/SWaT的AUROC约0.821、AP约0.724，但验证阈值下F1约0.289，测试正常点误报率约54.93%。'),
            dict(type='inference',text='SB个案提示校准阈值向测试分布迁移时存在失配；仅凭这些记录，不能把原因唯一归于网络容量、训练不足或某一预处理步骤。'),
            dict(type='measured',text='SB的同一模型在固定十点测试标签选优网格中，未调整F1可达约0.738、PA后约0.853。这是阈值/评估协议对照，不是模型训练改进。'),
            dict(type='limitation',text='这些是三个方法家族、三个不同数据集上的四个个案，不构成跨算法排行榜，也不能据此否定其余论文或认定所有高分均出于同一种原因。')],
        sb_distinct_operations=dict(PA_only_at_pooled_ratio1_percentage_points=sb['same_threshold_effect']['F1_increase_percentage_points'],
            changing_from_validation_primary_to_best_test_grid_noPA_percentage_points=100*(sb['best_test_label_selected_plain_f1']-sb['primary_validation_only_no_PA_f1']),
            latter_is_diagnostic_multiple_protocol_changes_not_isolated_PA=True),
        fair_comparison_requirements_zh=['统一分组与完整时间点支持，保留尾部和预处理差异说明','阈值仅由冻结的验证方案确定；PA和测试标签选优单列诊断',
            '主指标同时报告未调整F1、AUROC/AP与误报/事件结果','完整随机种子组及其不确定性；不将重放、分支或预训练重评当作新训练种子',
            '完整检查点/保存数组证据强度分别标注；原论文等价性和严格TAB复现另行证明'],
        original_scope=config['original_scope'],all_original_experiments_complete=False)


def render(report,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'verified_protocol_effects_v1'})
    fig,(ax,curve)=plt.subplots(1,2,figsize=(13.6,5.1),gridspec_kw={'width_ratios':[1.15,1]})
    y=np.arange(4);cases=report['cases'];h=.32
    ax.barh(y-h/2,[100*r['same_threshold_effect']['no_PA_f1'] for r in cases],height=h,label='No point adjustment',color='#176482')
    ax.barh(y+h/2,[100*r['same_threshold_effect']['PA_f1'] for r in cases],height=h,label='PA (uses test event labels)',color='#e2983a')
    ax.set(yticks=y,yticklabels=[r['label'] for r in cases],xlim=(0,110),xlabel='F1 (%)',title='Same model, scores and threshold within each row')
    ax.set_ylim(4.05,-.65);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for i,r in enumerate(cases):
        for offset,key in [(-h/2,'no_PA_f1'),(h/2,'PA_f1')]:
            value=100*r['same_threshold_effect'][key];ax.text(value+.7,i+offset,f'{value:.2f}',va='center',fontsize=9)
    ax.legend(loc='lower right',fontsize=8)
    sb=cases[-1];grid=sb['diagnostic_test_pooled_grid'];x=[r['ratio_percent'] for r in grid]
    curve.plot(x,[100*r['f_score'] for r in grid],'o-',color='#176482',label='No PA: test-pooled ratio grid')
    curve.plot(x,[100*r['adjust_f_score'] for r in grid],'o-',color='#e2983a',label='PA: same ratio grid')
    curve.axhline(100*sb['primary_validation_only_no_PA_f1'],ls='--',color='#555555',label='Frozen validation-only primary (no PA)')
    curve.set(xscale='log',ylim=(0,100),xlabel='Predeclared test-pooled ratio (%)',ylabel='F1 (%)',title='One unchanged local SB model / SWaT')
    curve.grid(alpha=.18);curve.legend(loc='upper right',fontsize=8)
    fig.suptitle('Evaluation-protocol effects in four verified cases — not an algorithm ranking',fontsize=13)
    fig.text(.5,.025,'Pi: native + independent replay. MOMENT: saved-score audit. SB: native replay. Grid/test-label selection is diagnostic.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.185,right=.985,top=.82,bottom=.18,wspace=.32)
    for ext in ['png','svg']:fig.savefig(folder/('protocol_effects.'+ext),dpi=170,metadata={'Creator':'IIA benchmark protocol evidence report'})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--config',default=CONFIG)
    args=p.parse_args();path=ROOT/args.config
    if args.register:
        if path.exists():raise FileExistsError('Preserve frozen protocol evidence configuration')
        c=configuration();verify(c)
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(c,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps(dict(configuration=args.config,evidence_files=len(c['evidence']))));return
    c=read(path);report=build_report(c)
    # Bind archived source bytes to the repository; no weights or scores are reread.
    for record in c['evidence']:
        if subprocess.check_output(['git','show','HEAD:'+record['path']],cwd=ROOT)!=(ROOT/record['path']).read_bytes():
            raise ValueError('Evidence differs from committed repository content')
    report['source_git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    folder=ROOT/c['output_directory'];folder.mkdir(exist_ok=False)
    write(folder/'report.json',report);render(report,folder)
    write(folder/'run_receipt.json',dict(passed=True,configuration=args.config,configuration_sha256=sha(path),
        generated_files_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        raw_arrays_reread=False,new_training_updates=0,new_model_inference=0,all_original_experiments_complete=False))
    print(json.dumps(dict(passed=True,cases=4,method_families=3,datasets=3,output=str(folder),new_benchmark_runs=0)))


if __name__=='__main__':main()
