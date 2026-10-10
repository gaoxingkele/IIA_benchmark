"""Publish verified complete GiFlow arrays and all pending native TimeGAN slots."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.timegan_native_common_v1 import verify_queue, full_budget

ROOT, legacy = previous.ROOT, previous.legacy
METRICS = ('native_mean_batch_mae', 'native_mean_batch_mse',
           'native_mean_batch_mape_percent', 'sqrt_native_mean_batch_mse')


def corrected_updates(captured, audit, source):
    if (not audit['full_test_arrays_and_metrics_passed'] or audit['track'] != 'reviewed_corrected'
            or audit['job_id'] != 'giflow_native__reviewed_corrected__air36__main__seed393'
            or audit['seed'] != 393 or audit['actual_full_training_epochs'] != 68
            or audit['actual_optimizer_updates'] != 3264 or audit['full_test_batches'] != 14
            or audit['masked_targets_with_window_repetitions'] != 268613
            or audit['test_used_in_validation'] or audit['author_equivalence_certified']
            or audit['selection']['tested_weights_match_validation_checkpoint'] is not True):
        raise ValueError('Complete reviewed original seed/data/budget and array audit required')
    selected = [r for r in captured if r['algorithm'] == 'GiFlow/main' and r['dataset'] == 'air36'
                and r['protocol'] == 'giflow_native/reviewed_corrected']
    if len(selected) != 4 or {r['metric'] for r in selected} != set(METRICS):
        raise ValueError('All four original reviewed metric slots required')
    for row in selected:
        if (int(row['n']) != 1 or int(row['required_repeats']) != 5
                or not math.isclose(float(row['mean']), audit['native_metrics'][row['metric']], rel_tol=1e-12)
                or not math.isclose(float(row['mean']), audit['independently_recomputed_metrics'][row['metric']],
                                    rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError('Original native full score and all five-seed slots must be retained')
        row.update(source=source, comparison='complete_native_seed_1_of_5_full_arrays_audited_model_replay_in_flight',
                   selection_boundary='validation-selected tested checkpoint; persistent EMA; no test-in-validation',
                   strict_TAB_result=False)
    return selected


def pending_timegan(model, queue, stamp, headers, source):
    full_budget(model)
    slots = {(j['dataset'], j['module']) for j in queue['jobs']}
    if slots != {(d,m) for d in ('stock','energy','sine') for m in ('gru','lstm','lstmLN')}:
        raise ValueError('Every original TimeGAN dataset/source module required')
    rows = []
    for dataset, module in sorted(slots):
        members = [j for j in queue['jobs'] if (j['dataset'],j['module']) == (dataset,module)]
        if len(members) != 5 or {j['seed'] for j in members} != set(range(1103,1108)):
            raise ValueError('Complete five native generator seed slots required')
        for metric in ('discriminative','predictive'):
            row = dict.fromkeys(headers,'')
            row.update(algorithm='TimeGAN-native/'+module,dataset=dataset,task='unconditional_time_series_generation',
                protocol='TimeGAN-author-native/source-defaults',metric=metric,n=0,required_repeats=5,
                captured_utc=stamp,source=source,author_equivalence_certified=False,strict_TAB_result=False,
                comparison='pending_full_original_source_training_and_evaluators',evaluator_repeats=10,
                metric_unit='native_abs_accuracy_minus_0.5' if metric=='discriminative' else 'native_normalized_feature_MAE',
                selection_boundary='original full-data normalization; random overlapping-window posthoc splits; historical optimized settings unknown',
                uncertainty_scope='independent_generator_seeds; ten evaluator trainings per generator are not extra generator seeds')
            rows.append(row)
    legacy.validate_rows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=legacy.read(ROOT/args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen audited metric publication evidence changed')
    base=ROOT/config['base_publication'];old=previous.verified_capture(base,'publication_validation.json')
    capture=ROOT/config['giflow_capture'];gf=previous.verified_capture(capture,'validation.json')
    if gf['canonical_slots']!=140 or gf['job_counts']['completed']!=2 or gf['seed_groups']!=28:
        raise ValueError('All original GiFlow canonical slots and complete two native trained seeds required')
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);headers,original=reader.fieldnames,list(reader)
    with (capture/'algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        captured=list(csv.DictReader(f))
    proof=legacy.read(ROOT/config['corrected_full_array_proof'])
    updates=corrected_updates(captured,proof,config['corrected_full_array_proof'])
    merged=previous.merge_audited_rows(original,updates)
    keys={previous.key(r) for r in updates}
    if any(a!=b for a,b in zip(original,merged) if previous.key(a) not in keys):
        raise ValueError('Other results or their capture timestamps changed')
    if sum(a!=b for a,b in zip(original,merged))!=4:
        raise ValueError('Only four verified original reviewed metrics may change')
    queue=legacy.read(ROOT/config['timegan_queue']);model=verify_queue(queue)
    for job in queue['jobs']:
        if (ROOT/job['output_directory']/'result.json').exists():
            raise ValueError('New complete native TimeGAN result requires a fresh full audit, not a pending row')
    stamp=datetime.now(timezone.utc).isoformat()
    added=pending_timegan(model,queue,stamp,headers,config['timegan_queue'])
    rows=merged+added;legacy.validate_rows(rows)
    measured=sum(r['mean'] not in ('',None) for r in rows)
    if len(rows)!=old['aggregate_metric_rows']+18 or measured!=old['measured_metric_rows']+4:
        raise ValueError('Exactly four newly measured and eighteen pending slots required')
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Metric publications are immutable')
    target.mkdir(parents=True)
    with (target/'all_algorithm_dataset_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    for name in ('overview.md','strict_algorithm_dataset.md','original_generation_jobs.csv'):
        shutil.copyfile(base/name,target/name)
    columns=['任务','算法／消融','数据集','协议','轮数','掩码／缺失率','评分配方','指标','完成／预定',
             '均值','STD','SE','95%下界','95%上界','论文值','采集UTC']
    values=[[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('epochs',''),
             str(r.get('pattern',''))+'/'+str(r.get('missing_ratio','')),r.get('score_recipe',''),r['metric'],
             f"{r['n']}/{r['required_repeats']}",*[legacy.fmt(r.get(k)) for k in
             ('mean','std','se','ci95_low','ci95_high','paper_mean')],r['captured_utc']] for r in rows]
    (target/'all_metrics.md').write_text('# 完整算法—数据集技术指标\n\n'
        '—为缺结果，0为实测零。不同协议分列；各轨道保留自己的采集时间。'
        '本版只新增GiFlow修正分支四项已完成模型指标和TimeGAN原始完整队列18项待完成指标；'
        '完整清单不表示全部原始实验完成。\n\n'+legacy.table(columns,values)+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    counts=Counter(r['task'] for r in rows);ready=Counter(r['task'] for r in rows if r['mean'] not in ('',None))
    summary=[[task,count,ready[task],count-ready[task]] for task,count in sorted(counts.items())]
    metrics=[[r['metric'],legacy.fmt(r['mean']),f"{r['n']}/{r['required_repeats']}"] for r in updates]
    (target/'README.md').write_text('# 完整算法—数据集实验技术指标\n\n'
        f'发布UTC：{stamp}。共{len(rows)}项指标、{measured}项有实测值、{len(rows)-measured}项缺完整结果，'
        f'覆盖{len({r["algorithm"] for r in rows})}个算法／配置名称。指标条数不是训练次数；'
        '原方向45论文/681方法基线消融/273原始任务仍未全部完成。\n\n'
        '[可筛选全部结果](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；'
        '[全部文字指标](all_metrics.md)；[异常检测F1矩阵](overview.md)；'
        '[异常检测全部指标](strict_algorithm_dataset.md)。\n\n'+
        legacy.table(['任务','登记指标','有实测值','缺完整结果'],summary)+'\n\n'
        'GiFlow／Air36／reviewed_corrected／main／20%point／seed393完成原生早停68轮、3264次优化；'
        '14批/1747窗口/268613目标全部预测数组和原生批均值指标独立重算通过。仍为1/5训练种子，'
        '不制造STD/SE/95%区间，原完整权重CPU重推断仍在运行，未称完成。'
        '[完整数组证据](../giflow_corrected_full_arrays_v1/full_array_audit.json)；'
        '[原生完整权重重推断登记](../giflow_corrected_execution_v1/launch.json)。\n\n'+
        legacy.table(['指标','原生完整训练值','完成／预定'],metrics)+'\n\n'
        '发布源码分支同种子MAE9.654690，修正分支10.951781；修正分支使用实际验证集、持续EMA和验证选中权重。'
        '多处修正及权重选择同时改变，单个配对种子不能归因，不能按低分丢弃结果或声称论文一致。\n\n'
        '新增TimeGAN-native的三个原始数据集×三个原始RNN选项×两个原始指标=18待完成聚合槽；'
        '每槽5个完整生成器种子，每生成器各10次判别/预测评估训练。这些评估器重复不另计生成器种子。'
        '45任务仍保留每阶段50000循环、完整batch128/seq24/hidden24/layers3和原始2000/5000评估更新。'
        '[原生注册/构图/全量数据证据](../timegan_native_preparation_v1/README.rst)。暂无TimeGAN完整成绩；'
        '原始源码默认参数为明确本地选择，历史优化参数和采用它的其他论文不规则数据任务仍待完成。\n\n'
        '其他全部轨道结果和采集时间与v17逐行保留；没有重新采集异常检测、SAITS、CFM等轨道。'
        '验证集阈值无PA、测试标签选阈值、PA、作者原生协议分别列出；尚未获完整TAB预算或作者等价认证。\n',encoding='utf-8')
    receipt=copy.deepcopy(old)
    receipt.update(published_utc=stamp,base_publication_utc=old['published_utc'],aggregate_metric_rows=len(rows),
        measured_metric_rows=measured,missing_metric_rows=len(rows)-measured,task_rows=dict(counts),measured_task_rows=dict(ready),
        algorithm_configuration_names=len({r['algorithm'] for r in rows}),replaced_metric_rows=4,changed_numeric_metric_rows=4,
        newly_measured_metric_rows=4,added_pending_metric_rows=18,timegan_full_original_registered_jobs=45,
        timegan_complete_training_seed_slots=0,giflow_complete_slots=2,giflow_full_native_models_replayed=1,
        giflow_release_full_checkpoint_replay_complete=True,giflow_corrected_full_checkpoint_replay_complete=False,
        giflow_independent_model_reinference_complete=False,other_track_rows_and_capture_times_preserved=True,
        all_original_experiments_complete=False)
    receipt['source_receipts']+=config['source_receipts']+[dict(path=args.config,sha256=legacy.sha(ROOT/args.config))]
    receipt['outputs']={p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',receipt)
    print(dict(rows=len(rows),measured=measured,missing=len(rows)-measured,newly_measured=4,pending_timegan=18))


if __name__=='__main__':main()
