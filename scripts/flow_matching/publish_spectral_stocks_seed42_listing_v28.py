"""Publish only four audited native Stocks baseline metrics, one trained seed."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT,legacy=previous.ROOT,previous.legacy
METRICS=('context_fid','correlational','discriminative','predictive')
CANONICAL='spectral_original__diffusion_ts__stocks__released_author_data__seed42'


def validated_native(proof,result,receipt):
    required=dict(passed=True,canonical_id=CANONICAL,seed=42,
        original_full_registered_generation_jobs=70,actual_full_native_model_count=1,
        planned_original_training_seed_count=5,actual_model_parameters=294390,
        paper_reported_parameters=291318,original_optimizer_updates=10000,
        original_gradient_accumulation=2,original_training_minibatches=20000,
        complete_reference_shape=[3662,24,6],complete_generated_shape=[4002,24,6],evaluated_generated_rows=3662,
        actual_full_original_checkpoint_tensors_restored_bitwise=True,
        complete_original_reference_array_bitwise_verified=True,all_original_generated_arrays_finite=True,
        all_four_five_repeat_evaluator_statistics_independently_recomputed=True,
        learned_evaluators_independently_retrained=False,generative_model_reinferred_by_this_audit=False,
        additional_training_seed_slots=0,author_equivalence_certified=False)
    if any(proof.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete actual original trained model/data/evaluator review required')
    if (result['job']['id']!=CANONICAL or result['status']!='completed' or result['benchmark_smoke']
            or receipt['id']!=CANONICAL or receipt['exitcode']!=0 or not receipt['full_result_validated']
            or receipt['resource_guard_aborted'] or set(proof['metrics'])!=set(METRICS)):
        raise ValueError('Actual successful complete original model result and guard receipt required')
    for key,item in proof['metrics'].items():
        if (item['evaluator_repeats']!=5 or item['independent_training_models']!=1
                or item['planned_original_training_models']!=5 or item['between_training_seed_STD'] is not None
                or item['between_training_seed_CI95'] is not None
                or item['raw_repeats']!=result['metrics'][key]['raw_repeats']):
            raise ValueError('Evaluator repeats are not independent generator training seeds')


def updates(original,proof,result,source):
    rows=[]
    for metric in METRICS:
        matches=[r for r in original if r['task']=='unconditional_time_series_generation'
            and r['algorithm']=='Spectral-author/diffusion_ts' and r['dataset']=='stocks'
            and r['protocol']=='released_author_data' and r['metric']==metric]
        if len(matches)!=1 or int(matches[0]['n'])!=0 or int(matches[0]['required_repeats'])!=5 or matches[0]['mean']!='':
            raise ValueError('Exactly the four original empty five-training-seed slots required')
        row=dict(matches[0])
        evaluation=result['metrics'][metric]
        row.update(n=1,mean=evaluation['mean'],std='',se='',ci95_low='',ci95_high='',
            evaluator_repeats=5,evaluator_std=evaluation['std'],evaluator_se=evaluation['se'],
            evaluator_ci95='%s / %s'%(evaluation['ci95_low'],evaluation['ci95_high']),
            uncertainty_scope='one_of_five_generator_training_seeds; five evaluator repeats for this model only',
            source=source,captured_utc=proof['captured_utc'],strict_TAB_result=False,
            author_equivalence_certified=False,
            comparison='complete_native_generator_1_of_5_arrays_checkpoint_and_evaluator_statistics_audited',
            selection_boundary='distribution reconstruction; full-source MinMax fit; original runtime TF graph seeds0..4; learned evaluators not independently rerun')
        rows.append(row)
    return rows


def check_increment(original,rows,changed):
    keys={previous.key(r) for r in changed}
    positions=[i for i,(a,b) in enumerate(zip(original,rows)) if a!=b]
    if len(rows)!=len(original) or len(keys)!=4 or len(positions)!=4 or any(previous.key(rows[i]) not in keys for i in positions):
        raise ValueError('Exactly four original cohort metric changes; every other row/timestamp unchanged')
    return positions


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)


def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=legacy.read(ROOT/args.config)
    for pin in config['source_receipts']:
        if legacy.sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen complete native publication evidence changed: '+pin['path'])
    proof=legacy.read(ROOT/config['full_native_proof'])
    result=legacy.read(ROOT/config['original_result'])
    receipt=legacy.read(ROOT/config['original_terminal_receipt'])
    validated_native(proof,result,receipt)
    if (legacy.sha(ROOT/config['original_result'])!=proof['original_result_sha256']
            or legacy.sha(ROOT/config['original_terminal_receipt'])!=proof['actual_terminal_receipt_sha256']):
        raise ValueError('Actual complete native result/terminal receipt differs from independent proof')
    for item in proof['artifacts']:
        if legacy.sha(item['path'])!=item['sha256']:
            raise ValueError('Actual complete original generated/reference/checkpoint artifact changed')
    base=ROOT/config['base_publication']
    old=previous.verified_capture(base,'publication_validation.json')
    fields,original=read_csv(base/'all_algorithm_dataset_metrics.csv')
    changed=updates(original,proof,result,config['full_native_proof'])
    rows=previous.merge_audited_rows(original,changed)
    positions=check_increment(original,rows,changed)
    measured=sum(r['mean'] not in ['',None] for r in rows)
    if len(rows)!=7283 or measured!=old['measured_metric_rows']+4:
        raise ValueError('Entire metric inventory retained with exactly four first measurements')
    job_fields,jobs=read_csv(base/'original_generation_jobs.csv')
    changed_jobs=[]
    for index,row in enumerate(jobs):
        if row['id']==CANONICAL:
            if row['status']!='no_complete_result':
                raise ValueError('Exactly the original pending canonical generator slot required')
            jobs[index]=dict(row,status='completed_full_native_artifacts_and_evaluator_statistics_audited',
                result_path=config['original_result'],result_sha256=proof['original_result_sha256'])
            changed_jobs.append(index)
    if len(jobs)!=122 or len(changed_jobs)!=1:
        raise ValueError('Retain all original generator jobs and update exactly one canonical slot')
    target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Published complete result inventories are immutable')
    target.mkdir(parents=True)
    write_csv(target/'all_algorithm_dataset_metrics.csv',fields,rows)
    write_csv(target/'original_generation_jobs.csv',job_fields,jobs)
    for name in ['overview.md','strict_algorithm_dataset.md','original_energy_jobs.csv']:
        shutil.copyfile(base/name,target/name)
    columns=['任务','算法/消融','数据集','协议','指标','训练种子完成/预定','均值','种子STD','种子SE',
        '种子95%下界','种子95%上界','论文值','评价重复数','评价STD','评价SE','评价95%区间','采集时间UTC']
    values=[[r['task'],r['algorithm'],r['dataset'],r['protocol'],r['metric'],
        '%s/%s'%(r['n'],r['required_repeats']),
        *[legacy.fmt(r.get(k)) for k in ['mean','std','se','ci95_low','ci95_high','paper_mean']],
        r.get('evaluator_repeats',''),legacy.fmt(r.get('evaluator_std')),legacy.fmt(r.get('evaluator_se')),
        r.get('evaluator_ci95',''),r['captured_utc']] for r in rows]
    (target/'all_metrics.md').write_text('# 完整算法—数据集实验技术指标 v28\n\n'
        '仅新增原生Diffusion-TS/Stocks seed42四项实测指标；其他7279行和采集时间保留。'
        '1/5是生成器训练种子数，五次评价的STD/SE/区间单列，不是五个训练种子。'
        'SMAP历史预处理问题及修正重跑仍未完成；作者/TAB等价未确认；—表示缺结果。\n\n'
        +legacy.table(columns,values)+'\n',encoding='utf-8')
    html=legacy.interactive_html(rows)
    note='<p>v28新增：Diffusion-TS/Stocks原生完整训练seed42，1/5训练种子。每项5次评价的STD/SE/区间见CSV与all_metrics.md；不作为训练种子不确定性。</p>'
    (target/'results.html').write_text(html.replace('<input id="query"',note+'<input id="query"'),encoding='utf-8')
    summary=[[r['metric'],'1/5',legacy.fmt(r['mean']),legacy.fmt(r['paper_mean']),
        legacy.fmt(r['evaluator_std']),r['evaluator_ci95']] for r in changed]
    (target/'README.rst').write_text('完整算法—数据集指标 v28\n==========================\n\n'
        f'{len(rows)}项登记指标，{measured}项有实测，{len(rows)-measured}项缺结果。仅新增'
        'Diffusion-TS/Stocks发布数据协议seed42的四项原生指标；其余7279行与v27逐字段一致，'
        '所有登记实验和原始时间戳保留。122个原始生成任务中仅该一个槽状态更新；其余121行一致。\n\n'
        +legacy.table(['指标','训练种子完成/预定','本地均值','论文值','评价STD','评价95%区间'],summary)
        +'\n\n实际完整10000优化器更新、梯度累积2即20000训练小批次；3662参考窗口，'
        '4002生成窗口，原批量2001，四项评价各五次完成。完整原生模型/EMA/优化器状态、'
        '全部原始及生成数组、评价原始值及统计独立核验。审核没有重新训练学习型评价器或重生成输出。'
        '一份生成器模型不是五个训练种子，训练种子STD/CI仍为空。全部源序列归一化及分布重建'
        '协议保留，不代表未来留出泛化或异常检测。实际294390参数与论文291318参数差异保留；'
        'TF图种子0..4的运行时修改保留，不宣称作者等价。\n\nSpectral Flow/Stocks五份完整'
        '训练检查点仍未通过完整批量采样；高阶图修正候选仅进入完整原批量数值审核，尚无指标。'
        'GiFlow新修正版种子及其他新增核验未混入本次四行更新。45篇论文、681方法/基线/消融、'
        '273原始数据任务及采用基线仍未全部完成，完整TAB与作者等价未确认。\n',encoding='utf-8')
    receipt=dict(old)
    receipt.update(published_utc=datetime.now(timezone.utc).isoformat(),base_publication=config['base_publication'],
        base_publication_utc=old['published_utc'],base_receipt_sha256=legacy.sha(base/'publication_validation.json'),
        measured_metric_rows=measured,missing_metric_rows=len(rows)-measured,
        measured_task_rows=dict(Counter(r['task'] for r in rows if r['mean'] not in ['',None])),
        changed_numeric_metric_rows=4,replaced_metric_rows=4,newly_measured_metric_rows=4,
        changed_row_positions=positions,newly_completed_canonical_training_slots=1,
        native_Stocks_Diffusion_TS_completed_generator_seeds=1,native_Stocks_Diffusion_TS_required_generator_seeds=5,
        changed_generation_job_rows=1,other_track_rows_and_capture_times_preserved=True,
        all_numeric_tables_byte_identical_to_base=False,base_numeric_csv_prefix_byte_identical=False,
        performance_rows_changed=True,additional_training_seed_slots=0,other_tracks_recaptured=False,
        all_original_experiments_complete=False,author_equivalence_certified=False,
        checks=dict(all_existing_metric_rows_retained=True,exactly_four_native_generation_metrics_added=True,
            all_other_7279_metric_rows_and_timestamps_unchanged=True,
            all_original122_generator_job_rows_retained=True,other121_generator_jobs_unchanged=True,
            evaluation_repeats_not_counted_as_training_seeds=True),
        source_receipts=old['source_receipts']+config['source_receipts']+
            [dict(path=args.config,sha256=legacy.sha(ROOT/args.config))])
    receipt['outputs']={p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',receipt)
    print(dict(metric_rows=len(rows),measured=measured,missing=len(rows)-measured,
        newly_measured_metrics=4,newly_completed_training_seed_slots=1))


if __name__=='__main__':
    main()
