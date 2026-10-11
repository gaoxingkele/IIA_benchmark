"""Publish four corrected GiFlow aggregates only after both full model replays."""
import argparse
import copy
import csv
from datetime import datetime,timezone
import math
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.publish_complete_text_metadata_v29 import complete_text
from scripts.flow_matching.summarize_giflow_canonical_v1 import aggregate
from scripts.flow_matching.verify_giflow_corrected_full_replay_v2 import full_coverage

ROOT,legacy=previous.ROOT,previous.legacy
METRICS=['native_mean_batch_mae','native_mean_batch_mse','native_mean_batch_mape_percent','sqrt_native_mean_batch_mse']

def validate_pair(proofs,results):
    if len(proofs)!=2 or len(results)!=2:
        raise ValueError('Exactly two original complete independently trained seed models required')
    for proof,result,seed,targets in zip(proofs,results,[393,394],[268613,264837]):
        job=result['job']
        expected='giflow_native__reviewed_corrected__air36__main__seed'+str(seed)
        coverage=full_coverage(result,job)
        batches=proof.get('full_batch_receipts',[])
        if (job['id']!=expected or proof['job_id']!=expected or job['seed']!=seed
                or job['track']!='reviewed_corrected' or result['parameters']!=545987
                or proof.get('full_model_replay_and_independent_saved_arrays_verified') is not True
                or proof['coverage']!=coverage or coverage['full_masked_test_targets']!=targets
                or len(batches)!=14 or sum(b.get('targets',0) for b in batches)!=targets
                or sum(b.get('windows',0) for b in batches)!=1747
                or proof['additional_training_seed_slots']!=0 or proof['author_equivalence_certified']
                or result['epochs_completed']!=68 or result['optimizer_updates']!=3264
                or result['selection']['test_used_in_validation']
                or result['selection']['tested_weights_match_validation_checkpoint'] is not True
                or proof['original_metrics']!=result['metrics']
                or proof['prediction_tolerance']!=dict(rtol=.0002,atol=.001)
                or proof['metric_tolerance']!=dict(rel_tol=1e-5,abs_tol=1e-5)):
            raise ValueError('Two actual complete corrected native seed-specific model/array proofs required')
        for metric in METRICS:
            value=result['metrics'][metric]
            if (not math.isfinite(value) or value<0 or not math.isclose(value,
                    proof['independently_recomputed_metrics'][metric],rel_tol=1e-5,abs_tol=1e-5)):
                raise ValueError('Every full original native metric and unchanged tolerance must agree')
    return {name:aggregate([result['metrics'][name] for result in results]) for name in METRICS}

def merge_pair(original,statistics,captured,source):
    merged=copy.deepcopy(original)
    changed=[]
    for i,row in enumerate(merged):
        if (row['algorithm']=='GiFlow/main' and row['dataset']=='air36'
                and row['protocol']=='giflow_native/reviewed_corrected'):
            if (row['task']!='imputation' or row['metric'] not in METRICS or row['n']!='1'
                    or row['required_repeats']!='5' or row['pattern']!='point' or row['missing_ratio']!='0.2'):
                raise ValueError('Exactly the original full corrected five-seed cohort required')
            stats=statistics[row['metric']]
            row.update(n='2',**{key:str(stats[key]) for key in ['mean','std','se','ci95_low','ci95_high']},
                captured_utc=captured,source=source,
                comparison='two_complete_native_training_and_mask_seeds_full_model_replays_independently_verified',
                uncertainty_scope='two_independent_original_training_and_mask_seeds_Student_t_df1',
                author_equivalence_certified='False',strict_TAB_result='False')
            changed.append(i)
    if len(changed)!=4 or {merged[i]['metric'] for i in changed}!=set(METRICS):
        raise ValueError('Only the four original corrected native metric slots may update')
    for i,(a,b) in enumerate(zip(original,merged)):
        if previous.key(a)!=previous.key(b) or i not in changed and a!=b:
            raise ValueError('All scientific identities and other row metadata/timestamps must remain exact')
    return merged,changed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=legacy.read(ROOT/args.config)
    for pin in config['source_receipts']:
        if legacy.sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen complete two-model publication evidence changed: '+pin['path'])
    proofs=[legacy.read(ROOT/p) for p in config['full_native_model_proofs']]
    results=[legacy.read(ROOT/p) for p in config['original_results']]
    statistics=validate_pair(proofs,results)
    base=ROOT/config['base_publication']
    receipt=previous.verified_capture(base,'publication_validation.json')
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        fields,original=reader.fieldnames,list(reader)
    if len(original)!=7283 or len(fields)!=31 or sum(bool(r['mean']) for r in original)!=1809:
        raise ValueError('All registered tracks and measured scope must remain intact')
    for row in original:
        if (row['algorithm']=='GiFlow/main' and row['dataset']=='air36'
                and row['protocol']=='giflow_native/reviewed_corrected'):
            if float(row['mean'])!=results[0]['metrics'][row['metric']]:
                raise ValueError('Original one-seed mean must match the actual first native result')
    captured=datetime.now(timezone.utc).isoformat()
    source=config['output_directory']+'/corrected_two_seed_metric_evidence.json'
    merged,changed=merge_pair(original,statistics,captured,source)
    legacy.validate_rows(merged)
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Preserve every immutable complete metric publication')
    target.mkdir(parents=True)
    with (target/'all_algorithm_dataset_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    for name in ['overview.md','strict_algorithm_dataset.md','original_generation_jobs.csv','original_energy_jobs.csv']:
        shutil.copyfile(base/name,target/name)
    text=complete_text(fields,merged)
    old='数值和每行时间戳与v28一致；v28新增的四项指标仍仅为'
    replacement='本版仅更新GiFlow修正分支Air36的四项聚合及相应时间戳，其余7279行与v29一致；Diffusion-TS/Stocks的四项生成指标仍仅为'
    if old not in text:raise ValueError('Full-field renderer explanatory boundary changed')
    text=text.replace('完整算法—数据集实验技术指标 v29','完整算法—数据集实验技术指标 v30',1).replace(old,replacement,1)
    (target/'all_metrics.md').write_text(text,encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(merged),encoding='utf-8')
    evidence=dict(captured_utc=captured,original_training_seeds=[393,394],required_training_seeds=5,
        full_native_models_replayed=2,full_test_batches=28,full_native_forward_calls=560,
        actual_seed_specific_masked_targets=[268613,264837],statistics=statistics,
        full_native_model_proofs=config['full_native_model_proofs'],original_results=config['original_results'],
        original_seed_metrics=[r['metrics'] for r in results],additional_training_seed_slots=0,
        native_unweighted_batch_mean_metrics_preserved=True,
        square_root_of_mean_batch_MSE_is_not_global_target_weighted_RMSE=True,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
    legacy.write(target/'corrected_two_seed_metric_evidence.json',evidence)
    summary=legacy.table(['指标','均值','训练种子STD','训练种子95%区间','完成/预定'],
        [[k,legacy.fmt(v['mean']),legacy.fmt(v['std']),str(v['ci95_low'])+' / '+str(v['ci95_high']),'2/5'] for k,v in statistics.items()])
    (target/'README.rst').write_text('完整算法—数据集指标 v30\n==========================\n\n'
        '7283项登记指标，1809项已有实测，5474项缺结果。仅更新GiFlow修正分支Air36/main/20%point的四项聚合，'
        '种子由1/5变为2/5。其余7279行、科学配置身份及采集时间逐字段保留；异常检测、生成、能量和概览导出逐字节保留v29。\n\n'
        +summary+'\n\n两份实际完整545987参数模型均按300轮上限及原早停完成68轮/3264更新，并完成原生检查点重推断及所有数组独立核验。'
        '每份1747窗口/14批/20Euler/280前向，实际目标分别268613和264837。STD/SE/Student-t95%区间来自两个独立训练及掩码种子，'
        '不是重复推断种子；n2区间宽，剩余三种子仍待完成。CPU/GPU预测及指标采用原冻结容差，未宣称逐点位相同。\n\n'
        '该修正分支持续EMA、验证选择检查点且无测试进入验证。指标保留原生重叠窗口无权重批均值；sqrt(mean_batch_MSE)不是全局目标加权RMSE。'
        '没有论文直接对应修正分支成绩，作者等价和完整TAB未认证。SMAP预处理修正及45论文/681方法基线消融/273原始数据任务与采用基线仍未完成。\n',encoding='utf-8')
    updated=copy.deepcopy(receipt)
    updated.update(published_utc=captured,base_publication=config['base_publication'],base_publication_utc=receipt['published_utc'],
        base_receipt_sha256=legacy.sha(base/'publication_validation.json'),changed_numeric_metric_rows=4,replaced_metric_rows=4,
        newly_measured_metric_rows=0,changed_row_positions=changed,performance_rows_changed=True,
        newly_completed_canonical_training_slots=0,newly_published_complete_native_training_slots=1,
        giflow_corrected_air36_main_training_seeds=2,giflow_corrected_air36_main_required_training_seeds=5,
        giflow_corrected_air36_full_native_models_replayed=2,giflow_corrected_seed394_full_replay_verified=True,
        additional_training_seed_slots=0,all_numeric_tables_byte_identical_to_base=False,
        other_track_rows_and_capture_times_preserved=True,all_CSV_fields_in_complete_text=True,
        all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False)
    # Remove metadata claims that described the earlier generation-only increment.
    updated.pop('base_numeric_csv_prefix_byte_identical',None)
    updated['checks']=dict(exactly_four_corrected_GiFlow_metrics_updated=True,other7279_rows_and_capture_times_exact=True,
        all_scientific_metric_identities_preserved=True,all_original_generator_and_anomaly_exports_byte_identical=True,
        both_full_original_native_models_replayed_and_arrays_verified=True)
    updated['source_receipts']+=config['source_receipts']+[dict(path=args.config,sha256=legacy.sha(ROOT/args.config))]
    updated['outputs']={p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',updated)
    print(dict(rows=7283,measured=1809,changed_metric_rows=4,corrected_training_seeds=2,required_seeds=5))

if __name__=='__main__':
    main()
