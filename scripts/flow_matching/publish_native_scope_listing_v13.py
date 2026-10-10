"""Publish completed native model evidence and explicitly missing supporting tasks."""
import argparse
import copy
import csv
from datetime import datetime,timezone
import io
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT,legacy=previous.ROOT,previous.legacy


def pending_rows(config,headers,now):
    rows=[]
    def row(**values):
        r={k:None for k in headers}
        r.update(n=0,required_repeats=5,author_equivalence_certified=False,strict_TAB_result=False,captured_utc=now)
        r.update(values);rows.append(r)
    us=previous.load_rows(ROOT/config['ushcn_pending_metrics'])
    for r in us:
        row(algorithm=r['algorithm'],dataset=r['dataset'],task='native_irregular_time_series_reconstruction',
            protocol='/'.join(['LS4-author-native',str(r['split_branch']),'fold'+str(r['fold']),str(r['profile']),str(r['selection'])]),
            metric=r['metric'],source=config['ushcn_pending_metrics'],epochs=500,
            comparison='fold'+str(r['fold']),pattern=r['profile'],metric_unit='native_normalized',
            selection_boundary='Native periodic test-MSE checkpoint selection' if r['selection']!='final_full_model' else 'Complete original final model')
    for profile in ['interpolation','extrapolation']:
        for selection in ['final_full_model','native_best_test_MSE_model']:
            for metric in ['eval_mse','crps','eval_full_nll']:
                row(algorithm='LS4-native/Cauchy-corrected',dataset='PhysioNet2012',task='native_irregular_time_series_reconstruction',
                    protocol='/'.join(['LS4-author-native','fixed-patient-split42',profile,selection]),metric=metric,epochs=500,
                    source=config['physionet_original_queue'],pattern=profile,metric_unit='native_normalized',
                    selection_boundary='Native periodic test-MSE checkpoint selection' if selection!='final_full_model' else 'Complete original final model')
    gt=legacy.read(ROOT/config['gtgan_native_model_config'])
    for case in gt['cases']:
        for metric in ['discriminative','predictive']:
            row(algorithm='GT-GAN/native-original',dataset='stock',task='irregular_time_series_generation',
                protocol='GT-GAN-author-command/'+case['id'],metric=metric,required_repeats=1,
                evaluator_repeats=case['native_evaluator_repeats'],missing_ratio=case['missing_rate'],
                source=config['gtgan_native_model_config'],selection_boundary=case['boundary'],
                uncertainty_scope='Native evaluator repeats are not independent generator training seeds',metric_unit='native_normalized')
    assert len(us)==120 and len(rows)==140
    assert all(r['n']==0 and r['mean'] is None for r in rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    config=legacy.read(ROOT/args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen source/model/data evidence changed')
    base=ROOT/config['base_publication'];old=previous.verified_capture(base,'publication_validation.json')
    gf=legacy.read(ROOT/config['giflow_corrected_full_model_proof'])
    if (not gf['full_checkpoint_contents_and_reinference_passed'] or gf['preserved_completed_native_batches']!=9
            or gf['newly_replayed_native_batches']!=5 or gf['full_test_windows']!=1747
            or gf['full_masked_test_targets']!=268613 or gf['observed_forward_calls']!=280
            or gf['additional_model_forwards_from_metadata_audit']!=0):raise ValueError('Actual full native model replay required')
    gt=legacy.read(ROOT/config['gtgan_original_loader_proof'])
    if not gt['passed'] or len(gt['records'])!=4 or gt['performance_result']:raise ValueError('Full native loader audit required')
    for r in gt['records']:
        if legacy.sha(r['path'])!=r['sha256'] or r['complete_windows']!=3662:raise ValueError('Full original native stock arrays required')
    source=base/'all_algorithm_dataset_metrics.csv';original=previous.load_rows(source)
    headers=next(csv.reader(io.StringIO(source.read_text(encoding='utf-8-sig'))))
    now=datetime.now(timezone.utc).isoformat();extra=pending_rows(config,headers,now);rows=original+extra
    keys=[(r['algorithm'],r['dataset'],r['task'],r['protocol'],r['metric'],r.get('score_recipe'),r.get('pattern'),r.get('missing_ratio'),r.get('epochs'),r.get('comparison')) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('All registered metric slots must stay unique')
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Keep each publication immutable')
    target.mkdir(parents=True)
    # Preserve every earlier numerical row byte for byte, then append empty slots.
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=headers);writer.writerows(extra)
    (target/'all_algorithm_dataset_metrics.csv').write_bytes(source.read_bytes()+output.getvalue().encode('utf-8'))
    for name in ['strict_algorithm_dataset.md','overview.md','original_generation_jobs.csv']:shutil.copyfile(base/name,target/name)
    title=['任务','算法／消融','数据集','协议','评分配方','指标','完成／预定','均值','STD','SE','95%下界','95%上界','论文值','核验时间UTC']
    values=[[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('score_recipe',''),r['metric'],f"{r['n']}/{r['required_repeats']}",
        *[legacy.fmt(r.get(k)) for k in ['mean','std','se','ci95_low','ci95_high','paper_mean']],r['captured_utc']] for r in extra]
    (target/'all_metrics.md').write_bytes((base/'all_metrics.md').read_bytes()+('\n\n## 新接入的原作者任务待完成指标\n\n'+legacy.table(title,values)+'\n').encode('utf-8'))
    (target/'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    measured=sum(r['mean'] is not None for r in rows);assert measured==old['measured_metric_rows']==1786
    (target/'README.md').write_text('# 算法—数据集完整指标与已知缺口\n\n'+f'发布：{now}。主表{len(rows)}条指标记录：{measured}条有实测值，{len(rows)-measured}条缺结果。指标行数不是训练次数；全部原论文实验仍未完成。\n\n'
        +'[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整文字表](all_metrics.md)；[异常检测全部指标](strict_algorithm_dataset.md)。\n\n'
        +'继承v12全部7117条历史记录与各自采集时间，数值逐字节保留；增加120条USHCN、12条原始PhysioNet、8条GT-GAN待完成指标。所有新槽位均为缺结果，不把数据审计、容量诊断或重推断计为新训练性能。\n\n'
        +'GiFlow/Air36/seed393实际保存的EMA权重已经通过全1747窗口、14批次、268613个目标、20步Euler采样的完整重推断，累计280次模型调用。原生MAE/MSE/MAPE分别为9.654690/359.798492/21.516833%；CPU重算均满足冻结容差。回执中两项列表别名导致的批次数统计已独立修正为9批保留、5批新算，原回执与全部预测保留。[修正与全数组核验](../giflow_native_checkpoint_receipt_audit_v3/corrected_full_native_replay_audit.json)。仍为1/5训练种子，不新增STD/CI，不获作者等价认证。\n\n'
        +'GT-GAN作者完整245个Git文件已下载并逐一核验；原始Stocks3685行、6变量、全部3662个24步窗口在0/30/50/70%缺失设置下完成原生预处理和样条核验。原生训练和评估尚未完成；30/50%使用明确标注的源程序参数扩展，历史表行等价未证明。\n\n'
        +'LS4原始PhysioNet插值/外推完整一轮GPU容量检查都通过，分别100/200次更新及25/50个完整测试批次。第一项完整500轮训练已启动；尚未形成500轮性能。USHCN100项500轮任务仍需完成。\n\n'
        +'[本轮完整执行证据](../gtgan_native_and_giflow_execution_v1/README.md)；[已完成GRASP τ=0.25完整模型核验](../grasp_tau025_checkpoint_replay_v1/checkpoint_replay_audit.json)。其他轨道按原快照保留，不把本次发布日期当作它们重新核验的时间。\n',encoding='utf-8')
    validation=copy.deepcopy(old);validation.update(published_utc=now,base_publication_utc=old['published_utc'],
        aggregate_metric_rows=len(rows),measured_metric_rows=measured,missing_metric_rows=len(rows)-measured,
        added_pending_metric_rows=len(extra),newly_measured_metric_rows=0,base_numeric_csv_prefix_byte_identical=True,
        giflow_independent_model_reinference_complete=True,giflow_replay_receipt_metadata_corrected=True,
        gtgan_full_native_original_data_audit_passed=True,physionet_full_native_both_gpu_capacity_passed=True,
        additional_training_seed_slots=0,other_tracks_recaptured=False,all_original_experiments_complete=False)
    validation['task_rows']={t:sum(r['task']==t for r in rows) for t in sorted({r['task'] for r in rows})}
    validation['source_receipts']+=config['source_receipts']+[dict(path=args.config,sha256=legacy.sha(ROOT/args.config))]
    validation['outputs']={p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',validation)
    print(dict(rows=len(rows),measured=measured,missing=len(rows)-measured,new_pending_slots=len(extra),new_performance_rows=0))


if __name__=='__main__':main()
