"""Analytic diagnostic from verified held-out label counts; no new model run."""
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write

source_folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_additional_datasets_full_native_closed_v1'
for path,digest in read(source_folder/'archive_validation.json')['files_sha256'].items():
    assert sha(source_folder/path)==digest
source=source_folder/'complete_primary_calibration_and_scope_metrics.json'
records=[]
for row in read(source)['records']:
    if row['validation_score_quantile_exceedance_percent']!=1.:continue
    n=row['test_points'];positive=round(row['test_anomaly_fraction']*n);p=positive/n
    assert 0<positive<n and p==row['test_anomaly_fraction']
    records.append(dict(dataset=row['dataset'],test_points=n,anomaly_points=positive,anomaly_fraction=p,
        analytic_always_anomaly_F1=2*positive/(n+positive),analytic_constant_score_AUROC=.5,
        analytic_constant_score_AP=p,model_validation_quantile_no_PA_F1=row['micro_no_PA_F1'],
        model_same_threshold_PA_F1=row['same_validation_threshold_PA_F1'],model_AUROC=row['AUROC'],model_AP=row['AP'],
        model_AP_minus_constant_score_AP=row['AP']-p,model_PA_minus_no_PA_F1=row['same_validation_threshold_PA_F1']-row['micro_no_PA_F1'],
        source_track=row['source_track']))
assert len(records)==6
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/PA_and_prevalence_analytic_diagnosis_v1'
folder.mkdir(parents=True,exist_ok=False)
write(folder/'analytic_references_and_observed_metrics.json',dict(records=records,source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),
    formulae=dict(always_anomaly_precision='P/N',always_anomaly_recall='1',always_anomaly_F1='2*P/(N+P)',
        tied_constant_score_AP='P/N',tied_constant_score_AUROC='0.5 when both classes present'),
    type='EXACT_OBSERVED_LABEL_COUNT_ALGEBRA_NOT_NEW_TRAINING_OR_LEADERBOARD_EXPERIMENT',
    model_metrics_from_complete_native_and_independent_review=True,new_model_training_or_inference=False,
    independent_lowlevel_metric_implementation_added=False,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False))
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
(folder/'all_six_dataset_analytic_references.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
lines=['# 高F1和高AP：PA及异常比例的解析核对','',
    '下面只用已通过完整原生重推理和独立复核的六组固定测试标签计数做代数计算，不是新增训练算法、重训种子或排行榜实验。设测试总点数N、异常点数P，则全部报异常时precision=P/N、recall=1、F1=2P/(N+P)。所有分数恒等时，有正负两类的AUROC=0.5，合并ties后的AP=P/N。这些是该固定样本上的精确参照，不推断其他总体。','',
    '|数据集|异常点比例|全部报异常F1解析值|MOMENT未PA F1|同阈值PA F1|MOMENT AUROC|MOMENT AP|恒分数AP解析值|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in records:
    lines.append(f"|{r['dataset']}|{r['anomaly_fraction']:.6f}|{r['analytic_always_anomaly_F1']:.6f}|{r['model_validation_quantile_no_PA_F1']:.6f}|{r['model_same_threshold_PA_F1']:.6f}|{r['model_AUROC']:.6f}|{r['model_AP']:.6f}|{r['analytic_constant_score_AP']:.6f}|")
lines += ['',
    'PRONTO day2的94.19%点标为异常，所以全部报异常的F1已经约0.97006、恒分数AP约0.94187；模型AP约0.95376，而固定验证阈值未PA F1约0.04675、同阈值PA F1可达1.0。经典TEP的异常比例83.33%，全部报异常F1约0.90909。因此仅报九十几F1或高AP，尚不足以说明模型有强区分能力。全部报异常会把每个正常点都误报，当然不是可用报警器。','',
    'PA按测试真实异常段补齐预测，能够显著提高点级召回和F1；它没有改善原始排序分数。这里比较的是同一固定验证阈值，未用测试F1调阈值或按测试AUROC翻转分数。每例AUROC/AP、未调整F1、PA诊断、事件召回和每1000正常点误报应同时看，原单模型条件置信区间保留在来源包。','',
    '这些证据不能直接判定某篇论文结论错误：CICIDS子集、经典TEP与多模式HDF5、PRONTO标签辅助正常筛选、SKAB含异常校准、模型尺寸/上下文、预训练配方和历史种子均需逐项对齐。当前证明的是本地已登记实现的重复执行与独立指标一致；历史论文配方/数据端到端等价及完整TAB执行器仍未证明。','',
    '后续原因定位仍保留在研究范围：先核对标签极性与分数方向，再检查跨组/跨日分布漂移和逐窗口RevIN是否吸收持久异常；AUROC低于0.5只触发诊断，禁止依据测试结果自动倒置分数。公平主比较继续使用冻结超参数、同一审计数据/组划分、验证集阈值及配对种子/分组不确定性；原作者口径、共同严格口径和新数据迁移分开保存。','',
    '原NODE、CrossAD、SAITS、MOMENT和其他已登记实验仍全部继续，本解析表不代替任何尚未完成的实验。']
(folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'analysis_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),source_native_archive_all_hashes_rechecked=True,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},new_experiment_or_training_result=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/PA_prevalence_analytic_diagnosis_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(datasets=6,exact_files=len(paths),new_experiments=0,always_positive_day2_F1=next(r for r in records if r['dataset']=='pronto_day2')['analytic_always_anomaly_F1'])))
