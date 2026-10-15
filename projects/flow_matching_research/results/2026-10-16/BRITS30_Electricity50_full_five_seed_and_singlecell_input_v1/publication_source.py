"""Verify the completed59 evidence union and full10-model matched seed cohort."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv
import itertools
import json
import math
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import t
import psutil
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v1 as prep
from scripts.flow_matching.verify_saits_completed_full_replay_batch_v1 import check_terminal

root=prep.ROOT;read,sha=prep.read,prep.sha
out=root/'projects/flow_matching_research/results/2026-10-16/BRITS30_Electricity50_full_five_seed_and_singlecell_input_v1'
out.mkdir(parents=True,exist_ok=False)
paths=[];snapshots=[];cache={}


def digest(path):
    path=Path(path);key=str(path.resolve()).casefold()
    if key not in cache:cache[key]=sha(path)
    return cache[key]


def write(name,value):
    prep.exclusive(out/name,value)


def snapshot(path,name,original=False):
    path=root/path;target=out/name
    assert not target.exists();shutil.copy2(path,target);assert path.read_bytes()==target.read_bytes()
    snapshots.append(dict(path=str(path),sha256=digest(path),snapshot=name))
    if original:paths.append(path.relative_to(root).as_posix())


previous='projects/flow_matching_research/results/2026-10-16/BRITS29_full_review_TAB_handoff_RCGAN_cold_v1/actual_58_complete_native_and_independent_model_union.json'
old=read(root/previous);rows={r['id']:r for r in old['records']};assert len(rows)==58
new_id='saits_paper_brits_electricity_author_missing0.5_seed30'
review_config='configs/reproducibility/fm_saits_completion_replay_queue_v1/'+new_id+'.review.json'
review=read(root/review_config);new_proof_path=root/review['output_directory']/'independent_full_replay_batch_verification.json'
new_proof=read(new_proof_path);new_record=new_proof['records'][0]
rows[new_id]=dict(id=new_id,method='brits',dataset='electricity',nominal_missing_ratio=.5,seed=30,
    metrics=new_record['independently_recomputed_metrics'],original_result_sha256=new_record['hashes']['result'],
    independent_review_configuration=review_config,proof_path=str(new_proof_path.relative_to(root)),
    proof_sha256=digest(new_proof_path),review_guard_sha256=digest(root/review['state_root']/'receipt.json'))
proofs={};cohort=[];all_models=[]
for identity,row in sorted(rows.items()):
    rc=read(root/row['independent_review_configuration']);path=root/row['proof_path'];proof=read(path)
    guard_path=root/rc['state_root']/'receipt.json';guard=read(guard_path)
    assert digest(path)==row['proof_sha256'] and digest(guard_path)==row['review_guard_sha256']
    assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
    assert proof['configuration_sha256']==digest(root/row['independent_review_configuration'])
    assert guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_native_batch_independent_review_passed']
    nc=read(root/rc['replay_configuration']);native_path=root/nc['artifact_directory']/'checkpoint_replay_audit.json'
    native=read(native_path);check_terminal(native,read(root/nc['state_root']/'receipt.json'),nc)
    assert proof['native_replay_report_sha256']==digest(native_path)
    assert native['configuration_sha256']==digest(root/rc['replay_configuration'])
    r=next(r for r in proof['records'] if r['id']==identity)
    nr=next(r for r in native['records'] if r['id']==identity)
    result_path=Path(nr['result_path']);result=read(result_path)
    assert digest(result_path)==r['hashes']['result']==row['original_result_sha256']
    assert row['metrics']==r['independently_recomputed_metrics']
    assert r['full_training_completion']['full_training_finished'] and not r['full_training_completion']['benchmark_smoke']
    all_models.append(row)
    if any(identity.startswith('saits_paper_'+method+'_electricity_author_missing0.5_seed') for method in ['saits','brits']):
        config=read(root/nr['config']);data=root/config['dataset_root'];output=root/config['output_root']
        files=dict(config=root/nr['config'],result=result_path,checkpoint=Path(nr['checkpoint_path']),arrays=Path(nr['replay_arrays']),
            dataset=data/'datasets.h5',resume=output/'resume.pt',original_group_statistics=output/'group_statistics.npz',data_audit=data/'data_audit.json')
        for key,value in files.items():assert digest(value)==r['hashes'][key],(identity,key)
        for pin in rc['source_receipts']+nc['source_receipts']:assert digest(root/pin['path'])==pin['sha256']
        assert r['full_test_shape']==[290,100,370] and r['held_out_coordinates']==4221734 and r['months']==10
        assert config==result['config'] and config['variant']=='author' and not config['corrected_masking']
        assert config['seed'] in [26,27,28,29,30] and r['actual_full_checkpoint_reinference_completed']
        cohort.append(dict(**row,parameters=nr['model_parameters'],full_grid_cells=10730000,heldout_coordinates=4221734,
            dataset_sha256=r['hashes']['dataset'],data_audit_sha256=r['hashes']['data_audit'],
            full_training_completion=r['full_training_completion'],validation_best=result['validation_best'],
            original_metrics=result['metrics'],month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],
            original_configuration=config))
        proofs[identity]=r
assert len(all_models)==59 and len(cohort)==10
live=read(root/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json')
assert set(rows)==set(live['verified_original_ids'])
write('actual_59_completed_native_and_independent_model_union.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    required_models=210,completed_models=59,remaining_models=151,records=all_models,
    current_scope='All59 native/independent success guard,config,proof,result hashes rechecked. Full checkpoint/resume/data/replay arrays and all source pins additionally rehashed for the matched10 model cohort.',
    all_original_experiments_complete=False))
by_method={method:sorted([r for r in cohort if r['method']==method],key=lambda r:r['seed']) for method in ['saits','brits']}
for method,records in by_method.items():
    assert [r['seed'] for r in records]==[26,27,28,29,30]
    assert len({r['parameters'] for r in records})==1
    configs=[{k:v for k,v in r['original_configuration'].items() if k not in ['id','seed','output_root']} for r in records]
    assert all(c==configs[0] for c in configs)
assert len({r['dataset_sha256'] for r in cohort})==len({r['data_audit_sha256'] for r in cohort})==1
critical=float(t.ppf(.975,4));assert math.isclose(critical,2.7764451051977987,rel_tol=1e-10)


def stats(values):
    values=np.asarray(values,dtype=np.float64);assert values.shape==(5,) and np.isfinite(values).all()
    mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/math.sqrt(5)
    assert math.isclose(mean,statistics.mean(values.tolist()),rel_tol=1e-14)
    assert math.isclose(sd,statistics.stdev(values.tolist()),rel_tol=1e-12)
    return dict(n=5,mean=mean,std_ddof1=sd,se=se,student_t95_low=mean-critical*se,student_t95_high=mean+critical*se,
        minimum=float(values.min()),maximum=float(values.max()),median=float(np.median(values)),
        leave_one_seed_out_means=[float(np.delete(values,i).mean()) for i in range(5)])


paper=dict(brits=dict(mae=1.037,rmse=1.538,mre=.555),saits=dict(mae=.876,rmse=1.377,mre=.469))
summary={method:{metric:stats([r['metrics'][metric] for r in records]) for metric in ['mae','rmse','mre']}
         for method,records in by_method.items()}
paired={}
for metric in ['mae','rmse','mre']:
    diff=[a['metrics'][metric]-b['metrics'][metric] for a,b in zip(by_method['saits'],by_method['brits'])]
    wins=sum(v<0 for v in diff);losses=sum(v>0 for v in diff);ties=sum(v==0 for v in diff);n=wins+losses
    sign_p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else None
    paired[metric]=dict(seed_paired_difference_SAITS_minus_BRITS=stats(diff),seed_values=diff,
        SAITS_lower_error_seeds=wins,BRITS_lower_error_seeds=losses,ties=ties,two_sided_exact_sign_p=sign_p,
        relative_error_reduction_ratio_of_means=1-summary['saits'][metric]['mean']/summary['brits'][metric]['mean'])
assert all(paired[m]['SAITS_lower_error_seeds']==5 and paired[m]['two_sided_exact_sign_p']==.0625 for m in paired)
target_configuration='configs/reproducibility/fm_saits_Electricity50_full_five_seed_summary.2026-10-16_v1.json'
prep.exclusive(root/target_configuration,dict(required_seeds=[26,27,28,29,30],methods=['saits','brits'],dataset='electricity',
    nominal_missing_ratio=.5,cohort_size=10,reference_paper='papers/literature/flow_matching/pdfs/saits.pdf',
    reference_paper_sha256=digest(root/'papers/literature/flow_matching/pdfs/saits.pdf'),reference_pdf_page=13,
    reference_table=4,reference_column='50%',reference_metrics=paper,paper_MRE_percent_converted_to_fraction=True,
    local_metric_source='Complete CPU-native reinference and independent every-coordinate recomputation, not rounded table values.',
    statistical_scope='Five preregistered training seeds, fixed split/mask/preprocessing. Nominal unadjusted t95 CI df4 under iid seed effect assumption; not new-dataset or temporal uncertainty.',
    original_scope=prep.registration()['original_scope'],author_equivalence_certified=False,strict_TAB_result=False))
write('ten_full_models_five_seed_metrics_and_paper_comparison.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    configuration=target_configuration,configuration_sha256=digest(root/target_configuration),completed_full_models=10,
    cohort=cohort,training_seed_statistics=summary,matched_seed_comparisons=paired,paper_table4_50percent=paper,
    paper_gap={m:{metric:dict(local_mean=summary[m][metric]['mean'],paper_value=paper[m][metric],
        signed_absolute_gap=summary[m][metric]['mean']-paper[m][metric],
        relative_gap_fraction=summary[m][metric]['mean']/paper[m][metric]-1) for metric in paper[m]} for m in paper},
    actual_mask_fraction=4221734/10730000,nominal_mask_fraction=.5,local_all_split_windows=1397,paper_all_split_windows=1400,
    all_source_guarded_native_and_independent_arrays_verified=True,all10_models_full_data_checkpoint_and_source_hashes_rechecked=True,
    new_training_seeds_from_this_publication=0,full_imputation_queue_complete=False,all_original_experiments_complete=False,
    boundary='Five seed effects on one frozen dataset and mask. Original per-rate historical tuning/preprocessing seeds and exact window generation are not established. '
        'The method ordering is consistent for this registered cohort; absolute paper agreement is not certified. No temporal/heldout-month uncertainty is inferred from seed intervals. '
        'All5 paired wins give exact two-sided sign p=.0625; no distribution-free5percent-significance claim. Nominal t intervals are unadjusted across metrics.'))
with (out/'all_ten_model_metrics.csv').open('x',encoding='utf-8',newline='') as f:
    fields=['id','method','dataset','seed','nominal_missing_ratio','parameters','full_grid_cells','heldout_coordinates','mae','rmse','mre']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in cohort:writer.writerow({k:r[k] for k in fields if k not in ['mae','rmse','mre']}|r['metrics'])
with (out/'five_seed_summary_and_paper_targets.csv').open('x',encoding='utf-8',newline='') as f:
    fields=['method','metric','n','mean','std_ddof1','se','student_t95_low','student_t95_high','paper_table4_50percent','local_minus_paper','author_equivalence_certified']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for method in summary:
        for metric,s in summary[method].items():writer.writerow(dict(method=method,metric=metric,**{k:s[k] for k in fields if k in s},
            paper_table4_50percent=paper[method][metric],local_minus_paper=s['mean']-paper[method][metric],author_equivalence_certified=False))
snapshot(review_config,'BRITS30_review_configuration.json',True)
snapshot(review['replay_configuration'],'BRITS30_native_configuration.json',True)
snapshot(new_proof_path,'BRITS30_independent_full_array_proof.json',True)
snapshot(root/review['output_directory']/'native_checkpoint_replay_audit.json','BRITS30_full_native_replay.json',True)
native_config=read(root/review['replay_configuration'])
snapshot(root/native_config['state_root']/'receipt.json','BRITS30_native_guard.json')
snapshot(root/review['state_root']/'receipt.json','BRITS30_independent_guard.json')
snapshot(root/'experiments/runs/flow_matching_campaign'/new_id/'result.json','BRITS30_original_result.json')

pc=read(root/prep.CONFIG);prep.verify(pc);base=root/pc['state_root'];launch=read(base/'launch.json');process=psutil.Process(launch['pid'])
assert process.create_time()==launch['create_time'] and process.cmdline()==launch['command']
status=read(base/'status.json');assert status['pid']==process.pid
write('actual_full_Multiome13176_reconstruction_execution.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    controller=launch,live_exact_identity=True,private_bytes=process.memory_info().private,state=status,
    stderr_bytes=(base/'controller.stderr.log').stat().st_size,full_reference_cells=33701,paper_cohort_cells=35384,
    features=23418,thresholds_and_budgets_relaxed=False,
    completion_is_unproven_until_full_matrix_PCA_report_and_successful_guard=True,all_original_experiments_complete=False))
for index,path in enumerate([prep.MODULE.replace('.','/')+'.py',prep.TEST,prep.CONFIG]):snapshot(path,'source%02d%s'%(index,Path(path).suffix),True)
xml=Path('F:/aicoding/IIA_Data/experiments/otcfm_multiome13176_reconstruction_tests_v2.xml');suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==25 and not sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
snapshot(xml,'input_reconstruction_validation25.xml')
write('validation_and_source_bindings.json',dict(source_mapping=snapshots,PCA_and_guard_regression_tests_passed=25,
    actual_full_original_complete_models_in_new_seed_cohort=10,independent_59_union_guard_config_result_validation_passed=True,
    original_10_model_checkpoint_resume_masks_arrays_and_source_file_hashes_checked=len(cache),
    statistics_independently_cross_checked_with_statistics_module=True,student_t95_critical_df4=critical,
    no_synthetic_test_is_a_benchmark_result=True,all_original_experiments_complete=False))
(out/'README.md').write_text(f'''# BRITS seed30 完整复核及 Electricity 名义50%缺失五种子比较

新增 BRITS Electricity、名义50%缺失、seed30 已完成训练与作者早停，验证集选中 epoch1032，next_epoch1064。完整检查点重推理及独立逐坐标复核守卫均退出0，未发生资源中止。独立 MAE=1.0707066172195736、RMSE=1.55865968201309、MRE=0.5732031164655268。完整290×100×370测试网格、4,221,734留出坐标和10个月份均已复核。原210模型队列完成复核并集增至59，剩151；不是全部完成。

SAITS 与 BRITS 在此固定数据/掩码上的26..30五个种子现已全部完成，共10个完整模型。重新核对全部成功原生/独立守卫、配置和结果，并针对10模型再次读取原始模型/恢复状态/数据/预测数组/月份统计和全部冻结源文件，校验哈希。

| 方法 | MAE 五种子均值 ± 样本SD | 名义 t95 区间（df4） | 论文 Table4 50% MAE |
|---|---|---|---|
| SAITS | {summary['saits']['mae']['mean']:.6f} ± {summary['saits']['mae']['std_ddof1']:.6f} | [{summary['saits']['mae']['student_t95_low']:.6f}, {summary['saits']['mae']['student_t95_high']:.6f}] | 0.876 |
| BRITS | {summary['brits']['mae']['mean']:.6f} ± {summary['brits']['mae']['std_ddof1']:.6f} | [{summary['brits']['mae']['student_t95_low']:.6f}, {summary['brits']['mae']['student_t95_high']:.6f}] | 1.037 |

此队列中 SAITS 的平均 MAE 比 BRITS 低约13.05%，五个配对种子的三项误差指标都更低，方法排序与论文方向一致。完整 MAE/RMSE/MRE、每种子数值、论文目标差值、配对差、留一种子敏感性见 CSV/JSON。五次同向胜出的双侧精确符号检验 p=0.0625，不能宣称分布无关的5%显著性；t区间假设独立同分布的种子效应，且未作跨指标多重校正。

**绝对数值仍不能认证复现一致。** 论文目标来自原PDF第13页Table4、50%列；MRE已从百分数换为比例。本地实际有放回掩码比例约39.345%，历史具体掩码/预处理种子及逐缺失率调参不可核实；总窗口1397与论文1400不同。种子区间只描述固定划分上的训练随机性，不覆盖数据划分/时间相关性/预处理不确定性，也不能用来对缺少不确定性的论文单值作等价性认证。这是插补误差，不是异常检测 F1/TAB 成绩。

同时登记并启动了 Multiome 13176 输入重建：须先对作者32606的全部33,701细胞×23,418基因验证唯一细胞对应和全部矩阵值，再将固定变换应用于论文13176的全部35,384细胞、四个时间点，重新计算中心化五PC并执行原作者标准化块。25项检查通过，包括稠密独立SVD对照、错误候选必须由全矩阵门禁拒绝、原标准化块一致性、全量/资源设置不可缩减及冷启动检查。实际进程/等待门槛见执行记录；若仍等待32GiB准入，不算数据已重建。原始历史PCA工件位级一致性尚不可认证，CITE特征映射也仍待解决。

原45篇论文、681项方法/基线/消融、273数据任务、702论文标量目标及全部未完成原生实验继续保留。未缩减训练、数据坐标、积分/度量网格或资源保护；未覆盖旧失败或原始数据。
''',encoding='utf-8',newline='\n')
snapshot(Path(__file__),'publication_source.py')
paths.append(target_configuration);paths.extend(p.relative_to(root).as_posix() for p in out.iterdir())
git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),paths=paths)
(root/'tmp/BRITS30_five_seed_singlecell_input_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(paths),completed_union_models=59,cohort_full_models=10,
    SAITS_MAE=summary['saits']['mae'],BRITS_MAE=summary['brits']['mae'],reconstruction_state=status['state']),ensure_ascii=False))
