"""Close the actual full5 model cohort and all8 fully read public CITE payloads."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scipy.stats import t as student_t
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as native
from scripts.flow_matching.otcfm_singlecell_result_integrity_v2 import inspect
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching import acquire_CITE13176_GEO_official_v1 as acquisition

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Multiome_first_full5_native_cohort_and_CITE13176_GEO8_v1'
assert not packet.exists()
queue=read(ROOT/native.QUEUE);native.verify(queue)
config_path='configs/reproducibility/fm_otcfm_singlecell_full120_checkpoint_replay.v2.json'
config=read(ROOT/config_path)
jobs=[j for j in queue['jobs'] if j['dataset']=='released_author_donor32606' and j['protocol']=='released_yaml'
      and j['method']=='cfm' and j['leaveout_timepoint']==1]
assert len(jobs)==5 and sorted(j['seed'] for j in jobs)==[42,43,44,45,46]
rows=[];archives=[];full_cost_entries=0
for job in sorted(jobs,key=lambda j:j['seed']):
    result,provenance=inspect(queue,job);key=Path(job['artifact_directory']).name
    replay_path=Path(config['artifact_root'])/key/'full_checkpoint_replay.json'
    guard_path=ROOT/config['state_root']/'attempts'/key/'guard_receipt.json'
    r,g=read(replay_path),read(guard_path)
    assert r['status']=='passed' and r['id']==job['id'] and r['configuration_sha256']==sha(ROOT/config_path)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['full_replay_report_sha256']==sha(replay_path)
    assert r['source_provenance']['result_sha256']==provenance['result_sha256']
    assert r['full_training_epochs']==result['actual_completed_epochs']==1000
    assert r['full_training_optimizer_updates']==result['epoch_history'][-1]['optimizer_updates']==10000
    assert r['original_posttest_selected_checkpoint_updates']==result['optimizer_updates']
    assert r['all_learned_tensors_and_optimizer_moments_checked'] and r['full_native_test_reinference_passed']
    assert r['native_metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert r['native_metrics']==r['original_metrics']==result['test_metrics'] and len(r['native_metrics'])==36
    assert r['extra_model_training_updates']==0 and not r['historical_paper_equivalence_certified'] and not r['strict_TAB_result']
    solves=r['full_test_population_distances_independently_recomputed'][0]['solves']
    assert len(solves)==4 and all(x['native_LP_result_code']==1 for x in solves)
    w1=[x for x in solves if x['power']==1]
    assert [x['full_cost_shape'] for x in w1]==[[7264,8544],[8544,10933]]
    assert all(x['independent_LP_result_code']==1 and x['full_costs_independently_recomputed']
        and x['objective_absolute_difference']<=x['LP_objective_perturbation_bound'] for x in w1)
    full_cost_entries+=sum(x['full_cost_shape'][0]*x['full_cost_shape'][1] for x in w1)
    for item in r['all_full_test_arrays_and_artifacts']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    rows.append(dict(id=job['id'],seed=job['seed'],epochs=1000,total_training_updates=10000,
        selected_epoch=r['selected_checkpoint_epoch'],selected_updates=r['selected_checkpoint_updates'],
        heldout_W1=result['primary_heldout_W1'],metrics=result['test_metrics'],
        result_sha256=provenance['result_sha256'],replay_sha256=sha(replay_path),guard_sha256=sha(guard_path),
        exact_all36_native_metric_equality=True,full_independent_W1_solves=w1))
    archives.append((job['seed'],dict(original_training_result=Path(provenance['result_path']),
        original_training_guard=Path(provenance['guard_path']),full_native_checkpoint_replay=replay_path,native_replay_guard=guard_path)))
assert full_cost_entries==777375840
metrics={}
for name in rows[0]['metrics']:
    values=[r['metrics'][name] for r in rows];actual=independent_stats(values)
    mean=math.fsum(values)/5;std=statistics.stdev(values)
    ci=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5)) if std else (mean,mean)
    check=dict(mean=mean,std=std,ci95_low=float(ci[0]),ci95_high=float(ci[1]))
    assert all(math.isclose(actual[k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in check.items())
    metrics[name]=dict(**actual,degrees_of_freedom=4,per_original_seed_values=values,independent_scalar_and_SciPy_t95_verified=True)

download=read(ROOT/acquisition.CONFIG);acquisition.validate_sources(download);raw=Path(download['output_directory'])
proof_path=raw/'all8_payload_integrity.json';proof=read(proof_path)
assert proof['status']=='all8official_payloads_complete_and_fully_read' and proof['files']==8
assert proof['total_bytes']==1780469041 and proof['configuration_sha256']==sha(ROOT/acquisition.CONFIG)
assert not proof['publisher_checksum_verified'] and not proof['historical_Kaggle_or_paper_processed_identity_certified']
assert read(ROOT/download['state_root']/'aria2c_exit.json')['exit_code']==0
official={s['filename']:s for s in download['sources']};source_rows=[]
for r in proof['records']:
    p=Path(r['path']);item=official[p.name]
    assert p.stat().st_size==r['bytes']==item['expected_bytes'] and sha(p)==r['sha256']
    assert r['every_HDF5_leaf_payload_read'] and not Path(str(p)+'.aria2').exists()
    assert r['day']==item['day'] and r['kind']==item['kind'] and r['url']==item['url']
    source_rows.append(dict(filename=p.name,path=str(p),day=r['day'],kind=r['kind'],bytes=r['bytes'],sha256=r['sha256'],
        cohort_shape_hints=r['cohort_shape_hints'],HDF5_leaf_datasets_fully_read=r['leaf_datasets'],
        numeric_nonfinite_including_metadata=r['numeric_nonfinite'],publisher_checksum_verified=False))
test=Path('F:/aicoding/IIA_Data/experiments/Multiome_full_five_CITE_GEO_payload_closure_20261016_v1_tests.xml')
suites=list(ET.parse(test).getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==58
assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
packet.mkdir(parents=True,exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.csv -text\n*.gz -text\n',encoding='utf-8',newline='\n')
for seed,files in archives:
    target=packet/('seed'+str(seed));target.mkdir()
    for label,p in files.items():shutil.copyfile(p,target/(label+'.json'))
primary=metrics[queue['primary_metric']]
write(packet/'full_five_seed_all36_metric_summary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    dataset='released_author_donor32606',method='cfm',protocol='released_yaml',leaveout_timepoint=1,seeds=[42,43,44,45,46],
    full_five_training_models_complete=True,full_five_native_checkpoint_replays_complete=True,
    all180_original_scalar_replays_exactly_equal=True,total_independent_full_cost_entries=full_cost_entries,
    native_exact_W1_W2_OPTIMAL_solves=20,independent_full_W1_OPTIMAL_solves=10,all36_metric_seed_summaries=metrics,
    per_model=rows,uncertainty_scope='small-n model-mean Student-t95/sampleSTD over five training seeds on one frozen dataset/split/leaveout;not across donors/datasets/leaveouts',
    historical_paper_identity_certified=False,strict_TAB_result=False,source_full120_still_required=True,
    source_other150_and90_cases_still_required=True,original_scope=queue['original_scope'],all_original_experiments_complete=False))
with (packet/'all36_full_five_seed_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['metric','primary','mean','sample_STD','model_mean_t95_low','model_mean_t95_high',*[f'seed{s}' for s in range(42,47)]])
    for name,r in metrics.items():writer.writerow([name,name==queue['primary_metric'],r['mean'],r['std'],r['ci95_low'],r['ci95_high'],*r['per_original_seed_values']])
shutil.copyfile(proof_path,packet/'all8_official_CITE_payload_integrity.json')
write(packet/'CITE_donor13176_raw_processed_source_manifest.json',dict(sources=source_rows,source_configuration=acquisition.CONFIG,
    source_configuration_sha256=sha(ROOT/acquisition.CONFIG),full_HDF5_leaf_read_and_local_SHA256_checked=True,
    publisher_checksum_available=False,raw_or_processed_historical_Kaggle_identity_certified=False,
    independent_GEO_native_PCA_preparation_and_experiments_complete=False))
for item in download['official_metadata']:
    p=ROOT/item['path'];assert sha(p)==item['sha256'];shutil.copyfile(p,packet/p.name)
for name in ['launch.json','aria2c_launch.json','aria2c_exit.json']:
    shutil.copyfile(ROOT/download['state_root']/name,packet/('CITE_'+name))
shutil.copyfile(test,packet/'validation_58_checks.xml')
live={}
for label,root in [('source120',queue['state_root']),('native120',config['state_root'])]:
    state=read(ROOT/root/'status.json');p=psutil.Process(state['pid']);assert p.create_time()==state['create_time'] and p.is_running()
    live[label]=dict(state=state,exact_process_command=p.cmdline())
write(packet/'actual_full120_live_source_and_replay_handles.json',live)
addendum=ROOT/'projects/flow_matching_research/materials/CITE13176_GEO_official_addendum.v1.json'
assert not addendum.exists()
write(addendum,dict(acquisition_configuration=acquisition.CONFIG,evidence=packet.relative_to(ROOT).as_posix(),sources=source_rows,
    all8_downloads_complete=True,all8_full_HDF5_leaf_payloads_read=True,local_integrity_SHA256s_verified=True,
    publisher_checksum_available=False,original_paper_or_Kaggle_processed_identity_certified=False,
    original_author_native_CITE13176_inputs_ready=False,independent_GEO_preprocessing_and_experiments_complete=False,
    existing_released32606_or_independent_multiome13176_sources_overwritten=False))
paths=[acquisition.CONFIG,'scripts/flow_matching/acquire_CITE13176_GEO_official_v1.py','tests/test_fm_CITE13176_GEO_official_acquisition_v1.py']
for index,p in enumerate(paths):shutil.copyfile(ROOT/p,packet/(f'source{index:02d}'+Path(p).suffix))
write(packet/'source_manifest.json',dict(files=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
    native_training_queue=dict(path=native.QUEUE,sha256=sha(ROOT/native.QUEUE)),
    native_replay_configuration=dict(path=config_path,sha256=sha(ROOT/config_path))))
(packet/'README.md').write_text(f'''首个Multiome完整五种子原生复核与CITE13176公开素材

Multiome已发布作者供体32606／CFM／released YAML／留出时间点1，原种子42–46全部完成每例1000epoch、10000实际训练更新、9093参数。五例均检查全部学习张量、共享状态、优化器矩/计数、选择及last检查点和全部原训练产物，再执行完整原生测试。全部36×5=180个标量与原结果逐项完全相同；原容差2e-6未改。源结果中post-test更新计数指选择检查点，实际训练总更新从完整epoch_history取10000，原字段没有被覆写。

主留出W1分别为{', '.join(f"{r['heldout_W1']:.10f}" for r in rows)}。五训练模型均值{primary['mean']:.10f}、样本STD{primary['std']:.10f}，均值95% t区间(df4)[{primary['ci95_low']:.10f},{primary['ci95_high']:.10f}]。全部36项指标的逐种子值及均值/STD/区间都在CSV与JSON中。此区间仅描述一个冻结划分、一个供体、一个留出时间点和一套训练配置的五模型不确定性，不是完整论文表格跨留出均值、跨供体稳健性或异常检测F1。

每例保留完整7264×8544、8544×10933代价几何，五例独立重建777375840个双精度欧式代价；20次原生W1/W2与10次独立完整W1求解均OPTIMAL，目标差均在完整代价扰动界内。两类LP使用同一POT库，独立的是代价构建及重求解，不宣称独立求解器或正式原对偶证明。该作者发布供体32606与论文描述13176仍不相同，历史训练超参数/权重身份未认证；原全120、另150、另90及全部其他实验保留。

另已通过aria2代理实际下载GEO供体13176四天(2/3/4/7)各一份原始10x计数和一份2025处理版MuData，共8文件1780469041字节，全部保存在F盘。每份官方HEAD长度与实际字节相符，所有HDF5叶数据集已逐块完整解码读取，重新计算SHA256并核对逐文件回执。原始与处理版独立保留。官方README提及的md5sum下载名称未取得，未获得发布者校验摘要，故只声明全载荷读取及本地SHA256完整性指纹，不伪称发布者checksum认证。

GEO2025处理版包含细胞/基因质控、1M库大小归一化、对数变换、双细胞过滤及不同RNA/蛋白模态；其身份不能直接等同于论文当年的Kaggle预处理数据。实际shape、文件来源和完整扫描记录随包保存。新的GEO原生输入准备、完整数值对齐及相应模型实验仍未完成；原CITE13176论文输入缺口没有被这些公开文件自动消除。来源：https://ftp.ncbi.nlm.nih.gov/geo/series/GSE305nnn/GSE305370/suppl/ 。

本轮58项相关检查通过；微小split夹具产生的空验证子集警告不对应任何原模型实验。研究总目标45论文/681方法基线消融/273数据任务/702论文标量以及其他所有未完成实验继续，不把一个完整五模型条件和8份下载解释为全部完成。
''',encoding='utf-8',newline='\n')
# Correct project tracking without changing any scientific queue or task scope.
task_path=ROOT/'projects/flow_matching_research/tasks.json';tasks=read(task_path)
for task in tasks['tasks']:
    if task['id']=='FM-001':
        task['material_addenda']=list(dict.fromkeys(task.get('material_addenda',[])+['materials/CITE13176_GEO_official_addendum.v1.json']))
    if task['id']=='FM-006':
        task['evidence']=list(dict.fromkeys(task.get('evidence',[])+[packet.relative_to(ROOT/'projects/flow_matching_research').as_posix()]))
    if task['id']=='FM-008':
        task['status']='in_progress'
        task['evidence']=list(dict.fromkeys(task.get('evidence',[])+[
            'results/2026-10-16/grasp_tau4_seed1106_full46_model_increment_v1',
            'results/2026-10-16/common105_Reflow_SMAP54_seed1103_validation_calibration_v1']))
        task['boundary']='Fixed-model posthoc calibration/error diagnostics are underway;prespecified full algorithm ablations,all training seeds,author-fidelity and fullTAB comparisons remain incomplete.'
write(task_path,tasks)
paths += [addendum.relative_to(ROOT).as_posix(),task_path.relative_to(ROOT).as_posix(),*[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]]
write(ROOT/'tmp/Multiome_full5_CITE_GEO8_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(set(paths))))
print(json.dumps(dict(full_models=5,full_native_metrics_checked=180,independent_cost_entries=full_cost_entries,
    primary_W1_summary=primary,full_CITE_files=8,CITE_bytes=proof['total_bytes'],exact_files=len(set(paths))),ensure_ascii=False))
