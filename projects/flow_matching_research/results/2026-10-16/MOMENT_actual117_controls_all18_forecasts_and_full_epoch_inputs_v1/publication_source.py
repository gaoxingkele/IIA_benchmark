"""Current complete117 controls,18forecast inputs and80M-index proofs."""
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import prepare_moment_icm_classification_train_only_v1 as control
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as forecast
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as epoch

ROOT,read,sha,write=control.ROOT,control.read,control.sha,control.write
cc=read(ROOT/control.CONFIG);control.verify(cc)
fc=read(ROOT/forecast.CONFIG);forecast.verify(fc)
ec=read(ROOT/epoch.CONFIG);epoch.verify(ec)
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/MOMENT_actual117_controls_all18_forecasts_and_full_epoch_inputs_v1'
assert not folder.exists()
files={};control_records=[];file_count=0
for job in cc['jobs']:
    name=job['dataset'];base=Path(cc['artifact_root'])/name;p=base/'preprocessing.json';r=read(p)
    guard_path=ROOT/cc['state_root']/'attempts'/name/'guard_receipt.json';g=read(guard_path)
    assert r['passed'] and r['job']==job and r['configuration_sha256']==sha(ROOT/control.CONFIG)
    assert r['strict_train_only_standardization'] and r['test_scaler_fit_calls']==0
    assert r['unchanged_TRAIN']['all_original_train_coordinates_checked']
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
    assert g['exact_child_handle']['command']==[cc['python'],'-B','-X','utf8','-u','-m',control.MODULE,'--config',control.CONFIG,'--dataset',name]
    for split,directory,record in [('TRAIN',Path(r['unchanged_TRAIN']['directory']),r['unchanged_TRAIN']),('TEST',base/'TEST',r['TEST'])]:
        for f,h in record['files_sha256'].items():assert sha(directory/f)==h;file_count+=1
    for split in ['TRAIN','TEST']:assert sha(ROOT/job['splits'][split]['path'])==job['splits'][split]['sha256']
    comp=r['complete_TEST_comparison_to_source_transductive']
    assert comp['labels.npy']['changed_coordinates']==comp['original_instance_rows.npy']['changed_coordinates']==0
    control_records.append(dict(dataset=name,TRAIN_instances=job['splits']['TRAIN']['instances'],TEST_instances=job['splits']['TEST']['instances'],
        checked_TRAIN_value_coordinates=r['unchanged_TRAIN']['checked_value_coordinates'],
        changed_TEST_value_coordinates=comp['values.npy']['changed_coordinates'],maximum_TEST_value_difference=comp['values.npy']['maximum_absolute_difference'],
        changed_TEST_mask_coordinates=comp['input_mask.npy']['changed_coordinates'],original_unscaled_exception=r['original_no_normalization_exception'],
        TEST_native_grid_adjusted=r['test_native_grid_adjustment'] is not None))
    files['classification/'+name+'/actual_control_proof.json']=p;files['classification/'+name+'/actual_control_guard.json']=guard_path
forecast_path=Path(fc['artifact_root'])/forecast.REPORT;fr=read(forecast_path)
fg=read(ROOT/fc['state_root']/'guard_receipt.json')
assert fr['passed'] and fr['configuration_sha256']==sha(ROOT/forecast.CONFIG) and fr['datasets']==6 and fr['horizon_dataset_pairs']==18 and fr['role_horizon_groups']==54
assert fg['passed'] and fg['exit_code']==0 and not fg['resource_guard_aborted'] and fg['proof_sha256']==sha(forecast_path)
forecast_rows=[]
for record in fr['source_records']:
    j=record['job'];assert sha(ROOT/j['path'])==j['sha256']
    assert record['training_scaler_fixed_across_all_roles'] and not record['forecast_target_roles_overlap']
    for role,r in record['roles'].items():
        for key,pin in r['files'].items():assert sha(Path(fc['artifact_root'])/j['dataset']/role/(key+'.npy'))==pin['sha256'];file_count+=1
        for h in r['windows_by_horizon']:
            assert h['every_window_bound_checked']
            forecast_rows.append(dict(dataset=j['dataset'],role=role,horizon=h['horizon'],channels=r['channels'],windows=h['windows'],
                input_coordinates=h['input_coordinates'],target_coordinates=h['target_coordinates'],
                target_raw_row_start=h['source_target_row_start'],target_raw_row_end_exclusive=h['source_target_row_end_exclusive']))
assert len(forecast_rows)==54
epoch_path=Path(ec['artifact_root'])/'full_PILE_epoch_plan_proof.json';er=read(epoch_path)
eg=read(ROOT/ec['state_root']/'guard_receipt.json')
assert er['passed'] and er['configuration_sha256']==sha(ROOT/epoch.CONFIG)
assert eg['passed'] and eg['exit_code']==0 and not eg['resource_guard_aborted'] and eg['proof_sha256']==sha(epoch_path)
assert er['count_audit']['selected_source_references']==2086 and er['count_audit']['total_items']==80850189
closed=er['index_closure'];assert closed['passed'] and closed['complete_epoch_batches']==39479 and closed['complete_epoch_items']==80850189
assert closed['every_partial_batch_retained'] and closed['all_original_source_items_exactly_once']
plan_path=Path(er['epoch_plan_file']);assert sha(plan_path)==er['epoch_plan_sha256'];plan=read(plan_path)
for p in [*plan['permutation_files'].values(),plan['batch_order_file']]:assert sha(plan_path.parent/p['file'])==p['sha256'];file_count+=1
files.update(actual_all6_forecast_input_proof=forecast_path,actual_forecast_guard=ROOT/fc['state_root']/'guard_receipt.json',
    actual_full_epoch_index_proof=epoch_path,actual_full_epoch_plan=plan_path,actual_epoch_guard=ROOT/ec['state_root']/'guard_receipt.json',
    frozen117_control_configuration=ROOT/control.CONFIG,frozen6_forecast_configuration=ROOT/forecast.CONFIG,frozen_full_epoch_configuration=ROOT/epoch.CONFIG)
checks={}
for label,p in [('control','tmp/MOMENT_ICM_full117_train_only_validation_20261015_v1.xml'),('forecast','tmp/MOMENT_ICM_forecast_input_all18_validation_20261015_v3.xml'),('epoch','tmp/MOMENT_ICM_disk_and_full_epoch_validation_20261015_v2.xml')]:
    suites=list(ET.parse(ROOT/p).getroot().iter('testsuite'));assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
    checks[label]=sum(int(x.get('tests',0)) for x in suites);files['unchanged_'+label+'_validation.xml']=ROOT/p
folder.mkdir(parents=True,exist_ok=False)
for relative,path in files.items():
    if not relative.endswith(('.json','.xml')):relative+='.json'
    target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);assert sha(path)==sha(target)
for name,rows in [('all117_control_differences.csv',control_records),('all54_role_horizon_counts.csv',forecast_rows)]:
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (folder/name).write_text(stream.getvalue(),encoding='utf-8',newline='')
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_control_datasets=117,actual_forecast_datasets=6,actual_forecast_horizon_pairs=18,
    actual_forecast_role_horizon_groups=54,epoch_items=80850189,epoch_batch_slots=39479,selected_sources=2086,
    current_generated_or_referenced_array_hashes_rechecked=file_count,
    control_datasets_with_changed_TEST_values=sum(r['changed_TEST_value_coordinates']>0 for r in control_records),
    control_datasets_with_native_grid_adjustment=sum(r['TEST_native_grid_adjusted'] for r in control_records),
    control_TEST_value_coordinates_changed=sum(r['changed_TEST_value_coordinates'] for r in control_records),
    all_TRAIN_source_coordinates_retained=True,TEST_scaler_fit_calls=0,unchanged_validation_tests=checks,
    actual_neural_training_or_SVM_performed=False,full2086_input_content_and_random_mask_audit_complete=False,
    all_original_experiments_complete=False)
write(folder/'actual_completed_input_wave_summary.json',summary)
(folder/'README.md').write_text(f'''# Tiny／ICM：117归一化对照、18预测输入和完整单轮索引实际完成

三条登记队列均已自然结束且所有科学worker成功退出、无资源中止。本包重新核对117个逐案数据证明及保护回执、全部TRAIN引用和TEST新数组哈希、六个预测源原文件及54个角色/horizon计数、完整单轮置换/批次顺序文件的哈希。共重核{file_count}个生成或引用数组文件；大数组继续保留原F盘。

分类对照完成117/117，全部官方TRAIN/TEST实例、通道、标签、行序保留，TEST标准化器fit次数0。完整逐坐标TRAIN复核已由实际worker完成，新旧TEST差异随逐案证明及CSV归档。实际有{summary['control_datasets_with_changed_TEST_values']}个数据集TEST值改变，{summary['control_datasets_with_native_grid_adjustment']}个采用明确的TRAIN原始长度对齐。来源不标准化例外保持；不声称所有变长/插值步骤都完全归纳式。还没有分类准确率或SVM结果。

六个预测源、三个horizon96/192/384、三个角色共54组全部准备通过。源下游70/10/20与预训练60/10/30分开；ETT源固定月份、256历史上下文和cubic全CSV插值边界均保留。每组窗口数、坐标数和目标原CSV行范围随CSV保存；没有新的预测MSE/MAE。

完整单轮选择2086来源、80,850,189项、39,479个批次槽，保留1219和74两个尾批。width1/8置换及批次顺序均已实际证明完整双射和每项恰好一次，原RandomState抽取顺序保留。这是索引准备，原证明明确没有读取全部训练内容或训练模型；不能据此认定80M项已被模型消费。全2086输入内容绑定和随机mask审计仍须另行完成，之后才允许两个完整预训练模型执行。

复用未变冻结源码的来源检查：{checks}；实际数据阶段真实成功回执与当前全文件哈希另外提供。没有重跑已完成队列、修改资源锁或放宽内存/预算。至少45篇/681方法记录/273数据任务、702论文标量目标及全部新增任务继续保留，研究目标未完成。

来源：[ICM论文](https://arxiv.org/abs/2409.13530v1)、[Time Series PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)、[Autoformer](https://github.com/thuml/Autoformer)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8');shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(**summary,files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_actual117_controls18forecasts_full_epoch_inputs.2026-10-16_v1.json'
write(ara,dict(actual117_control_datasets_complete=True,actual18_forecast_input_pairs_complete=True,actual80850189_epoch_index_complete=True,
    complete2086_content_and_mask_audits_not_claimed=True,evidence=folder.relative_to(ROOT).as_posix(),neural_benchmark_results_produced=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_actual_three_input_closures_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(**summary,exact_files=len(paths))))
