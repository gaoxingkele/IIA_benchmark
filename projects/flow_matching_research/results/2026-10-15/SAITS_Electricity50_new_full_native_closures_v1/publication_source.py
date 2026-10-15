"""Archive actual original models and all existing native/independent proofs."""
import csv
from datetime import datetime,timezone
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write

stamp=datetime.now(timezone.utc).isoformat()
state=read(ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json')
owner=psutil.Process(state['pid']);assert owner.create_time()==state['create_time']
assert state['required_original_models']==210
verified=state['verified_original_ids'];assert len(verified)==len(set(verified))>=51
identity=lambda seed:f'saits_paper_saits_electricity_author_missing0.5_seed{seed}'
seeds=[seed for seed in [26,27,28,29,30] if identity(seed) in verified]
assert seeds[:2]==[26,27]
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/SAITS_Electricity50_new_full_native_closures_v1'
assert not folder.exists()
memo={}
def digest(path):
    path=Path(path).resolve();stat=path.stat();key=(stat.st_size,stat.st_mtime_ns)
    if path in memo:
        old,value=memo[path];assert old==key,'Immutable completed input changed: '+str(path)
        return value
    value=sha(path);after=path.stat();assert key==(after.st_size,after.st_mtime_ns)
    memo[path]=(key,value);return value

bootstrap_code=r'''
import ast,json,sys
from pathlib import Path
import numpy as np
root=Path.cwd();tree=ast.parse((root/'scripts/flow_matching/run_saits.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patient_bootstrap')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-original-month-bootstrap>','exec'),namespace)
with np.load(sys.argv[1],allow_pickle=False) as data:
 assert data['statistics'].shape==(10,4)
 result=namespace['patient_bootstrap'](data['statistics'],int(sys.argv[2])+10000)
print(json.dumps(dict(numpy=np.__version__,intervals=result)))
'''
records=[];archives={};configs=[]
for seed in seeds:
    name=identity(seed);config_path=ROOT/f'configs/experiments/fm_saits_timeseries_jobs/{name}.json';model=read(config_path)
    source_base=ROOT/model['output_root'];source_path=source_base/'result.json';source=read(source_path)
    assert source['status']=='completed' and source['config']==model and source['config_sha256']==digest(config_path)
    assert (model['method'],model['dataset'],model['missing_ratio'],model['seed'])==('saits','electricity',.5,seed)
    assert source['parameters']==11511000 and source['target_count']==4221734 and source['test_windows']==290
    base=ROOT/f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{name}.native.json'
    review_config_path=base.with_name(name+'.review.json');nc,rc=read(base),read(review_config_path)
    for config in [nc,rc]:
        for pin in config['source_receipts']:assert digest(ROOT/pin['path'])==pin['sha256'],pin['path']
    native_path=Path(nc['artifact_directory'])/'checkpoint_replay_audit.json'
    review_path=ROOT/rc['output_directory']/'independent_full_replay_batch_verification.json'
    native,review=read(native_path),read(review_path)
    assert native['configuration_sha256']==digest(base) and review['configuration_sha256']==digest(review_config_path)
    assert native['full_checkpoint_contents_and_re_inference_passed'] and review['passed']
    assert native['full_models_replayed']==review['full_native_models_verified']==1
    assert native['metric_tolerance']==review['metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert review['native_replay_report_sha256']==digest(native_path) and review['all_original_grid_coordinates_independently_checked']
    assert not review['additional_model_inference_executed_by_this_review']
    n,r=native['records'][0],review['records'][0]
    assert n['id']==r['id']==name and r['seed']==seed and r['method']=='saits'
    assert n['model_parameters']==11511000 and n['full_native_batch_size']==128 and n['full_test_batches']==3
    assert r['full_test_shape']==[290,100,370] and r['full_test_grid_cells']==10730000
    assert r['held_out_coordinates']==4221734 and r['months']==10
    assert all(r[key] for key in ['actual_full_checkpoint_reinference_completed','full_truth_masks_units_groups_equal',
        'full_finite_predictions','full_group_statistics_independently_recomputed'])
    finish=r['full_training_completion']
    assert finish['full_training_finished'] and not finish['benchmark_smoke'] and finish['original_maximum_epochs']==10000
    assert finish['author_patience_remaining']==0 or finish['next_epoch']==10000
    assert source['metrics']==n['original_metrics']==r['original_metrics']
    for metric in ['mae','rmse','mre']:
        assert math.isclose(n['replayed_metrics'][metric],source['metrics'][metric],rel_tol=2e-6,abs_tol=2e-6)
        assert math.isclose(r['independently_recomputed_metrics'][metric],source['metrics'][metric],rel_tol=2e-6,abs_tol=2e-6)
    dataset=ROOT/model['dataset_root'];audit_path=dataset/'data_audit.json'
    if not audit_path.exists():audit_path=dataset/'audit.json'
    audit=read(audit_path)
    pins=dict(config=config_path,result=source_path,checkpoint=Path(n['checkpoint_path']),arrays=Path(n['replay_arrays']),
        dataset=dataset/'datasets.h5',resume=source_base/'resume.pt',original_group_statistics=source_base/'group_statistics.npz',data_audit=audit_path)
    assert {key:digest(path) for key,path in pins.items()}==r['hashes']
    assert digest(source_path)==n['result_sha256'] and digest(pins['checkpoint'])==n['checkpoint_sha256']
    assert audit['splits']['train']['shape']==[816,100,370] and audit['splits']['val']['shape']==[291,100,370]
    assert audit['total_samples']==1397 and not audit['paper_sample_count_matches']
    boot=subprocess.run([nc['python'],'-B','-X','utf8','-c',bootstrap_code,str(pins['original_group_statistics']),str(seed)],
        cwd=ROOT,capture_output=True,text=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert boot.returncode==0,boot.stdout+boot.stderr
    boot=json.loads(boot.stdout);assert boot['numpy']==source['environment']['numpy']=='1.23.5'
    assert boot['intervals']==source['month_block_bootstrap_95_ci']
    original_guard=Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts')/(name+'.json')
    g=read(original_guard);assert g['exit_code']==0 and g['full_result_validated'] and not g['resource_guard_aborted']
    archive=dict(original_result=source_path,frozen_model_configuration=config_path,original_training_guard=original_guard,
        data_audit=audit_path,native_checkpoint_replay_audit=native_path,independent_full_replay_batch_verification=review_path)
    for phase,cfg_path,cfg,flag in [('native',base,nc,'full_native_model_replay_passed'),
        ('review',review_config_path,rc,'full_native_batch_independent_review_passed')]:
        phase_root=ROOT/cfg['state_root'];status=read(phase_root/'status.json');receipt=read(phase_root/'receipt.json')
        assert status['state']=='completed' and status['active_pid'] is None and status[flag]
        assert receipt['exit_code']==0 and receipt[flag] and not receipt['resource_guard_aborted']
        archive['frozen_'+phase+'_configuration']=cfg_path;archive[phase+'_guard']=phase_root/'receipt.json';archive[phase+'_status']=phase_root/'status.json'
        configs.append(cfg_path.relative_to(ROOT).as_posix())
    archives[str(seed)]=archive
    records.append(dict(id=name,method='SAITS',dataset='Electricity',task='imputation',seed=seed,
        nominal_missing_fraction=.5,actual_test_heldout_fraction=4221734/10730000,
        completed_epoch_counter=finish['next_epoch'],source_maximum_epochs=10000,source_early_stop_patience=30,
        full_training_completion=finish,source_best_validation=source['validation_best'],
        cumulative_training_seconds_as_recorded=source['training_seconds'],parameters=11511000,
        original_metrics=source['metrics'],native_CPU_replay_metrics=n['replayed_metrics'],independent_float64_metrics=r['independently_recomputed_metrics'],
        month_block_bootstrap_95_ci=boot['intervals'],month_block_bootstrap_repetitions=2000,bootstrap_seed=seed+10000,months=10,
        original_bootstrap_exactly_reproduced_with_source_numpy=True,full_test_shape=[290,100,370],full_test_grid_cells=10730000,heldout_targets=4221734,
        metric_tolerance=native['metric_tolerance'],current_artifact_identities={key:dict(path=str(path),sha256=digest(path)) for key,path in pins.items()},
        source_environment=source['environment'],protocol_boundaries=source['protocol_boundaries'],
        citation=model['citation'],data_citation=model['data_citation'],
        full_native_and_independent_review_passed=True,original_prediction_bitwise_equivalence_certified=False,
        author_historical_pipeline_certified=False,strict_TAB_anomaly_detection_result=False))

assert len({r['current_artifact_identities']['dataset']['sha256'] for r in records})==1
xml=ROOT/'tmp/SAITS_Electricity50_seeds26_27_closure_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==37
assert all(int(s.get(key,0))==0 for s in suites for key in ['failures','errors','skipped'])
folder.mkdir(parents=True,exist_ok=False)
for seed,files in archives.items():
    target=folder/('seed'+seed);target.mkdir()
    for label,path in files.items():shutil.copyfile(path,target/(label+'.json'));assert digest(path)==sha(target/(label+'.json'))
shutil.copyfile(xml,folder/'validation_37_checks.xml');write(folder/'collector_status_snapshot.json',state)
write(folder/'full_original_models_and_complete_grid_metrics.json',dict(captured_utc=stamp,records=records,
    verified_original_training_seeds=seeds,required_original_training_seeds=[26,27,28,29,30],
    missing_original_training_seeds=[s for s in [26,27,28,29,30] if s not in seeds],
    between_training_seed_five_model_CI_certified=False,full_five_model_summary_pending=True,
    current_collector_verified_original_models=len(verified),required_original_models=210,
    archive_performs_additional_training=False,archive_performs_model_reinference=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),all_original_experiments_complete=False))
buffer=io.StringIO(newline='');fields=['method','dataset','task','seed','completed_epoch_counter','MAE','RMSE','MRE','parameters','test_grid_cells','heldout_targets']
writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writeheader()
for row in records:
    writer.writerow({**{key:row[key] for key in fields if key in row},'MAE':row['original_metrics']['mae'],
        'RMSE':row['original_metrics']['rmse'],'MRE':row['original_metrics']['mre'],'test_grid_cells':10730000})
(folder/'all_archived_model_metrics.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
lines=['# SAITS / Electricity / 名义50%缺失：新增完整训练及全量复核','',
    '每例11,511,000参数，原最大10,000 epoch/patience30停止规则已满足；最佳验证检查点完成全部290×100×370测试网格（3个batch128/最后尾批）、4,221,734留出坐标及10个月份的原生重推理和独立全量复核。原rtol=atol=2e-6保留。模型、resume/优化器/RNG、数据、真值/mask/单位/月份、全预测和分组统计的当前SHA256均核对。','',
    '|原训练种子|停止后的epoch计数|MAE|RMSE|MRE|','|---|---:|---:|---:|---:|']
for r in records:
    m=r['original_metrics'];lines.append(f"|{r['seed']}|{r['completed_epoch_counter']}|{m['mae']:.10g}|{m['rmse']:.10g}|{m['mre']:.10g}|")
lines += ['',f'本组已独立复核{len(seeds)}/5个原模型（{seeds}）；主收集器当前{len(verified)}/210项。原模型10个月份分组误差、seed+10000和2,000次bootstrap在原NumPy1.23.5下逐值重现95%区间，随JSON保存。它是固定模型的月份不确定性，未当作训练种子不确定性；全部5个原种子的汇总仍待单独核验。','',
    '名义50%有放回遮蔽实际留出39.34514445%的测试坐标。原作者月份划分按较早test、中间validation、较晚train；1397个序列与论文1400不一致，预处理种子及各缺失率历史调参未确定。保留原数据控制边界，不能称严格因果/TAB异常检测或历史论文端到端等价。原训练没有保留逐点预测，故不声称原始预测逐字节一致。原下载/数据审计的旧状态标签不覆盖本包的实际完成证明。','',
    '本轮37项相关检查通过。保存原字节配置、原结果、原生/独立证明和三阶段真实保护回执。大型模型/数组保留原磁盘位置和哈希；未重复训练或重推理已完成模型。原45篇/681方法消融记录/273数据任务及新增实验全部继续，完整研究目标未完成。']
(folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=stamp,new_original_models=len(records),tests_passed=37,
    complete_original_training_native_inference_and_independent_review_passed=True,
    files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/SAITS_Electricity50_full_native_closures.2026-10-15_v1.json'
write(ara,dict(method='SAITS',dataset='Electricity',nominal_missing_fraction=.5,seeds=seeds,
    evidence=folder.relative_to(ROOT).as_posix(),status='FULL_TRAINING_NATIVE_INFERENCE_AND_INDEPENDENT_REVIEW_PASSED',
    required_seeds=[26,27,28,29,30],between_model_five_seed_CI_certified=False,
    current_SAITS_BRITS_verified=len(verified),required_original_models=210,strict_TAB_result=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),*configs];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/SAITS_Electricity50_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(new_models=len(records),seeds=seeds,full_reviewed_models=len(verified),required_original_models=210,tests_passed=37,exact_files=len(paths))))
