"""Publish a completed original model using immutable full replay evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching.prepare_saits_physio import sha

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'saits_paper_brits_electricity_author_missing0.5_seed28'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

nc = f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{IDENTITY}.native.json'
rc = f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{IDENTITY}.review.json'
native_config, review_config = read(ROOT/nc), read(ROOT/rc)
for config in [native_config, review_config]:
    for pin in config['source_receipts']: assert sha(ROOT/pin['path']) == pin['sha256'], pin['path']
native_path = Path(native_config['artifact_directory'])/'checkpoint_replay_audit.json'
review_path = ROOT/review_config['output_directory']/'independent_full_replay_batch_verification.json'
native, review = read(native_path), read(review_path)
assert native['configuration_sha256'] == sha(ROOT/nc)
assert review['configuration_sha256'] == sha(ROOT/rc)
assert review['native_replay_report_sha256'] == sha(native_path)
assert native['full_checkpoint_contents_and_re_inference_passed'] and native['full_models_replayed'] == 1
assert not native['cuda_context_initialized'] and review['passed'] and review['full_native_models_verified'] == 1
assert review['all_original_grid_coordinates_independently_checked']
n,r = native['records'][0],review['records'][0]
assert n['id'] == r['id'] == IDENTITY and r['seed'] == 28
assert r['full_test_shape'] == [290,100,370] and r['held_out_coordinates'] == 4221734 and r['months'] == 10
assert all(r[k] for k in ['full_truth_masks_units_groups_equal','full_finite_predictions',
    'full_group_statistics_independently_recomputed','actual_full_checkpoint_reinference_completed'])
source = read(n['result_path']); model_config = read(ROOT/n['config'])
assert source['config'] == model_config and source['status'] == 'completed'
assert (model_config['method'], model_config['dataset'], model_config['seed'], model_config['missing_ratio']) == ('brits','electricity',28,.5)
assert source['parameters'] == n['model_parameters'] == 6993328
assert source['validation_best']['epoch'] == 736
completion = r['full_training_completion']
assert completion['full_training_finished'] and not completion['benchmark_smoke']
assert completion['original_maximum_epochs'] == 10000 and completion['next_epoch'] == 768 and completion['author_patience_remaining'] == 0
assert native['metric_tolerance'] == review['metric_tolerance'] == dict(rtol=2e-6,atol=2e-6)
output,data = ROOT/model_config['output_root'], ROOT/model_config['dataset_root']
artifacts = dict(config=ROOT/n['config'],result=Path(n['result_path']),checkpoint=Path(n['checkpoint_path']),
    arrays=Path(n['replay_arrays']),dataset=data/'datasets.h5',resume=output/'resume.pt',
    original_group_statistics=output/'group_statistics.npz',data_audit=data/'data_audit.json')
assert {k:sha(p) for k,p in artifacts.items()} == r['hashes']
sources = dict(original_result=Path(n['result_path']),original_model_configuration=ROOT/n['config'],
    frozen_native_configuration=ROOT/nc,frozen_review_configuration=ROOT/rc,
    actual_native_reinference=native_path,actual_independent_full_coordinate_review=review_path)
for phase, config, flag in [('native',native_config,'full_native_model_replay_passed'),('review',review_config,'full_native_batch_independent_review_passed')]:
    base=ROOT/config['state_root']; guard,status=read(base/'receipt.json'),read(base/'status.json')
    assert guard['exit_code'] == 0 and guard[flag] and not guard['resource_guard_aborted']
    assert status['state'] == 'completed' and status['active_pid'] is None
    sources[phase+'_guard']=base/'receipt.json'; sources[phase+'_terminal_status']=base/'status.json'
original_guard=Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts')/(IDENTITY+'.json')
guard=read(original_guard)
assert guard['exit_code'] == 0 and guard['full_result_validated'] and not guard['resource_guard_aborted']
sources['actual_original_training_guard']=original_guard
collector_path=ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json'
collector=read(collector_path)
assert IDENTITY in collector['verified_original_ids'] and len(collector['verified_original_ids']) >= 57
assert psutil.Process(collector['pid']).create_time() == collector['create_time']
original_state_path=original_guard.parent.parent/'status.json'
original_state=read(original_state_path)
assert psutil.Process(original_state['pid']).create_time() == original_state['create_time']
assert original_state['active_job']=='saits_paper_brits_electricity_author_missing0.5_seed29'

bootstrap_code = r'''
import ast,json,sys
from pathlib import Path
import numpy as np
node=next(n for n in ast.parse((Path.cwd()/'scripts/flow_matching/run_saits.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='patient_bootstrap')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-bootstrap>','exec'),ns)
with np.load(sys.argv[1],allow_pickle=False) as arrays:
 assert arrays['statistics'].shape==(10,4)
 print(json.dumps(dict(numpy=np.__version__,intervals=ns['patient_bootstrap'](arrays['statistics'],10028))))
'''
check=subprocess.run([native_config['python'],'-B','-X','utf8','-c',bootstrap_code,str(artifacts['original_group_statistics'])],
    cwd=ROOT,capture_output=True,text=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert check.returncode == 0,check.stdout+check.stderr
bootstrap=read_value=json.loads(check.stdout)
assert bootstrap['numpy']=='1.23.5' and bootstrap['intervals']==source['month_block_bootstrap_95_ci']
validation=ROOT/'tmp/BRITS_Electricity50_seed28_full57_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==35
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
sources['validation_35_checks']=validation

other_id='saits_paper_saits_electricity_author_missing0.5_seed28'
matched_path=ROOT/'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1'/other_id/'independent_full_replay_batch_verification.json'
matched=read(matched_path)
assert matched['passed'] and matched['full_native_models_verified']==1
m=matched['records'][0]
assert m['hashes']['dataset']==r['hashes']['dataset'] and m['held_out_coordinates']==r['held_out_coordinates']
assert m['full_test_shape']==r['full_test_shape'] and m['seed']==28
comparison=dict(shared_dataset_sha256=r['hashes']['dataset'],seed=28,
    SAITS=m['independently_recomputed_metrics'],BRITS=r['independently_recomputed_metrics'],
    BRITS_minus_SAITS={k:r['independently_recomputed_metrics'][k]-m['independently_recomputed_metrics'][k] for k in ['mae','rmse','mre']},
    independent_five_seed_comparison_complete=False,strict_TAB_result=False)
sources['matched_SAITS_independent_review']=matched_path
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/BRITS_Electricity50_seed28_full57_closed_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in sources.items():
    target=folder/(name+('.xml' if path.suffix=='.xml' else '.json'))
    shutil.copyfile(path,target); assert sha(target)==sha(path)
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),id=IDENTITY,seed=28,method='BRITS',dataset='Electricity',
    nominal_missing_ratio=.5,actual_heldout_fraction=4221734/10730000,full_grid_cells=10730000,held_out_coordinates=4221734,
    parameter_count=6993328,next_epoch=768,best_epoch_source_index=736,original_training_guard_passed=True,
    full_native_reinference_and_independent_coordinate_review_passed=True,
    independent_metrics=r['independently_recomputed_metrics'],original_month_block_bootstrap_95_ci=source['month_block_bootstrap_95_ci'],
    original_bootstrap_exact_recomputation=bootstrap,month_groups=10,training_seed_uncertainty_complete=False,
    artifact_paths_and_hashes={k:dict(path=str(p),sha256=sha(p)) for k,p in artifacts.items()},
    source_proof_hashes={k:sha(p) for k,p in sources.items()},metric_tolerance=review['metric_tolerance'],
    source_training_seconds=source['training_seconds'],verified_original_SAITS_BRITS_models=len(collector['verified_original_ids']),
    required_original_SAITS_BRITS_models=210,BRITS50_verified_seeds=[26,27,28],BRITS50_required_seeds=[26,27,28,29,30],
    additional_training_seeds=0,author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
write(folder/'actual_full_closure_summary.json',summary)
write(folder/'matched_seed28_SAITS_BRITS_comparison.json',comparison)
write(folder/'source_and_collector_status_snapshot.json',dict(source=original_state,collector=collector))
(folder/'README.md').write_text('# BRITS：Electricity 名义50%缺失率，seed28完整闭环\n\n原6,993,328参数模型按冻结的最大10000 epoch、patience30及验证集选择规则自然结束：next_epoch=768，最佳epoch源索引736。源训练、完整检查点CPU重推理、独立新进程全坐标复核及三阶段资源守卫均通过，未缩减预算、重训或替换数据。\n\n实际测试网格为290×100×370，共10,730,000个坐标、4,221,734个留出目标和10个月份组。独立复算MAE=1.0519082629504706、RMSE=1.5655934200907982、MRE=0.563139411732428。作者有放回掩码抽样使实际留出比例约39.345%，不能解释为无放回精确50%；这些是原标准化单位的插补误差，不是异常检测F1。\n\n模型、数据、恢复状态、原指标、原生预测与分组统计重新核验哈希；原NumPy1.23.5、seed10028的月份bootstrap区间重新执行并与原报告完全一致，原2e-6容差未改。该区间描述单模型月份敏感性，不替代5个独立训练种子的不确定性。\n\n另附相同数据与留出目标下的SAITS/BRITS seed28配对结果。BRITS该缺失率已闭环seed26、27、28；29、30仍待完成，不能据此宣称完整五种子比较。原210项队列当前至少57项通过完整复核，seed29由原队列接续。\n\n35项相关验证通过。论文历史逐缺失率调参、预处理种子及1397/1400序列差异仍未完全对齐，不认证原论文端到端等价或完整TAB结果。总体研究任务尚未完成。\n\n来源：[SAITS论文及其BRITS对比](https://arxiv.org/abs/2202.08516)、[Electricity数据](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)。\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(tests_passed=35,all_full_artifact_hashes_current=True,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/BRITS_Electricity50_seed28_full57_closed.2026-10-16_v1.json'
write(ara,dict(id=IDENTITY,evidence=folder.relative_to(ROOT).as_posix(),complete_original_model_reinference=True,
    independently_reviewed_full_grid=True,verified_original_models=len(collector['verified_original_ids']),required_original_models=210,
    five_seed_cohort_complete=False,strict_TAB_result=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),nc,rc,review_config['output_directory']]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(p for p in subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines() if not p.endswith('.log'))
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/BRITS_Electricity50_seed28_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),verified_models=len(collector['verified_original_ids']),metrics=r['independently_recomputed_metrics'],comparison=comparison)))
