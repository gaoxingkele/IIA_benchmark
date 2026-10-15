"""Close all five original Electricity40 BRITS models with actual full proofs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import summarize_saits_full_five_seed_cohort_v1 as stats
from scripts.flow_matching import audit_saits_full_checkpoint_replay_v1 as native_module

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
from scripts.flow_matching.prepare_saits_physio import sha
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

validation=ROOT/'tmp/BRITS_Electricity40_all5_cohort_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==9
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
rows=[];evidence=[];files={};configs=[]
for seed in stats.SEEDS:
    name=f'saits_paper_brits_electricity_author_missing0.4_seed{seed}'
    nc=f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{name}.native.json'
    rc=f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{name}.review.json'
    ncfg,rcfg=read(ROOT/nc),read(ROOT/rc)
    for cfg in [ncfg,rcfg]:
        for pin in cfg['source_receipts']:assert sha(ROOT/pin['path'])==pin['sha256'],pin['path']
    npath=Path(ncfg['artifact_directory'])/'checkpoint_replay_audit.json'
    rpath=ROOT/rcfg['output_directory']/'independent_full_replay_batch_verification.json'
    native,review=read(npath),read(rpath)
    assert native['full_checkpoint_contents_and_re_inference_passed'] and native['full_models_replayed']==1
    assert review['passed'] and review['full_native_models_verified']==1 and review['all_original_grid_coordinates_independently_checked']
    assert native['metric_tolerance']==review['metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert review['native_replay_report_sha256']==sha(npath) and not native['cuda_context_initialized']
    n,r=native['records'][0],review['records'][0]
    assert n['id']==r['id']==name and r['seed']==seed
    cfg=read(ROOT/n['config']);source=read(Path(n['result_path']))
    assert cfg['method']=='brits' and cfg['dataset']=='electricity' and cfg['missing_ratio']==.4 and cfg['seed']==seed
    assert cfg['variant']=='author' and not cfg['corrected_masking'] and not cfg['diagnostic']
    assert source['status']=='completed' and source['config']==cfg and source['parameters']==n['model_parameters']==6993328
    assert n['full_test_shape']==r['full_test_shape']==[290,100,370]
    assert source['test_windows']==290 and source['target_count']==r['held_out_coordinates']==n['held_out_coordinates']==3537183
    assert source['bootstrap_groups']==r['months']==10 and source['bootstrap_group']=='month'
    done=r['full_training_completion'];assert done['full_training_finished'] and done['original_maximum_epochs']==10000 and not done['benchmark_smoke']
    assert done['next_epoch']>=10000 or done['author_patience_remaining']==0
    assert r['actual_full_checkpoint_reinference_completed'] and r['full_truth_masks_units_groups_equal'] and r['full_group_statistics_independently_recomputed']
    base=ROOT/cfg['output_root'];data=ROOT/cfg['dataset_root']
    pins=dict(config=ROOT/n['config'],result=Path(n['result_path']),checkpoint=Path(n['checkpoint_path']),arrays=Path(n['replay_arrays']),
        resume=base/'resume.pt',original_group_statistics=base/'group_statistics.npz',dataset=data/'datasets.h5',data_audit=data/'data_audit.json')
    actual_hashes={key:sha(path) for key,path in pins.items()};assert actual_hashes==r['hashes']
    archive=dict(original_result=base/'result.json',frozen_original_model_configuration=ROOT/n['config'],
        frozen_native_configuration=ROOT/nc,frozen_independent_configuration=ROOT/rc,native_full_replay=npath,independent_full_review=rpath)
    for stage,cfg_stage,flag in [('native',ncfg,'full_native_model_replay_passed'),('review',rcfg,'full_native_batch_independent_review_passed')]:
        b=ROOT/cfg_stage['state_root'];receipt=read(b/'receipt.json');status=read(b/'status.json')
        assert receipt['exit_code']==0 and receipt[flag] and not receipt['resource_guard_aborted']
        assert status['state']=='completed' and status['active_pid'] is None
        archive[stage+'_guard']=b/'receipt.json';archive[stage+'_terminal_status']=b/'status.json'
    sg=Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts')/(name+'.json')
    guard=read(sg);assert guard['exit_code']==0 and guard['full_result_validated'] and not guard['resource_guard_aborted']
    archive['actual_original_training_guard']=sg
    files[str(seed)]=archive;configs += [nc,rc]
    rows.append(dict(seed=seed,method='brits',dataset='electricity',nominal_missing_ratio=.4,dataset_sha256=actual_hashes['dataset'],
        original_metrics=source['metrics'],independent_metrics=r['independently_recomputed_metrics'],
        actual_native_CPU_replay_metrics=n['replayed_metrics'],full_training_completion=done,
        validation_best=source['validation_best'],training_seconds=source['training_seconds'],
        original_month_block_bootstrap_95_ci=source['month_block_bootstrap_95_ci'],full_test_shape=r['full_test_shape'],
        heldout_targets=3537183,source_artifacts={k:dict(path=str(path),sha256=actual_hashes[k]) for k,path in pins.items()}))
    evidence.append(dict(id=name,native_proof_sha256=sha(npath),independent_review_sha256=sha(rpath),
        original_training_guard_sha256=sha(sg),all_original_values_kept=True))
summary=stats.summarize(rows)
assert summary['full_five_seed_cohort_complete'] and summary['independently_trained_models']==5
collector=read(ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json')
assert all(f'saits_paper_brits_electricity_author_missing0.4_seed{s}' in collector['verified_original_ids'] for s in stats.SEEDS)
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/BRITS_Electricity40_all_five_original_models_closed_v1'
folder.mkdir(parents=True,exist_ok=False)
for seed,mapping in files.items():
    (folder/f'seed{seed}').mkdir()
    for name,path in mapping.items():
        target=folder/f'seed{seed}'/(name+'.json');shutil.copyfile(path,target);assert sha(path)==sha(target)
shutil.copyfile(validation,folder/'validation_9_cohort_checks.xml')
shutil.copyfile(ROOT/'scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py',folder/'frozen_five_seed_summary_source.py')
write(folder/'full_five_seed_result_and_uncertainty.json',dict(summary=summary,original_models=rows,full_native_evidence=evidence,
    exact_data_grid_coordinates_per_model=10730000,actual_heldout_fraction=3537183/10730000,
    full_original_SAITS_BRITS_models_reviewed=len(collector['verified_original_ids']),required_original_models=210,
    new_training_seed=False,author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False))
write(folder/'collector_status_snapshot.json',collector)
lines=['# BRITS Electricity名义40%：五个原模型全量闭合\n',
    '原种子26..30均完成冻结最大10000 epoch/patience30训练规则，完整6993328参数最佳验证MAE检查点实际重新推理全部290×100×370测试网格、3537183留出目标和10个月份组。五个均有真实原训练保护回执、完整原生重推理和第二独立全坐标验证成功证明，保持原2e-6相对/绝对容差。大型checkpoint、数据及预测数组保留原磁盘位置，路径/哈希随包登记。\n',
    '|指标|五模型均值|样本标准差|Student t95%区间（df4）|\n|---|---:|---:|---|']
for key,s in summary['summaries'].items():lines.append(f"|{key.upper()}|{s['mean']:.9f}|{s['std']:.9f}|[{s['ci95_low']:.9f},{s['ci95_high']:.9f}]|")
lines += ['\n这是同一冻结数据划分上五个独立训练模型的描述性小样本区间，使用样本SD、SE及df4 t分位数；原NumPy/SciPy统计和独立Python标量归约/固定df4分位数核对一致。每个模型的10个月份block-bootstrap区间另行保留，不把月份当作独立训练种子。没有删除表现较差的第30种子，也没有用部分五模型计算区间。\n',
    '原名义40%采用有放回遮蔽，实际留出3537183/10730000≈32.965%。保留此来源规则；该任务为插补，MAE/RMSE是来源标准化单位，MRE为比例，不能与异常检测F1混排。原预测未保存，本次CPU原生与原指标在冻结容差内一致，不声称原预测逐字节相同；原历史每缺失率调参与精确预处理种子未完全确定，未认证作者论文端到端等价或TAB结果。\n',
    '9项五模型统计门禁检查通过，完整输入、参数、early-stop/optimizer/RNG、原生网格/真值/mask/月份的验证由随包实际全量证明提供。本组五个原种子已全部闭合，非新训练种子；完整210模型队列目前至少49项通过，另外工作以实际进程和完整证明继续。至少45篇/681方法记录/273数据任务及新增实验范围不变，全部研究目标未完成。\n',
    '来源为[SAITS论文中的BRITS基线](https://arxiv.org/abs/2202.08516)，[Electricity原数据](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)；固定作者代码commit660b87f19c1277065f314f24134f646229e89ca9。']
(folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_models=5,new_model_closures=1,
    tests_passed=9,source_artifacts_full_hashes_verified=True,
    files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/BRITS_Electricity40_all_five_original_models_closed.2026-10-15_v1.json'
write(ara,dict(method='BRITS',dataset='Electricity',nominal_missing_ratio=.4,original_training_seeds=list(stats.SEEDS),
    full_native_replay_and_independent_review_models=5,complete_five_seed_summary=summary,evidence=folder.relative_to(ROOT).as_posix(),
    required_original_SAITS_BRITS_models=210,new_training_seed=False,author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),*configs];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/BRITS_Electricity40_all5_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(full_original_models=5,full_SAITS_BRITS_queue_closed=len(collector['verified_original_ids']),tests=9,
    means={k:v['mean'] for k,v in summary['summaries'].items()},exact_publication_files=len(paths))))
