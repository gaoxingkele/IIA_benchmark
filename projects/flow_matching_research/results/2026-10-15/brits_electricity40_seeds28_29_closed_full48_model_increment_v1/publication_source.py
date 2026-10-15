"""Archive two completed original BRITS models without rerunning training/inference."""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def git(cwd, *args):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()

# Extract only the frozen NumPy bootstrap function. No model imports or execution.
import numpy as np
module = ast.parse((ROOT / 'scripts/flow_matching/run_saits.py').read_text(encoding='utf-8'))
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'patient_bootstrap')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-bootstrap-only>', 'exec'), namespace)
collector_path = ROOT / 'experiments/runs/fm_saits_completion_replay_queue_v1/status.json'
collector = read(collector_path)
verified_ids = collector['verified_original_ids']
assert len(verified_ids) == len(set(verified_ids)) == 48
ident = lambda seed: f'saits_paper_brits_electricity_author_missing0.4_seed{seed}'
assert all(ident(seed) in verified_ids for seed in [26,27,28,29])
assert ident(30) not in verified_ids
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/brits_electricity40_seeds28_29_closed_full48_model_increment_v1'
assert not folder.exists()

records, originals, configs = [], {}, []
for seed in [28, 29]:
    name = ident(seed)
    model_config_path = ROOT / f'configs/experiments/fm_saits_timeseries_jobs/{name}.json'
    model_config = read(model_config_path)
    source_base = ROOT / model_config['output_root']
    source = read(source_base / 'result.json')
    evidence_base = ROOT / f'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/{name}'
    native_path = evidence_base / 'native_checkpoint_replay_audit.json'
    review_path = evidence_base / 'independent_full_replay_batch_verification.json'
    native, review = read(native_path), read(review_path)
    assert native['full_checkpoint_contents_and_re_inference_passed'] and review['passed']
    assert native['full_models_replayed'] == review['full_native_models_verified'] == 1
    assert native['metric_tolerance'] == review['metric_tolerance'] == dict(rtol=2e-6, atol=2e-6)
    assert review['all_original_grid_coordinates_independently_checked']
    assert review['author_equivalence_certified'] is False and review['strict_TAB_result'] is False
    assert review['additional_model_inference_executed_by_this_review'] is False
    assert review['native_replay_report_sha256'] == sha(native_path)
    n, r = native['records'][0], review['records'][0]
    assert n['id'] == r['id'] == name and r['seed'] == seed and r['method'] == 'brits'
    assert r['full_test_shape'] == [290,100,370] and r['full_test_grid_cells'] == 10730000
    assert r['held_out_coordinates'] == 3537183 and r['months'] == 10
    assert n['model_parameters'] == source['parameters'] == 6993328
    assert n['full_test_batches'] == 3 and n['full_native_batch_size'] == 128
    assert r['actual_full_checkpoint_reinference_completed']
    assert r['full_truth_masks_units_groups_equal'] and r['full_finite_predictions']
    assert r['full_group_statistics_independently_recomputed']
    finished = r['full_training_completion']
    assert finished['full_training_finished'] and finished['author_patience_remaining'] == 0
    assert finished['original_maximum_epochs'] == 10000 and not finished['benchmark_smoke']
    assert finished['next_epoch'] == {28:978,29:1115}[seed]
    assert source['config'] == model_config and source['config_sha256'] == sha(model_config_path)
    assert source['environment']['numpy'] == np.__version__
    assert source['metrics'] == n['original_metrics'] == r['original_metrics']
    for metric in ['mae','rmse','mre']:
        assert np.isclose(n['replayed_metrics'][metric], r['original_metrics'][metric], rtol=2e-6, atol=2e-6)
        assert np.isclose(r['independently_recomputed_metrics'][metric], r['original_metrics'][metric], rtol=2e-6, atol=2e-6)
    pins = {'config':model_config_path, 'result':source_base/'result.json', 'checkpoint':source_base/'best.pt',
            'resume':source_base/'resume.pt', 'original_group_statistics':source_base/'group_statistics.npz',
            'arrays':Path(n['replay_arrays']), 'dataset':ROOT/model_config['dataset_root']/'datasets.h5',
            'data_audit':ROOT/model_config['dataset_root']/'data_audit.json'}
    # Dataset audit file names come from the frozen original data directory.
    if not pins['data_audit'].exists(): pins['data_audit'] = ROOT/model_config['dataset_root']/'audit.json'
    for key,path in pins.items(): assert sha(path) == r['hashes'][key], (name,key,path)
    with np.load(source_base/'group_statistics.npz', allow_pickle=False) as data:
        assert data['statistics'].shape == (10,4)
        intervals = namespace['patient_bootstrap'](data['statistics'], seed + 10000)
    assert intervals == source['month_block_bootstrap_95_ci']
    child = ROOT / f'experiments/runs/fm_saits_completion_replay_queue_v1/children/{name}'
    archive_files = {
        'original_result.json':source_base/'result.json',
        'frozen_model_configuration.json':model_config_path,
        'native_checkpoint_replay_audit.json':native_path,
        'independent_full_replay_batch_verification.json':review_path,
    }
    for stage, flag in [('native','full_native_model_replay_passed'),('review','full_native_batch_independent_review_passed')]:
        stage_config_path = ROOT/f'configs/reproducibility/fm_saits_completion_replay_queue_v1/{name}.{stage}.json'
        cfg = read(stage_config_path)
        report = native if stage == 'native' else review
        assert report['configuration_sha256'] == sha(stage_config_path)
        for pin in cfg['source_receipts']: assert sha(ROOT/pin['path']) == pin['sha256'], pin['path']
        status, receipt = read(child/stage/'status.json'), read(child/stage/'receipt.json')
        assert status['state'] == 'completed' and status['active_pid'] is None and status[flag]
        assert receipt['exit_code'] == 0 and receipt[flag] and not receipt['resource_guard_aborted']
        for namepart in ['receipt.json','status.json','worker.stdout.log','worker.stderr.log']:
            archive_files[f'{stage}_{namepart}'] = child/stage/namepart
        archive_files[f'frozen_{stage}_configuration.json'] = stage_config_path
        configs.append(stage_config_path.relative_to(ROOT).as_posix())
    actual_native_path = Path(n['replay_arrays']).parent/'checkpoint_replay_audit.json'
    assert sha(actual_native_path) == sha(native_path)
    originals[str(seed)] = archive_files
    records.append(dict(id=name, seed=seed, method='BRITS', dataset='Electricity', task='imputation',
        nominal_missing_fraction=0.4, actual_heldout_fraction=3537183/10730000,
        original_metrics=source['metrics'], native_CPU_replay_metrics=n['replayed_metrics'],
        independent_float64_metrics=r['independently_recomputed_metrics'],
        original_month_block_bootstrap_95_ci=intervals,
        bootstrap_exactly_reproduced=True, bootstrap_repetitions=2000, bootstrap_seed=seed+10000,
        bootstrap_groups=10, between_model_five_seed_CI=False,
        training_epochs=finished['next_epoch'], training_updates={28:6846,29:7805}[seed],
        validation_best=source['validation_best'], training_seconds=source['training_seconds'],
        source_environment=source['environment'], source_configuration=model_config_path.relative_to(ROOT).as_posix(),
        parameters=6993328, full_test_shape=r['full_test_shape'], full_test_coordinates=10730000,
        heldout_targets=3537183, original_artifacts={k:dict(path=str(p),sha256=r['hashes'][k]) for k,p in pins.items()},
        metric_tolerance=native['metric_tolerance'], full_checkpoint_and_independent_array_review_passed=True,
        historical_author_pipeline_certified=False, strict_TAB_anomaly_detection_result=False))

xml_path = ROOT / 'tmp/BRITS_Electricity40_seeds28_29_closure_validation_20261015_v1.xml'
suites = list(ET.parse(xml_path).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites) == 37
assert all(int(s.get(k,0)) == 0 for s in suites for k in ['failures','errors','skipped'])
folder.mkdir(parents=True, exist_ok=False)
for seed, files in originals.items():
    (folder/f'seed{seed}').mkdir()
    for name,path in files.items():
        target = folder/f'seed{seed}'/name
        shutil.copyfile(path,target); assert sha(path) == sha(target)
shutil.copyfile(xml_path,folder/'validation_37_checks.xml')
write(folder/'collector_status_snapshot.json',collector)
write(folder/'two_original_models_and_full_replay_metrics.json',dict(records=records,
    original_SAITS_BRITS_models_closed=48, required_original_models=210,
    Electricity_BRITS_nominal40_cohort=dict(required_seeds=[26,27,28,29,30],closed_seeds=[26,27,28,29],pending_seed=30),
    five_model_uncertainty_available=False, author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False))
buf=io.StringIO(newline='')
fields=['method','dataset','task','seed','training_epochs','training_updates','MAE','RMSE','MRE','parameters','test_windows','heldout_targets']
writer=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n'); writer.writeheader()
for row in records:
    writer.writerow({**{k:row[k] for k in fields if k in row},'MAE':row['original_metrics']['mae'],
        'RMSE':row['original_metrics']['rmse'],'MRE':row['original_metrics']['mre'],'test_windows':290})
(folder/'two_original_model_results.csv').write_text(buf.getvalue(),encoding='utf-8',newline='')
(folder/'README.md').write_text('''# BRITS Electricity 40%：新增两个完整模型及原生复核

原 seed28、29 已按固定最大 10,000 epoch、patience30 的停止规则分别完成 978 / 1,115 个 epoch（6,846 / 7,805 次更新）。两个均为 6,993,328 参数完整模型；保存的最佳验证 MAE 检查点已重新执行全部 290×100×370 测试坐标，原生 batch128、3 个批次、3,537,183 个留出目标及 10 个月份组均完整保留。

|原始种子|MAE|RMSE|MRE|
|---|---:|---:|---:|
|28|1.0308663845|1.5044929981|0.5519872308|
|29|1.0276293755|1.5352202654|0.5502539277|

独立进程重新核对全量 HDF5 真值、mask、样本号、月份、有限预测和逐月分组误差；原始结果、模型、resume/优化器/RNG、数据及预测数组的 SHA256 均核验通过。CPU 原生重推理和独立 float64 指标均符合预先冻结的 rtol=atol=2e-6，未放松容差。原训练未保存逐点预测，因此不能声称原始预测逐字节相同。

本轮以相同原 NumPy 环境隔离提取原 bootstrap 函数，使用原 10 个月份分组误差、seed+10000 和 2,000 次抽样，逐值重现两组原 95% 区间。区间表达单模型月份不确定性。此队列已闭合 48/210 个原 SAITS/BRITS 模型，本 BRITS40 组为种子 26–29 的 4/5；seed30 仍训练，不报告不完整的五模型均值或种子置信区间。

论文/代码来源是 SAITS 实验中的 BRITS 基线：固定 WenjieDu/SAITS commit 660b87f19c1277065f314f24134f646229e89ca9。[SAITS 论文](https://arxiv.org/abs/2202.08516)，[Electricity 数据](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)。原名义 40% 有放回遮蔽在测试网格实际留出约 32.965% 坐标。历史每缺失率调参、历史预处理种子仍未完全确定。原始结果内 acquired_not_executed 是下载快照的旧来源标签，不能替代这里的真实训练和原生复核证明。此为插补实验，不能用于严格 TAB 异常检测排名，也未认证历史论文端到端等价。

37 项相关验证通过。原配置、原结果、原生及独立全量证明、各次成功退出码和资源保护状态均保留原字节。大型模型和数组保留原磁盘位置，索引保存路径及 SHA256。没有新增训练种子、重试失败训练或更改运行中的模型。至少 45 篇 / 681 方法记录 / 273 数据任务及后来新增实验仍在范围内，完整研究目标未完成。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    archived_original_models=2, full_native_test_grid_per_model=10730000, all_heldout_targets_per_model=3537183,
    tests_passed=37, original_bootstrap_intervals_exactly_reproduced=True,
    bootstrap_numpy=np.__version__, original_proofs_preserved=True,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/brits_Electricity40_seeds28_29_full_native_closed.2026-10-15_v1.json'
write(ara,dict(method='BRITS',dataset='Electricity',nominal_missing_fraction=0.4,original_seeds=[28,29],
    completed_full_epochs=[978,1115],full_native_model_inference_and_independent_array_review=True,
    evidence=folder.relative_to(ROOT).as_posix(), original_SAITS_BRITS_closed=48, original_SAITS_BRITS_required=210,
    cohort_closed_seeds=[26,27,28,29],cohort_pending_seeds=[30],five_model_CI_available=False,
    actual_month_block_bootstrap_intervals_exactly_reproduced=True, strict_TAB_result=False,
    current_goal_turn_classification='progress_two_original_BRITS_models_native_and_independent_closure_published',
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),*configs]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=ROOT/'tmp/BRITS_Electricity40_seeds28_29_publication_manifest_20261015_v1.json'
write(manifest,dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(new_original_models=2,closed_original_models=48,required_models=210,
    exact_publication_files=len(paths),manifest=manifest.relative_to(ROOT).as_posix(),tests_passed=37)))
