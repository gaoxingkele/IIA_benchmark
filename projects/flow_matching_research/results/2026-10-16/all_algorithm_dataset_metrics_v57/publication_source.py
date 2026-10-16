"""Advance nineteen verified metric cells and preserve all other catalog rows."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
from scripts.flow_matching import publish_native_condition_catalog_v55 as p

ROOT = p.ROOT
base = ROOT / 'projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v56'
cross = ROOT / 'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1104_full13_v1'
spectral = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy43_full7_actual_handoff_v2'
config = deepcopy(p.read(ROOT / p.CONFIG))
config.update(cross_frozen_bindings=(cross/'all13_proof_bindings.json').relative_to(ROOT).as_posix(),
    cross_frozen_cohorts=(cross/'all22_primary_cohort_completion.json').relative_to(ROOT).as_posix(),
    spectral_frozen_catalog=(spectral/'all7_complete_generator_reexecution_catalog.json').relative_to(ROOT).as_posix(),
    spectral_frozen_cohorts=(spectral/'all14_five_seed_cohort_completion.json').relative_to(ROOT).as_posix())
config['completed_cross_ids'] = sorted(p.read(ROOT/config['cross_frozen_bindings'])['records'])
config['completed_spectral_ids'] = sorted(p.read(ROOT/config['spectral_frozen_catalog'])['records'])
jobs, cr, cases, sr, cc, sc = p.independent_evidence(config)
assert len(cr) == 13 and len(sr) == 7 and cc['complete_primary_cohorts'] == sc['complete_cohorts'] == 1
with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f); before = list(reader); headers = reader.fieldnames
captured = datetime.now(timezone.utc).isoformat()
fresh = p.cross_rows(headers, jobs, cr, captured, config) + p.spectral_rows(headers, cases, sr, captured, config)
fresh = {p.key(r): r for r in fresh}
cross_id = 'crossad_ablation__row1__gecco__seed1104'
spectral_id = 'spectral_original__diffusion_ts__energy__released_author_data__seed43'
after = deepcopy(before)
allowed = {'n', 'mean', 'source', 'captured_utc', 'comparison', 'evaluator_repeats', 'evaluator_std', 'evaluator_se', 'evaluator_ci95'}
changed = []
for i, row in enumerate(after):
    try:
        recipe = json.loads(row['score_recipe'])
    except (ValueError, TypeError):
        continue
    if not isinstance(recipe, dict) or recipe.get('canonical_id') not in {cross_id, spectral_id}:
        continue
    assert row['n'] == '0' and not row['mean'] and row['required_repeats'] == '1'
    expected = fresh[p.key(row)]
    assert expected['n'] == '1' and expected['mean']
    for field in row:
        if field not in allowed:
            assert row[field] == expected[field], (i, field)
    row.update({k: expected[k] for k in allowed})
    changed.append(i)
assert len(changed) == 19
for i, (old, new) in enumerate(zip(before, after)):
    if i not in changed:
        assert old == new
    else:
        assert all(old[k] == new[k] for k in old if k not in allowed)
with (ROOT/'projects/flow_matching_research/results/2026-10-12/all_algorithm_dataset_metrics_v54/all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
    legacy = list(csv.DictReader(f))
p.validate(legacy, after, 13, 7)
assert len(after) == 10991 and sum(bool(x['mean']) for x in after) == 2354
inputs = [base/'all_algorithm_dataset_metrics.csv', ROOT/config['cross_frozen_bindings'],
    ROOT/config['cross_frozen_cohorts'], ROOT/config['spectral_frozen_catalog'], ROOT/config['spectral_frozen_cohorts'],
    ROOT/(p.MODULE.replace('.', '/')+'.py'), ROOT/p.TEST]
for name in ['full_array_proof', 'independent_metric_report']:
    inputs.append(Path(p.read(ROOT/config['cross_frozen_bindings'])['records'][cross_id][name+'_path']))
entry = p.read(ROOT/config['spectral_frozen_catalog'])['records'][spectral_id]
inputs.extend([Path(entry['report_path']), Path(entry['guard_path'])])
config_path = ROOT/'configs/reproducibility/fm_native_condition_catalog_CrossAD13_Spectral7.2026-10-16_v57.json'
config.update(base_catalog=base.relative_to(ROOT).as_posix(), output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v57',
    changed_fields=sorted(allowed), changed_positions=changed,
    source_receipts=[dict(path=str(x), sha256=p.sha(x)) for x in inputs],
    new_complete_checkpoint_conditions_in_catalog=2, new_model_training_or_inference_from_publication=False,
    all_original_experiments_complete=False)
assert not config_path.exists()
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
output = ROOT/config['output_directory']; output.mkdir(exist_ok=False)
(output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n', encoding='utf-8')
with (output/'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader(); writer.writerows(after)
page = p.interactive(after, headers).replace('实验指标 v55', '实验指标 v57').replace('算法—数据集—指标 v55', '算法—数据集—指标 v57')
(output/'results.html').write_text(page, encoding='utf-8')
(output/'README.md').write_text('''# 算法—数据集—指标 v57

总计10991行，2354行有数值。本次仅推进CrossAD row1/GECCO/seed1104的15个指标槽位，以及Diffusion-TS/Energy/seed43完整生成重跑的4个槽位，共19行。其余10972行和原捕获时间逐字段保持。原v54全部8626行及旧单位、论文值、认证和不确定性字段不变。

CrossAD累计13/139个完整GPU条件；row1完成2/5种子，其余四种重建消融各1/5，完整主要五种子组仍1/22。原精确作者消融架构未公开，继续标为论文重建。三个阈值来自验证集，严格大于且无PA；没有混入测试标签oracle或完整TAB结果。

生成全量重跑7/70，Energy完成2/5训练模型，完整五训练种子组仍1/14。每个条件的五次评估重复只写入evaluator_*，训练种子SD/SE/CI保持空白；历史采样RNG未知，不声称历史生成字节或作者性能认证。零新增生成器训练更新。

相关完整条件报告、守卫、源/重放全数组及独立指标再次核验。调度交接已在Energy43结束后实际完成，保留全部成功和失败记录，只退出两个空闲调度进程，未停止数值进程；交接不计作新增实验。

指标行数不是训练次数。固定检查点n=1/required_repeats=1只代表一个完整条件，不能替代五/十训练种子组。空白还包含已完成但未定义指标，不能拿空白数计算未完成实验数。其他算法按原捕获时间保留，近期独立结果包仍需逐轨合并。45论文/681方法记录/273数据任务/702既有指标及原论文、TAB、新数据的全部目标尚未完成。
''', encoding='utf-8')
status = dict(captured_utc=captured, configuration=config_path.relative_to(ROOT).as_posix(), configuration_sha256=p.sha(config_path),
    original_scope=config['original_scope'], metric_rows=10991, numeric_rows=2354,
    changed_positions=changed, changed_metric_cells=19, all_other10972_rows_and_times_unchanged=True,
    CrossAD_verified_conditions=13, CrossAD_required_conditions=139, spectral_verified_conditions=7, spectral_required_conditions=70,
    complete_crossad_primary_cohorts=1, complete_spectral_five_seed_cohorts=1,
    new_model_training_or_inference_from_publication=False, all_original_experiments_complete=False)
(output/'publication_validation.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
shutil.copyfile(__file__, output/'publication_source.py')
for entry in config['source_receipts']:
    assert p.sha(entry['path']) == entry['sha256']
(output/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(output).as_posix(), sha256=p.sha(x))
    for x in sorted(output.rglob('*')) if x.is_file()]), indent=2)+'\n', encoding='utf-8')
ara = ROOT/'projects/flow_matching_research/ara/execution/native_condition_catalog.2026-10-16_v57.json'
assert not ara.exists()
ara.write_text(json.dumps(dict(result_packet=output.relative_to(ROOT).as_posix(), metric_rows=10991, numeric_rows=2354,
    new_complete_checkpoint_conditions_in_catalog=2, new_model_training_or_inference_from_publication=False,
    all_original_experiments_complete=False), indent=2)+'\n', encoding='utf-8')
readme = ROOT/'projects/flow_matching_research/README.rst'
text = readme.read_text(encoding='utf-8')
assert 'v56' in text and '2335行有数值' in text
text = text.replace('v56', 'v57').replace('2335行有数值', '2354行有数值')
text = text.replace('已完整核验的12个CrossAD条件和6个生成重跑条件', '已完整核验的13个CrossAD条件和7个生成重跑条件')
readme.write_text(text, encoding='utf-8', newline='\n')
manifests = [p.read(ROOT/'tmp/CrossAD_row1_seed1104_full13_manifest_20261016_v1.json'),
             p.read(ROOT/'tmp/Spectral_Energy43_full7_actual_handoff_manifest_20261016_v2.json')]
assert manifests[0]['previous_root'] == manifests[1]['previous_root']
assert manifests[0]['previous_parent'] == manifests[1]['previous_parent']
paths = set().union(*(set(m['paths']) for m in manifests))
paths |= {config_path.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(), readme.relative_to(ROOT).as_posix()}
paths |= {x.relative_to(ROOT).as_posix() for x in output.rglob('*') if x.is_file()}
target = ROOT/'tmp/CrossAD13_Spectral7_catalog_v57_manifest_20261016_v1.json'
assert not target.exists()
target.write_text(json.dumps(dict(previous_root=manifests[0]['previous_root'], previous_parent=manifests[0]['previous_parent'],
    paths=sorted(paths)), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(files=len(paths), rows=len(after), numeric_rows=2354, changed_metric_cells=19)))
