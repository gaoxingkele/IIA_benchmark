from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import csv
import json
import shutil
import subprocess
from scripts.flow_matching import publish_native_condition_catalog_v55 as p

ROOT=p.ROOT
base=ROOT/'projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v55'
closure=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_full12_v1'
identity='crossad_ablation__row5__gecco__seed1103'
binding=p.read(closure/'all12_proof_bindings.json')['records'][identity]
for field in ['full_array_proof','independent_metric_report']:
    assert p.sha(binding[field+'_path'])==binding[field+'_sha256']
guard=p.read(closure/'independent_metric_guard.json')
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert guard['report_sha256']==binding['independent_metric_report_sha256']
proof=p.read(binding['independent_metric_report_path'])
assert p.read(closure/'actual_original_training_budget_verification.json')['actual_completed_epochs']==20
with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
after=deepcopy(before);changed=[];captured=datetime.now(timezone.utc).isoformat()
for i,row in enumerate(after):
    try:recipe=json.loads(row['score_recipe'])
    except (ValueError,TypeError):continue
    if not isinstance(recipe,dict) or recipe.get('canonical_id')!=identity:continue
    assert row['n']=='0' and not row['mean'] and row['required_repeats']=='1'
    ratio=row['protocol'].split('/validation_',1)[1].split('percent',1)[0]
    field={'F1':'f1','AUROC':'auroc','AP':'average_precision'}.get(row['metric'],row['metric'])
    row.update(n='1',mean=repr(proof['independent_strict_metrics'][ratio]['micro'][field]),
        source=binding['independent_metric_report_path'],captured_utc=captured,
        comparison='full_condition_verified_one_checkpoint_not_a_training_seed_cohort')
    changed.append(i)
assert len(changed)==15
allowed={'n','mean','source','captured_utc','comparison'}
for i,(old,new) in enumerate(zip(before,after)):
    if i not in changed:assert old==new
    else:assert all(old[k]==new[k] for k in old if k not in allowed)
with (ROOT/'projects/flow_matching_research/results/2026-10-12/all_algorithm_dataset_metrics_v54/all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    legacy=list(csv.DictReader(f))
p.validate(legacy,after,12,6)
assert len(after)==10991 and sum(bool(r['mean']) for r in after)==2335
config_path=ROOT/'configs/reproducibility/fm_native_condition_catalog_row5.2026-10-16_v56.json'
inputs=[base/'all_algorithm_dataset_metrics.csv',closure/'all12_proof_bindings.json',
    closure/'independent_metric_guard.json',closure/'actual_original_training_budget_verification.json',
    Path(binding['independent_metric_report_path']),Path(binding['full_array_proof_path']),
    ROOT/(p.MODULE.replace('.','/')+'.py'),ROOT/p.TEST]
config=dict(base_catalog=base.relative_to(ROOT).as_posix(),canonical_id=identity,
    original_scope=p.read(base/'publication_validation.json')['original_scope'],
    source_receipts=[dict(path=str(x),sha256=p.sha(x)) for x in inputs],
    changed_fields=sorted(allowed),changed_positions=changed,
    new_complete_benchmark_conditions=1,new_training_seed_slots_from_publication=0,
    all_original_experiments_complete=False)
assert not config_path.exists()
config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
output=ROOT/'projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v56'
output.mkdir(exist_ok=False)
(output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
with (output/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(after)
page=p.interactive(after,headers)
page=page.replace('<title>实验指标 v55</title>','<title>实验指标 v56</title>')
page=page.replace('<h1>算法—数据集—指标 v55</h1>','<h1>算法—数据集—指标 v56</h1>')
(output/'results.html').write_text(page,encoding='utf-8')
(output/'README.md').write_text('''# 算法—数据集—指标 v56

总计10991行，2335行有数值。保留v55全部槽位，只将CrossAD row5/GECCO/seed1103的3个验证阈值×5指标共15个单元格从待完成更新为已完整核验。其余10976行、捕获时间和所有原单位、协议、预算、论文值、不确定性及认证字段不变。

新增的是一个完整条件，不是15次实验。该条件完成20轮、8640个训练批次；全量检查点重放、原始完整测试点/尾部及标准库独立指标审计均通过。它使用subseries与global-context，关闭multi-scale与cross-scale；原精确消融架构未公开，仍为论文重建实现。

CrossAD累计12/139条件；五种重建消融均仅完成seed1103，每种1/5，完整主要五种子组仍1/22。生成重跑保持6/70，完整五种子组1/14。固定检查点的n=1/required_repeats=1不是完整五/十种子组；评估器内部重复仍只在evaluator_*列中，训练种子SD/SE/CI没有伪造。

原v54全部8626行和旧捕获时间继续保留；其他近期独立结果包仍需逐轨核验合并。空白包含未运行和已完成但未定义的指标，不能据此计算未完成实验数。全部45论文/681方法记录/273数据任务/702既有论文指标及原论文/TAB/新数据的完整目标未完成。
''',encoding='utf-8')
status=dict(captured_utc=captured,configuration=config_path.relative_to(ROOT).as_posix(),
    configuration_sha256=p.sha(config_path),metric_rows=10991,numeric_rows=2335,
    changed_positions=changed,changed_metric_cells=15,all_other10976_rows_and_times_unchanged=True,
    CrossAD_verified_conditions=12,CrossAD_required_conditions=139,
    spectral_verified_conditions=6,spectral_required_conditions=70,
    new_model_training_or_inference_from_publication=False,all_original_experiments_complete=False)
(output/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
shutil.copyfile(__file__,output/'publication_source.py')
for path in inputs:assert p.sha(path)==next(x['sha256'] for x in config['source_receipts'] if x['path']==str(path))
for i in changed:
    row=after[i];ratio=row['protocol'].split('/validation_',1)[1].split('percent',1)[0]
    field={'F1':'f1','AUROC':'auroc','AP':'average_precision'}.get(row['metric'],row['metric'])
    assert float(row['mean'])==proof['independent_strict_metrics'][ratio]['micro'][field]
(output/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(output).as_posix(),sha256=p.sha(x))
    for x in sorted(output.rglob('*')) if x.is_file()]),indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/native_condition_catalog.2026-10-16_v56.json'
assert not ara.exists()
ara.write_text(json.dumps(dict(result_packet=output.relative_to(ROOT).as_posix(),metric_rows=10991,
    numeric_rows=2335,new_complete_benchmark_conditions=1,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
manifest_path=ROOT/'tmp/CrossAD_row5_full12_publication_manifest_20261016_v1.json'
manifest=p.read(manifest_path)
manifest['paths']=sorted(set(manifest['paths'])|{config_path.relative_to(ROOT).as_posix(),
    ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst'}|
    {x.relative_to(ROOT).as_posix() for x in output.rglob('*') if x.is_file()})
target=ROOT/'tmp/CrossAD_row5_full12_catalog_v56_manifest.json';assert not target.exists()
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),rows=10991,numeric_rows=2335,changed_metric_cells=15)))
