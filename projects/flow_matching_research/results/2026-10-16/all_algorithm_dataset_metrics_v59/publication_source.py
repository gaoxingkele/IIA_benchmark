from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import csv
import json
import shutil
import subprocess
from scripts.flow_matching import publish_native_condition_catalog_v55 as p

ROOT=p.ROOT
base=ROOT/'projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v58'
closure=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row3_seed1104_full15_v1'
identity='crossad_ablation__row3__gecco__seed1104'
binding=p.read(closure/'all15_proof_bindings.json')['records'][identity]
for field in ['full_array_proof','independent_metric_report']:
    assert p.sha(binding[field+'_path'])==binding[field+'_sha256']
guard=p.read(closure/'independent_metric_guard.json')
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert guard['report_sha256']==binding['independent_metric_report_sha256']
proof=p.read(binding['independent_metric_report_path'])
assert p.read(closure/'actual_original_training_budget_verification.json')['actual_completed_epochs']==19
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
p.validate(legacy,after,15,7)
assert len(after)==10991 and sum(bool(r['mean']) for r in after)==2384
config_path=ROOT/'configs/reproducibility/fm_native_condition_catalog_row3.2026-10-16_v59.json'
inputs=[base/'all_algorithm_dataset_metrics.csv',closure/'all15_proof_bindings.json',
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
output=ROOT/'projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v59'
output.mkdir(exist_ok=False)
(output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
with (output/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(after)
page=p.interactive(after,headers)
page=page.replace('<title>实验指标 v55</title>','<title>实验指标 v59</title>')
page=page.replace('<h1>算法—数据集—指标 v55</h1>','<h1>算法—数据集—指标 v59</h1>')
(output/'results.html').write_text(page,encoding='utf-8')

(output/'README.md').write_text('''# 算法—数据集—指标 v59

总计10991行，2384行有数值。保留v58全部原槽位、单位、论文值、协议和认证字段，只更新CrossAD row3/GECCO/seed1104三个验证阈值×五指标共15个单元格；其余10976行与原捕获时间逐字段保持。新增的是一个完整训练条件，不是15次训练。

row3启用multi-scale、cross-scale，关闭subseries、global-context；精确作者消融架构未公开，继续标为论文重建。实际19轮、8208个完整训练批次，best验证轮为16，按原patience3早停；原20轮上限不变。全量源/检查点数组逐字节重放、全部原始点和尾部、独立标准库指标审计通过。

CrossAD累计15/139条件；row1、row2和row3各完成2/5种子，其余两种消融各1/5，主要五种子完整组仍1/22。生成重跑保持7/70、完整五训练种子组1/14。固定检查点n=1不代表五/十训练种子组；生成评估器内部重复只写evaluator_*，没有伪造训练种子SD/SE/CI。

原v54全部8626行继续保留。近期TAB完整54实体指标对照、TranAD源码全109预处理、单细胞等独立证据包尚未并入本表，不把旧捕获时间写成最新全项目状态。空白包含未完成和已执行但指标未定义，不能用空白数计算未运行实验。全部45论文/681方法记录/273数据任务/702既有标量及原论文、TAB、新数据完整目标未完成。
''',encoding='utf-8')
status=dict(captured_utc=captured,configuration=config_path.relative_to(ROOT).as_posix(),configuration_sha256=p.sha(config_path),
    original_scope=config['original_scope'],metric_rows=10991,numeric_rows=2384,changed_positions=changed,changed_metric_cells=15,
    all_other10976_rows_and_times_unchanged=True,CrossAD_verified_conditions=15,CrossAD_required_conditions=139,
    spectral_verified_conditions=7,spectral_required_conditions=70,new_model_training_or_inference_from_publication=False,
    all_original_experiments_complete=False)
(output/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
shutil.copyfile(__file__,output/'publication_source.py')
for path in inputs:
    assert p.sha(path)==next(x['sha256'] for x in config['source_receipts'] if x['path']==str(path))
for i in changed:
    row=after[i];ratio=row['protocol'].split('/validation_',1)[1].split('percent',1)[0]
    field={'F1':'f1','AUROC':'auroc','AP':'average_precision'}.get(row['metric'],row['metric'])
    assert float(row['mean'])==proof['independent_strict_metrics'][ratio]['micro'][field]
(output/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(output).as_posix(),sha256=p.sha(x))
    for x in sorted(output.rglob('*')) if x.is_file()]),indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/native_condition_catalog.2026-10-16_v59.json';assert not ara.exists()
ara.write_text(json.dumps(dict(result_packet=output.relative_to(ROOT).as_posix(),metric_rows=10991,numeric_rows=2384,
    new_complete_checkpoint_conditions_in_catalog=1,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')

manifest=p.read(ROOT/'tmp/CrossAD_row3_seed1104_full15_manifest_20261016_v1.json')
paths=set(manifest['paths'])|{config_path.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst'}
paths|={x.relative_to(ROOT).as_posix() for x in output.rglob('*') if x.is_file()}
target=ROOT/'tmp/CrossAD15_catalog_v59_manifest_20261016_v1.json';assert not target.exists()
target.write_text(json.dumps(dict(previous_root=manifest['previous_root'],previous_parent=manifest['previous_parent'],paths=sorted(paths)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths),rows=10991,numeric_rows=2384,changed_metric_cells=15)))
