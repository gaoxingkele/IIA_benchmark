"""Classify every actual TAB21 nonfinite cell using full saved scores/reference."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
import csv
import shutil
import subprocess
import numpy as np
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as tab
from scripts.flow_matching.audit_completed_undefined_affiliation_v1 import review_empty_prediction
from scripts.flow_matching.tab_range_metrics import load_reference
from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest

root=tab.ROOT
configuration='configs/reproducibility/fm_TAB21_actual_undefined_affiliation.2026-10-16_v1.json'
out=root/'projects/flow_matching_research/results/2026-10-16/TAB21_actual_undefined_affiliation_full_reference_v1'
read,sha=tab.read,tab.sha


def registration():
    tc=read(root/tab.CONFIG)
    source=root/tab.TAB/'ts_benchmark/evaluation/metrics'
    paths=[tab.CONFIG,'scripts/flow_matching/run_TAB_full878_metric_suite_v1.py',
        'scripts/flow_matching/audit_completed_undefined_affiliation_v1.py',
        'scripts/flow_matching/tab_range_metrics.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py']
    paths.extend(str(p.relative_to(root)) for p in (source/'affiliation').glob('*.py'))
    paths.extend(str((source/n).relative_to(root)) for n in ['classification_metrics_label.py','utils.py','vus_metrics.py'])
    return dict(original_TAB_configuration=tab.CONFIG,case_key='e8a311b6a8a4968fb22a',
        first_guarded_entities=21,expected_metric_calls=16884,expected_nonfinite_cells=832,
        expected_undefined_metrics=['affiliation_f','affiliation_precision'],
        all_original_entity_test_points_required=True,
        audit_source_snapshot=str((out/'review_source.py').relative_to(root)),
        audit_source_sha256=sha(Path(__file__)),
        source_receipts=[dict(path=p,sha256=sha(root/p)) for p in paths],
        original_scope=tc['original_scope'],output_directory=str(out.relative_to(root)),
        boundary='Existing full-source/native/review completion and actual guarded first21/54entities only. '
                 'Reuse frozen validation and diagnostic pooledTRAIN+TEST thresholds;no selection or repair. '
                 'Replay exact original affiliation functions on every original point for every undefined cell. '
                 'No new model inference/training or full TAB certification or leaderboard statistics.')


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


assert not (root/configuration).exists() and not out.exists()
config=registration()
write(root/configuration,config)
tc=read(root/config['original_TAB_configuration'])
entry=next(e for e in tc['entries'] if e['key']==config['case_key'])
bound=tab.ready_case(tc,entry)
assert bound is not None and len(bound['dataset']['entities'])==54
assert sha(bound['lineage']['native_scores'])==bound['lineage']['native_scores_sha256']
for pin in config['source_receipts']:
    assert sha(root/pin['path'])==pin['sha256']
dataset=bound['dataset']
label_sources={policy:record for policy,record in dataset['labels_only_controls'].items()}
for record in label_sources.values():
    assert sha(record['path'])==record['sha256']
reference=load_reference(root/tab.TAB/'ts_benchmark/evaluation/metrics')
levels=tab.thresholds(tc,bound['result'])
all_cells=Counter()
reviewed=[]
fragments=[]
threshold_review=[]
with np.load(bound['lineage']['native_scores'],allow_pickle=False) as scores:
    validation=np.concatenate([scores[e['array_prefix']+'_validation'] for e in dataset['entities']])
    training=np.concatenate([scores[e['array_prefix']+'_train'] for e in dataset['entities']])
    testing=np.concatenate([scores[e['array_prefix']+'_test'] for e in dataset['entities']])
    pools={'validation':validation,'diagnostic_pooled_TEST':np.concatenate([training,testing])}
    for level,threshold in levels.items():
        origin,ratio=level.split('/')
        actual=float(np.percentile(pools[origin],100-float(ratio),method='linear'))
        assert actual==threshold,(level,actual,threshold)
        threshold_review.append(dict(level=level,frozen_threshold=threshold,
            independently_recomputed_threshold=actual,points=len(pools[origin]),
            threshold_uses_TEST_scores=origin=='diagnostic_pooled_TEST',threshold_uses_TEST_labels=False))
    for index in range(config['first_guarded_entities']):
        prefix='e%03d'%index
        path,proof=tab.checked_fragment(tc,entry,prefix)
        assert proof['source_lineage']==bound['lineage']
        actual_scores=scores[prefix+'_test']
        assert actual_scores.ndim==1 and np.isfinite(actual_scores).all()
        assert proof['source_scores_digest']==array_digest(actual_scores)
        attempt,_=tab.paths(tc,entry,prefix)
        fragments.append(dict(path=str(path),sha256=sha(path),guard_sha256=sha(attempt/'guard_receipt.json')))
        for policy,cells in proof['metric_results'].items():
            with np.load(label_sources[policy]['path'],allow_pickle=False) as archive:
                labels=archive[prefix+'_labels']
            assert labels.shape==actual_scores.shape and array_digest(labels)==proof['label_digests'][policy]
            all_cells.update(cell['status'] for cell in cells.values())
            for metric,cell in cells.items():
                if cell['status']=='finite':
                    continue
                level,field=metric.rsplit('/',1)
                assert cell['status']=='nonfinite' and field in config['expected_undefined_metrics']
                if field=='affiliation_f':
                    continue
                saved={f:cells[level+'/'+f]['value'] for f in ['affiliation_f','affiliation_precision','affiliation_recall']}
                assert cells[level+'/affiliation_f']['status']=='nonfinite'
                assert cells[level+'/affiliation_recall']['status']=='finite'
                record=review_empty_prediction(labels,actual_scores,levels[level],
                    dict(status='undefined_reference_nan',**saved),reference)
                # The reused helper's old metadata label was validation-specific.
                # Preserve its exact numeric work while identifying both origins.
                record['strict_comparison']='score > unchanged frozen original threshold'
                record['threshold_origin']=level.split('/')[0]
                reviewed.append(dict(entity=proof['entity'],array_prefix=prefix,label_policy=policy,
                    threshold_level=level,classified_nonfinite_cells=2,
                    original_cell_statuses={f:cells[level+'/'+f]['status'] for f in saved},**record))
assert sum(all_cells.values())==config['expected_metric_calls']
assert all_cells['nonfinite']==config['expected_nonfinite_cells'] and all_cells['error']==0
assert sum(r['classified_nonfinite_cells'] for r in reviewed)==all_cells['nonfinite']
assert len(reviewed)==416
assert all(r['predicted_events']==0 and r['actual_reference_events']>0 for r in reviewed)
for pin in config['source_receipts']:
    assert sha(root/pin['path'])==pin['sha256']
assert tab.ready_case(tc,entry)['lineage']==bound['lineage']
out.mkdir(parents=True,exist_ok=False)
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=configuration,
    configuration_sha256=sha(root/configuration),case=entry['case'],required_full_entities=54,
    guarded_entities_audited=21,metric_calls_checked=sum(all_cells.values()),metric_outcomes=dict(all_cells),
    complete_undefined_reference_reviews=len(reviewed),nonfinite_cells_classified=832,
    unclassified_nonfinite_cells=0,original_reference_functions_executed=3*len(reviewed),
    frozen_thresholds_independently_recomputed=threshold_review,full_point_reviews=reviewed,
    source_lineage=bound['lineage'],fragment_bindings=fragments,
    numeric_outputs_repaired_or_coerced=False,original_results_modified=False,
    model_training_or_checkpoint_inference_executed=False,full_TAB_case_completed=False,
    reason='Every nonfinite cell is original affiliation precision/F with zero predicted events and actual ground truth events; '
           'the exact untouched reference returns NaN/NaN/0 for precision/F/recall. No missing evaluation or model inference failure.',
    all_original_experiments_complete=False)
write(out/'full_actual_nonfinite_reference_review.json',report)
with (out/'all_416_no_prediction_cases.csv').open('x',encoding='utf-8',newline='') as stream:
    fields=['entity','array_prefix','label_policy','threshold_level','full_test_points','actual_anomaly_points',
            'actual_reference_events','predicted_positive_points','predicted_events','score_maximum','frozen_threshold',
            'classified_nonfinite_cells','classification']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in reviewed)
shutil.copy2(root/configuration,out/'configuration.json')
shutil.copy2(__file__,out/'review_source.py')
assert sha(out/'review_source.py')==config['audit_source_sha256']
(out/'README.md').write_text('''# 对 TAB 前 21 个完整实体中所有非有限值的原函数复核

逐一核对 TranAD float64、原完整 SMAP、种子 1104 已完成的前 21/54 个实体成功守卫，保留全部 16,884 次指标结果。832 个非有限单元全部来自 affiliation precision/F，各 416 个，没有指标执行异常。

使用完整保存分数、四种原始标签、冻结的 3 个验证集阈值及 10 个 TRAIN+TEST 分数诊断阈值，复算全部 13 个阈值并与原记录逐位一致。对每个非有限 precision/F 对，在所有真实时间戳上执行未经修改的原 TAB affiliation precision/F/recall 函数，共 416 项输入、1,248 次原函数调用。416 项均有实际异常事件，同时没有任何预测事件。原函数返回 precision/F=NaN、recall=0，与原记录完全一致，832 个单元全部获得原因分类。

这些非有限值表示“评价已经执行，固定阈值没有产生预测事件，原参考定义下 precision/F 未定义”；不能记作未运行、补成 0 或删去后计算无条件均值。无需为了消除 NaN 修改阈值或修补原 affiliation 算法。此结论仅解释所审查的 832 个单元，不能据此把模型整体低性能归因于阈值，也不能推断与论文高 F1 差异的全部原因。

TRAIN+TEST 分数阈值是已冻结诊断控制，使用了测试分数分布；它与验证集单独校准的阈值分别标识，没有选择最优阈值或使用测试标签重新调参。本次未修改任何原结果、训练/重推理模型、执行 VUS、增加训练种子或更改排行榜。54 实体的完整 TAB 案例和其余原研究实验仍未完成。全部原 45/681/273/702 研究范围保留。
''',encoding='utf-8',newline='\n')
git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
              paths=[configuration]+[p.relative_to(root).as_posix() for p in out.iterdir()])
(root/'tmp/TAB21_nonfinite_reference_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(classified_nonfinite_cells=832,reviewed_entity_label_threshold_cases=len(reviewed),
    original_function_calls=3*len(reviewed),all_13_thresholds_exact=True,original_results_unchanged=True)))
