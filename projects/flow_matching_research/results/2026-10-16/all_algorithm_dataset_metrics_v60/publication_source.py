"""Append complete original TAB metric controls without inventing macro coverage.

A fixed model is one completed case even where its full-entity metric is
undefined. Entity coverage and conditional bootstrap intervals have dedicated
columns; training-seed uncertainty is never inferred from entity counts.
"""
from __future__ import annotations
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from scripts.flow_matching import publish_native_condition_catalog_v55 as p
ROOT, read, sha = p.ROOT, p.read, p.sha
MODULE='scripts.flow_matching.publish_TAB_complete_case_catalog_v60'
TEST='tests/test_fm_TAB_complete_case_catalog_v60.py'
CONFIG='configs/reproducibility/fm_TAB_complete_case_catalog.2026-10-16_v60.json'
ENTITY_FIELDS=('aggregation_unit','entity_total','entity_finite','entity_error','entity_nonfinite',
               'defined_subset_macro_mean','entity_bootstrap_ci95')


def complete_case_rows(headers, report, source, captured):
    if (not report['passed'] or report['entities']!=54 or report['metric_calls']!=43416
            or report['case']['seed']!=1104 or report['case']['dataset']!='smap'
            or report['case']['family']!='tranad64' or len(report['macro_metrics'])!=4):
        raise ValueError('Exact completed full54 fixed-model original TAB case required')
    case=report['case'];policies=case['required_label_policies']
    if set(report['macro_metrics'])!=set(policies) or report['full_TAB_training_or_author_harness_equivalence']:
        raise ValueError('Every label view and original non-equivalence boundary required')
    rows=[]
    for policy in policies:
        if len(report['macro_metrics'][policy])!=201:
            raise ValueError('Every original201 metric control is required')
        for identity,value in report['macro_metrics'][policy].items():
            total,finite,error,nonfinite=(value[k] for k in ['total_entities','finite_entities','error_entities','nonfinite_entities'])
            if total!=54 or total!=finite+error+nonfinite:
                raise ValueError('Complete original entity denominator and explicit outcomes required')
            mean=value['complete_entity_macro_mean'];ci=value['complete_entity_macro_CI95']
            if finite==54:
                if (not isinstance(mean,(int,float)) or not math.isfinite(mean) or ci is None or len(ci)!=2
                        or not all(math.isfinite(x) for x in ci) or value['CI_resamples']!=200):
                    raise ValueError('All54 finite entities and original200 bootstrap required')
            elif mean is not None or ci is not None or value['CI_resamples']!=0:
                raise ValueError('A defined subset cannot impersonate the full54 macro mean or CI')
            subset=value['defined_subset_macro_mean']
            if ((finite==0 and subset is not None) or (finite>0 and (subset is None or not math.isfinite(subset)))
                    or value['training_seed_uncertainty'] or value['CI_unit']!='entity_conditional_on_fixed_model_and_threshold'):
                raise ValueError('Explicit defined-subset value and fixed-model entity uncertainty required')
            metric=identity.rsplit('/',1)[1];level=identity.rsplit('/',1)[0]
            row=dict.fromkeys(headers,'')
            row.update(algorithm='TranAD/FP64_local_original_full109',dataset='SMAP/full54_entities',task='anomaly_detection',
                protocol='original_TAB21_fixed_score_controls/entity_macro/'+case['id']+'/'+policy+'/'+level,
                metric=metric,n='1',required_repeats='1',mean=repr(mean) if mean is not None else '',
                author_equivalence_certified='False',strict_TAB_result='False',captured_utc=captured,source=str(source),
                metric_unit='fraction',comparison='complete_fixed_case_full54_macro' if finite==54 else 'complete_execution_full54_macro_undefined_preserve_coverage',
                score_recipe=json.dumps(dict(canonical_id=case['id'],case_training_seed=1104,label_policy=policy,
                    official_metric_control=identity,aggregation='entity_macro',TAB_original_metric_functions=True,
                    local_model_is_author_architecture=False),sort_keys=True),aggregation_unit='entity_macro_conditional_on_one_fixed_model',
                entity_total='54',entity_finite=str(finite),entity_error=str(error),entity_nonfinite=str(nonfinite),
                defined_subset_macro_mean=repr(subset) if subset is not None else '',entity_bootstrap_ci95=json.dumps(ci) if ci is not None else '',
                uncertainty_scope='One fixed local FP64 model. n=1/required_repeats=1 counts a complete case,not54training seeds. '
                    'Original200 entity-bootstrap intervals appear only in entity_bootstrap_ci95; training-seed std/SE/CI stay blank. '
                    'Undefined full macro stays blank; separately named subset mean retains exact coverage.')
            row['selection_boundary']=('Continuous original scores; no threshold selection.' if level=='score' else
                'Frozen validation-only score percentile; strict >. ' if level.startswith('validation/') else
                'Original fixed train+TEST-score percentile diagnostic; strict >; no best-TEST-F1 selection. ')
            if level!='score':
                row['selection_boundary']+=('Original PA uses TEST labels within each separate entity.' if metric.startswith('adjust_') else
                    'Original named label metric; fixed predictions; no cross-entity event joining.')
            rows.append(row)
    if len(rows)!=804 or len({p.key(x) for x in rows})!=804:
        raise ValueError('All804 unique controls must remain present')
    return rows


def verify_packet(packet):
    integrity=read(packet/'packet_integrity.json')
    for item in integrity['files']:
        if sha(packet/item['path'])!=item['sha256']:
            raise ValueError('Complete source evidence packet changed')
    summary=read(packet/'complete_case_TAB_metric_controls.json')
    guard=read(packet/'actual_complete_summary_guard.json')
    proof=read(packet/'independent_verification.json')
    if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['proof_sha256']!=sha(packet/'complete_case_TAB_metric_controls.json')
            or proof['complete_test_points']!=435826 or proof['complete_original_TAB_calls']!=43416
            or proof['separately_recomputed_full_point_raw_PA_confusions_and_tie_ranking_cells']!=22896
            or not proof['all804_macro_means_coverage_and_200_entity_bootstraps_independently_recomputed']):
        raise ValueError('Actual complete entity guards and independent closure required')
    lineage=summary['source_lineage']
    for key,digest in [('native_scores','native_scores_sha256'),('native_proof','native_sha256'),('review_proof','review_sha256')]:
        if sha(lineage[key])!=lineage[digest]:
            raise ValueError('Fixed full source/native model lineage changed')
    return summary


def publish(configuration=CONFIG):
    config=read(ROOT/configuration)
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen source/catalog changed: '+pin['path'])
    base=ROOT/config['base_catalog'];packet=ROOT/config['complete_case_packet']
    report=verify_packet(packet)
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);old_headers=reader.fieldnames
    if len(before)!=10991 or sum(bool(x['mean']) for x in before)!=2384:
        raise ValueError('Preserve the complete frozen v59 catalog')
    if set(ENTITY_FIELDS)&set(old_headers):
        raise ValueError('Entity coverage fields must not overwrite old column semantics')
    headers=old_headers+list(ENTITY_FIELDS);rows=[]
    for old in before:
        row=dict(old);row.update(dict.fromkeys(ENTITY_FIELDS,''));rows.append(row)
    captured=datetime.now(timezone.utc).isoformat()
    added=complete_case_rows(headers,report,packet/'complete_case_TAB_metric_controls.json',captured)
    rows.extend(added)
    if len(rows)!=11795 or sum(bool(r['mean']) for r in rows)!=3084:
        raise ValueError('All804 controls including104 undefined full macros must be represented')
    for old,new in zip(before,rows):
        if old!={k:new[k] for k in old_headers} or any(new[k] for k in ENTITY_FIELDS):
            raise ValueError('Every legacy field/capture time and new blank entity field must remain unchanged')
    if {p.key(r) for r in before}&{p.key(r) for r in added}:
        raise ValueError('New entity-macro controls must remain separate from old models/protocols')
    if any(any(r[k] for k in ['std','se','ci95_low','ci95_high','paper_mean']) for r in added):
        raise ValueError('Never invent paper values or training-seed uncertainty')
    output=ROOT/config['output_directory'];output.mkdir(exist_ok=False)
    (output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
    with (output/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    page=p.interactive(rows,headers).replace('实验指标 v55','实验指标 v60').replace('算法—数据集—指标 v55','算法—数据集—指标 v60')
    (output/'results.html').write_text(page,encoding='utf-8')
    status=dict(captured_utc=captured,original_scope=config['original_scope'],configuration=configuration,configuration_sha256=sha(ROOT/configuration),
        metric_rows=11795,numeric_rows=3084,original10991_rows_all31_original_fields_unchanged=True,
        new_complete_fixed_model_cases=1,new_metric_rows=804,new_full54_finite_macro_rows=700,new_full_macro_undefined_rows=104,
        new_training_seed_slots=0,new_neural_training_or_inference=False,all_original_experiments_complete=False)
    (output/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('''# 算法—数据集—指标 v60

总11795行，3084行有数值。原v59的10991行、全部31个既有字段和捕获时间逐字段保持；只追加TranAD FP64／SMAP／seed1104原始TAB函数的804个完整固定评分对照。新增700项可定义完整54实体macro均值，104项因部分实体返回非有限值而保留完整macro空白，不能补零或当作未执行。

新增七列明确实体覆盖、部分有效子集均值与实体bootstrap区间。n=1/required_repeats=1代表一个完整固定模型条件；54个实体和200次实体bootstrap都不是五训练种子。训练种子SD/SE/CI列留空，实体条件区间只在entity_bootstrap_ci95列。defined_subset_macro_mean与entity_finite、entity_error、entity_nonfinite一起显示，不能当完整分母性能。

四套标签边界、六种分数指标、十五种标签指标、三个验证阈值及十个原训练+测试分数固定比例诊断全部保留，不挑最好标签或测试F1。PA明确使用测试真值且只在各实体内部执行。macro与其他轨道micro不同；表中auc_pr遵循原函数average_precision_score定义。

该模型仍是有明确结构、学习率、窗口等偏差的本地FP64实现；原始TAB指标函数对照不认证作者模型、完整TAB训练策略或历史环境。新增行author_equivalence_certified和strict_TAB_result均为False。原完整54实体源/原生检查点、43416次实际调用、原守卫及22896个独立基础指标和全部804项独立bootstrap闭合证明已经核验。

878案例/2057标签视图的完整队列继续保留。本表只追加已完整核验案例的指标，未完成的其余案例仍以原配置与执行矩阵为准，不能把缺少指标行解释为无需执行。CrossAD15/139与生成完整重跑7/70沿用v59捕获状态；单细胞、新TranAD源码等独立结果仍需逐轨合并。全部45论文/681方法记录/273数据任务/702原始指标及原论文、TAB、新数据整体目标未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),output/'publication_source.py')
    (output/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(output).as_posix(),sha256=sha(x)) for x in sorted(output.rglob('*')) if x.is_file()]),indent=2)+'\n',encoding='utf-8')
    return status


if __name__=='__main__':
    print(json.dumps(publish()))
