"""Publish all25 SPOT cases and update only the newly closed CrossAD condition.

Undefined calibration is distinct from a nonfinite metric and a pending case.
No neural execution, threshold fitting or source result overwrite occurs.
"""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
from scripts.flow_matching import publish_native_condition_catalog_v55 as p
from scripts.flow_matching import evaluate_tranad_original_SPOT_full30_v1 as spot
from scripts.flow_matching import audit_tranad_original_SPOT_completed_cases_v1 as independent
ROOT,read,sha=p.ROOT,p.read,p.sha
MODULE='scripts.flow_matching.publish_SPOT_full25_condition_catalog_v61'
TEST='tests/test_fm_SPOT_full25_condition_catalog_v61.py'
CONFIG='configs/reproducibility/fm_SPOT_full25_condition_catalog.2026-10-16_v61.json'
BASE='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v60'
SPOT_PACKET='projects/flow_matching_research/results/2026-10-16/TranAD_SPOT_first3_complete_full54_independent_v1'
CROSS_PACKET='projects/flow_matching_research/results/2026-10-16/CrossAD_row4_seed1104_full16_v1'
CROSS_ID='crossad_ablation__row4__gecco__seed1104'
EXTRA_FIELD='entity_calibration_undefined'


def registration():
    sc=read(ROOT/spot.CONFIG);spot.verify(sc)
    files=[MODULE.replace('.','/')+'.py',TEST,spot.CONFIG,spot.TEST,spot.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/audit_tranad_original_SPOT_completed_cases_v1.py',
        p.MODULE.replace('.','/')+'.py',BASE+'/all_algorithm_dataset_metrics.csv',
        BASE+'/all878_original_TAB_case_scope_and_captured_status.csv',
        SPOT_PACKET+'/packet_integrity.json',SPOT_PACKET+'/independent_full_array_audit.json',
        CROSS_PACKET+'/packet_integrity.json',CROSS_PACKET+'/all16_proof_bindings.json',
        'configs/experiments/fm_crossad_original_CUDA_full139.v2.json',
        'configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json']
    return dict(base_catalog=BASE,SPOT_packet=SPOT_PACKET,CrossAD_packet=CROSS_PACKET,
        output_directory='projects/flow_matching_research/results/2026-10-16/all_algorithm_dataset_metrics_v61',
        original_scope=sc['original_scope'],SPOT_configuration=spot.CONFIG,required_SPOT_unique_cases=25,
        required_SPOT_comparison_slots=30,required_SPOT_actual_label_views=65,required_SPOT_metric_rows=130,
        complete_SPOT_cases=3,updated_CrossAD_condition=CROSS_ID,complete_CrossAD_conditions=16,
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in files],all_original_experiments_complete=False)


def spot_rows(headers,config,cases,captured):
    completed={r['case']['id']:r for r in cases}
    if len(completed)!=len(cases):raise ValueError('No duplicate fixed-model cases')
    if len(config['entries'])!=25 or sum(len(r['entry']['case']['required_label_policies']) for r in config['entries'])!=65:
        raise ValueError('Keep every25 actual case and65 actual label views')
    rows=[]
    for record in config['entries']:
        case=record['entry']['case'];result=completed.get(case['id'])
        required={'smap':54,'msl':27,'smd':28}[case['dataset']]
        summaries={}
        if result:
            if result['case']!=case or result['full_entities']!=required:raise ValueError('Complete frozen case and all entities required')
            summaries={(r['label_policy'],r['mode']):r for r in result['full_denominator_macro_controls']}
            expected={(policy,mode) for policy in case['required_label_policies'] for mode in ['raw','source_PA']}
            if len(summaries)!=len(result['full_denominator_macro_controls']) or set(summaries)!=expected:
                raise ValueError('All labels/raw/PA controls required')
        for policy in case['required_label_policies']:
            for mode in ['raw','source_PA']:
                value=summaries.get((policy,mode));row=p.blank(headers)
                row.update(algorithm='TranAD/FP64_local_'+('original_full109' if record['source_module']==spot.original.MODULE else 'common105'),
                    dataset=case['dataset'].upper()+f'/full{required}_entities',task='anomaly_detection',metric='F1',metric_unit='fraction',
                    protocol='original_Grimshaw_SPOT_fixed_score/entity_macro/'+case['id']+'/'+policy+'/'+mode,
                    source=SPOT_PACKET+'/independent_full_array_audit.json' if result else spot.CONFIG,captured_utc=captured,
                    score_recipe=json.dumps(dict(canonical_id=case['id'],case_training_seed=case['seed'],label_policy=policy,
                        mode=mode,calibration='original_Grimshaw_SPOT_complete_fit_training_only',risk=1e-5,
                        source_epsilon=1e-5,source_PA_index0_preserved=True,local_model_is_author_architecture=False),sort_keys=True),
                    comparison='pending_full_original_SPOT_case_and_independent_audit',entity_total=str(required),
                    aggregation_unit='entity_macro_conditional_on_one_fixed_model',
                    selection_boundary='One frozen training-score Grimshaw fit per entity;all labels share threshold;strict >;no TEST-label threshold selection.'+
                        (' Original PA uses TEST labels and preserves source index0 behavior.' if mode=='source_PA' else ' No PA.'),
                    uncertainty_scope='One fixed local model,n1 denotes a complete case. No training-seed SD/SE/CI or entity bootstrap estimated. '
                        'Calibration-undefined entities have their own column;full macro remains undefined,defined subset is separate. '
                        'Shared original SMD references are comparison slots,not new runs.')
                if value:
                    finite,undefined=value['defined_entities'],value['undefined_entities']
                    mean,subset=value['full_entity_macro_f1'],value['explicitly_defined_subset_macro_f1']
                    if value['required_entities']!=required or finite+undefined!=required or finite<0 or undefined<0:
                        raise ValueError('Complete entity denominator and explicit calibration outcomes required')
                    if value['independent_training_repetitions']!=0 or value['training_seed_interval'] is not None:
                        raise ValueError('Do not invent between-training-seed uncertainty')
                    if (undefined and (mean is not None or value['full_macro_status']!='undefined_constant_calibration_entity')
                            or not undefined and (mean is None or not math.isfinite(mean) or value['full_macro_status']!='defined')
                            or finite and (subset is None or not math.isfinite(subset))
                            or not finite and subset is not None):
                        raise ValueError('Undefined full macro cannot be zero-imputed or replaced by a subset')
                    row.update(n='1',mean=repr(mean) if mean is not None else '',entity_finite=str(finite),entity_error='0',
                        entity_nonfinite='0',entity_calibration_undefined=str(undefined),defined_subset_macro_mean=repr(subset) if subset is not None else '',
                        comparison='complete_fixed_case_full_entity_macro' if not undefined else 'complete_execution_full_macro_undefined_constant_calibration')
                rows.append(row)
    if len(rows)!=130 or len({p.key(r) for r in rows})!=130:raise ValueError('All130 distinct full-scope controls required')
    return rows


def verify_packet(packet):
    for item in read(packet/'packet_integrity.json')['files']:
        if sha(packet/item['path'])!=item['sha256']:raise ValueError('Full immutable source evidence packet changed')


def evidence(c):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    for name in ['SPOT_packet','CrossAD_packet']:verify_packet(ROOT/c[name])
    source=read(ROOT/spot.CONFIG);spot.verify(source)
    report=read(ROOT/c['SPOT_packet']/'independent_full_array_audit.json')
    if report['complete_cases']!=3:raise ValueError('Exact three independently closed full cases required')
    for actual in report['cases']:
        record=next(r for r in source['entries'] if r['entry']['case']==actual['case'])
        if independent.audit_case(source,record)!=actual:raise ValueError('All original full-array independent SPOT checks required')
    cc=read(ROOT/cold.CONFIG);original,q=cold.verify(cc);mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==CROSS_ID)
    proof=read(ROOT/c['CrossAD_packet']/'full_checkpoint_array_proof.json')
    metric=read(ROOT/c['CrossAD_packet']/'independent_metric_report.json')
    if groups.condition(original,q,job)!=proof or metrics.audit(mc,q,upstream,job)!=metric:
        raise ValueError('Actual new CrossAD complete numerical guard/array/metric closure required')
    return source,report['cases'],job,metric,mc


def publish():
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Frozen full-scope publication configuration changed')
    sc,cases,job,metric,mc=evidence(c)
    with (ROOT/c['base_catalog']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);old_headers=reader.fieldnames
    if len(before)!=11795 or sum(bool(r['mean']) for r in before)!=3084 or EXTRA_FIELD in old_headers:
        raise ValueError('Exact complete v60 catalog required')
    headers=old_headers+[EXTRA_FIELD];rows=[dict(row,**{EXTRA_FIELD:''}) for row in before]
    captured=datetime.now(timezone.utc).isoformat()
    completed=p.cross_rows(headers,[job],{CROSS_ID:metric},captured,dict(cross_independent_artifact_root=mc['artifact_root']))
    replacements={p.key(r):r for r in completed};changed=0
    for index,row in enumerate(rows):
        key=p.key(row)
        if key not in replacements:continue
        if row['n']!='0' or row['mean']:raise ValueError('Never overwrite an already completed metric')
        rows[index]=replacements[key];changed+=1
    if changed!=15:raise ValueError('Exactly all15 previously pending CrossAD controls required')
    added=spot_rows(headers,sc,cases,captured)
    if {p.key(r) for r in before}&{p.key(r) for r in added}:raise ValueError('Keep original SPOT distinct from TAB and old POT approximation')
    rows+=added
    if len(rows)!=11925 or sum(bool(r['mean']) for r in rows)!=3107:raise ValueError('All pending/undefined/finite outcomes required')
    for old,new in zip(before,rows):
        if p.key(old) not in replacements and (old!={k:new[k] for k in old_headers} or new[EXTRA_FIELD]):
            raise ValueError('Every other legacy value and capture time must remain unchanged')
    output=ROOT/c['output_directory'];output.mkdir(exist_ok=False)
    (output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
    with (output/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    (output/'results.html').write_text(p.interactive(rows,headers).replace('实验指标 v55','实验指标 v61').replace('算法—数据集—指标 v55','算法—数据集—指标 v61'),encoding='utf-8')
    for name in ['all878_original_TAB_case_scope_and_captured_status.csv']:
        shutil.copyfile(ROOT/c['base_catalog']/name,output/name)
    with (output/'five_shared_original_SMD_comparison_references.csv').open('w',encoding='utf-8-sig',newline='') as f:
        refs=sc['shared_original_SMD_comparison_references'];writer=csv.DictWriter(f,fieldnames=list(refs[0]));writer.writeheader();writer.writerows(refs)
    status=dict(captured_utc=captured,original_scope=c['original_scope'],metric_rows=len(rows),numeric_rows=3107,
        legacy_rows_unchanged=11780,previously_pending_CrossAD_rows_completed=15,CrossAD_complete_conditions=16,CrossAD_required_conditions=139,
        new_SPOT_scope_rows=130,SPOT_unique_cases=25,SPOT_comparison_slots=30,SPOT_complete_cases=3,
        SPOT_completed_controls=24,SPOT_pending_controls=106,SPOT_defined_full_macro_controls=8,
        SPOT_executed_undefined_full_macro_controls=16,new_model_training_or_inference=False,all_original_experiments_complete=False)
    (output/'publication_validation.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('''# 算法—数据集—指标 v61

11925行，3107项有数值。原v60中11780行的既有字段与捕获时间保持；只将CrossAD row4/seed1104的15项原待完成控制填入实际独立核验值，并新增全部25个独立SPOT案例、65实际标签视图、原始/PA两种模式共130项范围行。

CrossAD完整条件16/139；完整五种子主要组仍1/22。新增SPOT已完成3个固定模型案例对应24个宏观F1控制，其中8项完整分母有定义，16项因恒定训练分数实体无法拟合而完整macro未定义；另106项尚待完整模型与校准产物，n=0。未定义完成项n=1且mean空白，与未执行不同。

新增entity_calibration_undefined专列，不能把无法校准误写为非有限指标或用子集均值填补。原始SMAP两案例各53/54实体有定义，完整macro留空；matched105种子1103全54实体的宏观F1为0.091164434880，原作者PA后0.217451363757。PA仍使用测试标签，二值预测ROC/AUC不是连续分数AUROC。全部为固定旧本地FP64模型上的原作者Grimshaw SPOT控制，不认证作者模型架构或严格TAB训练策略。

共享5个原始SMD案例构成matched比较位置，共30比较位置但只有25实际运行；随附引用表，绝不重复计为新增模型/种子。n=1仅指一个完整固定案例，训练种子SD/SE/CI与实体bootstrap均不虚构。未挑最好标签、种子、阈值或测试F1。

原TAB878案例/2057标签范围矩阵保持。生成完整重跑仍沿用原7/70状态，其他独立增量另有证据包。本表没有重训或推理，不以新表行数缩小45篇、681方法记录、273数据任务、702原指标及原论文/TAB/新数据的全部任务。整体未完成。
''',encoding='utf-8')
    shutil.copyfile(ROOT/(MODULE.replace('.','/')+'.py'),output/'publication_source.py')
    return status


if __name__=='__main__':print(json.dumps(publish()))
