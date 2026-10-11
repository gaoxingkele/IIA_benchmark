"""Register all139 original CrossAD jobs and append every native metric obligation."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import io
import json
import shutil

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT,read,digest
from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy,previous

ORACLE=(('precision','P_best'),('recall','R_best'),('f1','F1_best'),('accuracy','ACC_best'))
AUTHOR=(('auroc','_auc','AUC_ROC'),('average_precision','_auc','AUC_PR'),
    ('R_AUC_ROC','_r_auc','R_AUC_ROC'),('R_AUC_PR','_r_auc','R_AUC_PR'),
    ('VUS_ROC','_vus','VUS_ROC'),('VUS_PR','_vus','VUS_PR'),
    ('Affiliation_ACC','_affiliation','Affiliation_ACC'),('Affiliation_P','_affiliation','Affiliation_P'),
    ('Affiliation_R','_affiliation','Affiliation_R'),('Affiliation_F1','_affiliation','Affiliation_F1'))
POINT=('precision','recall','f1','auroc','average_precision')


def identity(row):return tuple(row[k] for k in ('algorithm','dataset','task','protocol','metric'))


def metric_catalog(recipe):
    result=[(recipe+'_test_label_oracle',metric,'oracle',field) for metric,field in ORACLE]
    result += [(recipe+'_author_evaluation',metric,group,field) for metric,group,field in AUTHOR]
    for ratio,percentile in [('0.1','99.9'),('1','99'),('5','95')]:
        result += [(recipe+'_validation'+percentile+'_no_PA',metric,'validation/'+ratio,metric) for metric in POINT]
    if len(result)!=29:raise ValueError('Every original source indicator and registered validation ratio required')
    return result


def complete_cohorts(queue):
    jobs=queue['jobs'];groups=defaultdict(list)
    if len(jobs)!=139 or len({j['id'] for j in jobs})!=139:
        raise ValueError('Preserve all139 unique original registered jobs')
    if queue['settings']['evaluation']['strict_false_alarm_percent']!=[.1,1,5]:
        raise ValueError('All original validation operating points required')
    for job in jobs:groups[job['author_recipe'],job['dataset']].append(job)
    if len(groups)!=29:raise ValueError('All29 original native recipe/dataset cohorts required')
    for (recipe,dataset),cases in groups.items():
        seeds=[j['seed'] for j in cases]
        expected=[2025] if recipe=='release_checkpoint' else queue['settings']['seeds']
        if sorted(seeds)!=sorted(expected) or len(set(seeds))!=len(expected):
            raise ValueError('Every original unique seed and fixed-weight evaluation must remain')
    return groups


def publish(config_path):
    config=read(config_path)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen full inventory/publication source changed')
    queue=read(config['queue']);groups=complete_cohorts(queue)
    base=ROOT/config['base_publication'];target=ROOT/config['output_directory']
    receipt=previous.verified_capture(base,'publication_validation.json')
    old_bytes=(base/'all_algorithm_dataset_metrics.csv').read_bytes()
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);headers=reader.fieldnames;old=list(reader)
    if len(old)!=7283 or len(headers)!=31:raise ValueError('Every previous scientific slot and field required')
    # Other tracks also distinguish masking pattern, missing ratio and budgets.
    # Keep their complete records byte for byte; this native key is CrossAD only.
    existing={identity(row):row for row in old if row['algorithm'].startswith('crossad_complete/')}
    if len(existing)!=26:raise ValueError('Exactly26 prior native CrossAD slots required')
    point=read(config['completed_point_audit']);model=read(config['completed_model_replay'])
    guard=read(config['completed_model_guard_receipt'])
    if (not point['passed'] or not model['full_original_model_reinference_passed']
            or not model['full_original_author_metric_reexecution_passed']
            or not model['original_full_state_dictionary_loaded_strictly']
            or model['cuda_context_initialized'] or model['additional_training_seed_slots']!=0
            or guard['exit_code'] or guard['resource_guard_aborted'] or not guard['full_native_model_replay_passed']):
        raise ValueError('Complete original model, source evaluators and successful guarded execution required')
    job=next(j for j in queue['jobs'] if j['id']==model['job_id'])
    if (job['id']!=point['job_id'] or job['seed']!=2025 or job['author_recipe']!='fresh_train'
            or job['dataset']!='MSL' or model['source_result_sha256']!=digest(ROOT/job['output_directory']/'result.json')):
        raise ValueError('Exactly the audited original MSL main seed2025 required')
    for name,expected in model['replay_outputs'].items():
        if digest(ROOT/config['model_artifact_directory']/name)!=expected:
            raise ValueError('Actual full model/evaluator replay output changed')
    captured=datetime.now(timezone.utc).isoformat();native_rows=[];new_rows=[];cohorts=[]
    for (recipe,dataset),cases in sorted(groups.items()):
        cohorts.append(dict(recipe=recipe,dataset=dataset,original_job_ids=[j['id'] for j in cases],
            original_seed_ids=[j['seed'] for j in cases],required_repeats=len(cases),
            fixed_pretrained_weight_evaluation=recipe=='release_checkpoint',
            paper_reconstructed_ablation=recipe.startswith('paper_reconstructed_')))
        for protocol,metric,group,field in metric_catalog(recipe):
            row={key:'' for key in headers}
            row.update(algorithm='crossad_complete/'+recipe,dataset=dataset,task='anomaly_detection',
                protocol=protocol,metric=metric,n='0',required_repeats=str(len(cases)),
                author_equivalence_certified='False',strict_TAB_result='False',metric_unit='0-1',
                captured_utc=captured,source=config['inventory_proof_path'],
                comparison='separate_native_flat_source_protocol_not_strict_leaderboard',
                selection_boundary='Original flat author data, source column deletion, scaler fitted on all nominal training rows before validation split, original nonoverlap author test versus full native-tail validation controls; entity-grouped TAB/paper equivalence uncertified.',
                uncertainty_scope='Fixed pretrained weights' if recipe=='release_checkpoint' else 'Six original independent training seeds required; fewer seeds remain partial. Original temporal-block bootstrap is conditional on each fitted model and not training-seed uncertainty.')
            key=identity(row)
            if key in existing:
                row=dict(existing[key])
            elif (recipe,dataset)==('fresh_train','MSL'):
                if group=='oracle':value=model['original_author_metrics']['_best_f1'][0][field]
                elif group.startswith('validation/'):
                    value=point['validation_only_point_metrics'][group.split('/',1)[1]]['metrics'][field]
                else:value=model['original_author_metrics'][group][0][field]
                row.update(n='1',mean=str(float(value)),source=config['completed_model_replay'],
                    epochs=str(point['training_budget']['original_maximum_epochs']))
                new_rows.append(row)
            else:new_rows.append(row)
            native_rows.append(row)
    merged=old+new_rows
    if (len(native_rows)!=841 or len(new_rows)!=815 or len(merged)!=8098
            or len({identity(r) for r in native_rows})!=841
            or any(identity(r) in existing for r in new_rows)
            or sum(bool(r['mean']) for r in new_rows)!=29):
        raise ValueError('Every native obligation and exactly29 audited new native metrics required')
    if target.exists():raise FileExistsError('Preserve prior complete publications')
    target.mkdir(parents=True)
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=headers);writer.writerows(new_rows)
    (target/'all_algorithm_dataset_metrics.csv').write_bytes(old_bytes+buffer.getvalue().encode('utf-8'))
    for p in base.iterdir():
        if p.is_file() and p.name not in ('all_algorithm_dataset_metrics.csv','all_metrics.md','results.html','README.rst','publication_validation.json'):
            shutil.copyfile(p,target/p.name)
    inventory=dict(captured_utc=captured,queue=config['queue'],queue_sha256=digest(config['queue']),
        registered_original_native_jobs=139,registered_native_cohorts=29,metric_slots_per_native_cohort=29,
        registered_native_metric_slots=841,prior_native_metric_slots_retained=26,new_native_metric_slots=815,
        original_scope=config['original_scope'],cohorts=cohorts,completed_main_training_seeds=[2025],
        completed_main_dataset='MSL',complete_native_model_replay=config['completed_model_replay'],
        full_independent_point_audit=config['completed_point_audit'],newly_measured_metric_rows=29,
        completed_main_required_training_seeds=6,all_original_experiments_complete=False)
    proof_path=ROOT/config['inventory_proof_path'];proof_path.parent.mkdir(parents=True,exist_ok=True)
    with proof_path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(inventory,indent=2)+'\n')
    with (target/'crossad_complete_native_metric_grid.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader();writer.writerows(native_rows)
    intro='原139项CrossAD任务全部登记为29个方法版本—数据集组合，每组包含原作者14项指标及三个验证比例的15项点级指标。发布权重、完整训练和论文消融转录分开；全部旧7283项指标及采集时间不变，仅新增已完整重放MSL主方法seed2025的29项实测值，其余新增槽位留空。原六种子要求、所有未完成任务及SMAP历史身份保留，作者及完整TAB等价未认证。'
    (target/'all_metrics.md').write_text('# 完整算法—数据集实验技术指标 v37\n\n'+intro+'\n\n'+
        legacy.table(headers,[[r.get(k) or '—' for k in headers] for r in merged])+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(merged),encoding='utf-8')
    (target/'README.rst').write_text('完整算法—数据集指标 v37\n==========================\n\n'+intro+
        '\n\n原生作者测试73728点与完整控制测试73729点分别核验；原模型1327979参数、20epoch上限、patience3，实际在12epoch停止。'
        '重放完整原模型，严格恢复原THOP计数缓冲区，未丢弃算法权重；原250阈值、100窗口范围/VUS与SPOT初始化模型推断、阈值及Affiliation原算法完整重算。'
        '首次缺少计数缓冲区的严格加载失败记录保留；原容差及原始队列没有放宽。\n\n'
        '本版8098项登记指标：2088项实测，6010项缺结果。原45论文/681方法基线消融/273原始数据任务及采用基线全部保留，尚未完成。\n',encoding='utf-8')
    updated=dict(receipt)
    updated.update(published_utc=captured,base_publication=config['base_publication'],
        base_publication_utc=receipt['published_utc'],changed_row_positions=[],
        base_receipt_sha256=digest(base/'publication_validation.json'),aggregate_metric_rows=8098,
        algorithm_configuration_names=len({r['algorithm'] for r in merged}),
        task_rows=dict(Counter(r['task'] for r in merged)),
        measured_task_rows=dict(Counter(r['task'] for r in merged if r['mean'])),
        measured_metric_rows=2088,missing_metric_rows=6010,appended_native_metric_rows=815,
        newly_measured_metric_rows=29,replaced_metric_rows=0,changed_numeric_metric_rows=0,
        base_numeric_csv_prefix_byte_identical=True,all_previous7283_rows_and_capture_times_exact=True,
        all_numeric_tables_byte_identical_to_base=False,performance_rows_changed=True,
        registered_CrossAD_jobs=139,registered_CrossAD_native_cohorts=29,
        registered_CrossAD_native_metric_slots=841,completed_CrossAD_MSL_main_seeds=1,
        required_CrossAD_MSL_main_seeds=6,additional_training_seed_slots=0,
        newly_completed_canonical_training_slots=1,all_original_experiments_complete=False,
        author_equivalence_certified=False,strict_TAB_result=False,other_tracks_recaptured=False,
        original_scope=config['original_scope'],source_receipts=receipt['source_receipts']+config['source_receipts']+
            [dict(path=config_path,sha256=digest(config_path)),dict(path=config['inventory_proof_path'],sha256=digest(proof_path))],
        checks=dict(all_old7283_records_unchanged=True,all139_original_native_jobs_registered=True,
            all29_native_cohorts_with_complete_seed_obligations=True,all31fields_retained=True,
            original_full_model_and_all_source_metrics_replayed=True,missing_results_not_zero=True))
    updated['outputs']={p.name:digest(p) for p in target.iterdir() if p.is_file()}
    (target/'publication_validation.json').write_text(json.dumps(updated,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(registered_rows=8098,measured_rows=2088,missing_rows=6010,
        native_jobs=139,native_cohorts=29,new_native_slots=815,newly_measured_metrics=29)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    publish(parser.parse_args().config)
