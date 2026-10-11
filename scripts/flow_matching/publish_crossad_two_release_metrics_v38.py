"""Fill all remaining original GECCO/MSL released-weight metrics after replay."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import math
import shutil
import io

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT,read,digest
from scripts.flow_matching.publish_crossad_complete_native_inventory_v37 import identity,metric_catalog,complete_cohorts
from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy,previous


def verified_release(point,model,guard):
    if (not point['passed'] or not point['fixed_release_checkpoint_evaluation']
            or point['completed_training_seed_slots']!=0 or point['training_epochs_completed']!=0
            or not model['full_original_model_reinference_passed']
            or not model['full_original_author_metric_reexecution_passed']
            or not model['original_full_state_dictionary_loaded_strictly']
            or not model['original_SPOT_init_model_inference_and_threshold_reexecution_passed']
            or not model['fixed_release_checkpoint_evaluation'] or model['completed_training_seed_slots']!=0
            or model['registered_required_release_evaluations']!=1 or model['additional_training_seed_slots']!=0
            or model['cuda_context_initialized'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or not guard['full_native_model_replay_passed']):
        raise ValueError('Complete fixed-weight model and evaluator replay only; never a fresh training seed')
    if point['job_id']!=model['job_id'] or point['full_native_points']!=model['full_score_points']:
        raise ValueError('Every original full score point and source job must agree')


def value_for(protocol,metric,group,field,point,model):
    if group=='oracle':return float(model['original_author_metrics']['_best_f1'][0][field])
    if group.startswith('validation/'):
        return float(point['validation_only_point_metrics'][group.split('/',1)[1]]['metrics'][field])
    return float(model['original_author_metrics'][group][0][field])


def publish(config_path):
    config=read(config_path)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen complete released-weight publication source changed')
    base=ROOT/config['base_publication'];target=ROOT/config['output_directory']
    receipt=previous.verified_capture(base,'publication_validation.json')
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);header=reader.fieldnames;old=list(reader)
    if len(old)!=8098 or len(header)!=31:raise ValueError('Full original8098slots and31fields required')
    rows=[dict(r) for r in old];lookup={identity(r):i for i,r in enumerate(rows) if r['algorithm'].startswith('crossad_complete/')}
    queue=read(config['queue']);complete_cohorts(queue)
    if sorted(r['dataset'] for r in config['completed_releases'])!=['GECCO','MSL']:
        raise ValueError('Both original completed released-weight evaluations required')
    captured=datetime.now(timezone.utc).isoformat();positions=[];evidence=[]
    for case in config['completed_releases']:
        point,model,guard=read(case['point_audit']),read(case['model_replay']),read(case['guard_receipt'])
        verified_release(point,model,guard)
        job=next(j for j in queue['jobs'] if j['id']==model['job_id'])
        if job['mode']!='release_checkpoint' or job['dataset']!=case['dataset'] or job['seed']!=2025:
            raise ValueError('Exactly original fixed release evaluation required')
        if model['source_result_sha256']!=digest(ROOT/job['output_directory']/'result.json'):
            raise ValueError('Original released-weight source changed')
        for name,expected in model['replay_outputs'].items():
            if digest(ROOT/case['model_artifact_directory']/name)!=expected:raise ValueError('Full native replay changed')
        changed=[]
        for protocol,metric,group,field in metric_catalog('release_checkpoint'):
            position=lookup[('crossad_complete/release_checkpoint',case['dataset'],'anomaly_detection',protocol,metric)]
            row=rows[position];value=value_for(protocol,metric,group,field,point,model)
            if not math.isfinite(value):raise ValueError('All measured release metrics must be finite')
            if row['mean']:
                if not math.isclose(float(row['mean']),value,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Previously published released-weight result must still replay')
                continue
            if row['n']!='0' or row['required_repeats']!='1':raise ValueError('Only missing fixed-weight slots may be filled')
            row.update(mean=str(value),n='1',captured_utc=captured,source=case['model_replay'],
                uncertainty_scope='One fixed original released checkpoint evaluation; zero newly trained models and no training-seed confidence interval.')
            positions.append(position);changed.append(dict(position=position,protocol=protocol,metric=metric,mean=value))
        if len(changed)!=16:raise ValueError('Exactly16 remaining metrics per completed release required')
        evidence.append(dict(dataset=case['dataset'],original_job=job['id'],original_release_checkpoint=job['release_checkpoint'],
            point_audit=case['point_audit'],full_model_replay=case['model_replay'],
            newly_filled_metric_slots=changed,previous13metric_records_and_capture_times_retained=True,
            completed_training_seed_slots=0))
    if len(set(positions))!=32 or sum(bool(r['mean']) for r in rows)!=2120:
        raise ValueError('Exactly32 previously missing fixed-weight values and2120 total measured slots required')
    if any(a!=b for i,(a,b) in enumerate(zip(old,rows)) if i not in positions):
        raise ValueError('Every other8066record and capture time must be unchanged')
    prefix=(ROOT/config['preserved_original7283_csv']).read_bytes()
    if not (base/'all_algorithm_dataset_metrics.csv').read_bytes().startswith(prefix):
        raise ValueError('The original7283 scientific records must remain an exact prefix')
    if target.exists():raise FileExistsError('Preserve every publication attempt')
    target.mkdir(parents=True)
    for file in base.iterdir():
        if file.is_file() and file.name not in {'all_algorithm_dataset_metrics.csv','crossad_complete_native_metric_grid.csv','all_metrics.md','results.html','README.rst','publication_validation.json'}:
            shutil.copyfile(file,target/file.name)
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=header);writer.writerows(rows[7283:])
    (target/'all_algorithm_dataset_metrics.csv').write_bytes(prefix+buffer.getvalue().encode('utf-8'))
    with (target/'crossad_complete_native_metric_grid.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=header);writer.writeheader()
        writer.writerows(r for r in rows if r['algorithm'].startswith('crossad_complete/'))
    intro='保持完整8098项指标槽位、139项CrossAD原生任务及29个组合的种子要求。GECCO/MSL原发布权重的完整模型、全部原生评价与原始标签对齐核验通过，各补齐16项指标；原有26项发布权重指标不变，其他8066条记录和采集时间不变。没有新增训练或种子。作者/完整TAB等价未认证，SMAP历史身份保留。'
    (target/'all_metrics.md').write_text('# 完整算法—数据集实验技术指标 v38\n\n'+intro+'\n\n'+legacy.table(header,[[r.get(k) or '—' for k in header] for r in rows])+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    (target/'README.rst').write_text('完整算法—数据集指标 v38\n==========================\n\n'+intro+'\n\n8098项登记指标：2120项有实测值，5978项缺结果。固定发布权重1/1表示一次完整权重评估，不能计作训练种子；完整从头训练与消融六种子要求保持。\n',encoding='utf-8')
    increment=dict(captured_utc=captured,base_publication=config['base_publication'],publication=config['output_directory'],
        changed_positions=sorted(positions),completed_released_weight_evaluations=evidence,
        original_scope=config['original_scope'],additional_training_seed_slots=0,new_completed_training_jobs=0,
        all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False)
    proof=ROOT/config['increment_proof_path'];proof.parent.mkdir(parents=True,exist_ok=True)
    with proof.open('x',encoding='utf-8') as stream:stream.write(json.dumps(increment,indent=2)+'\n')
    updated=dict(receipt)
    updated.update(published_utc=captured,base_publication=config['base_publication'],base_publication_utc=receipt['published_utc'],
        base_receipt_sha256=digest(base/'publication_validation.json'),measured_metric_rows=2120,missing_metric_rows=5978,
        measured_task_rows=dict(Counter(r['task'] for r in rows if r['mean'])),replaced_metric_rows=32,changed_numeric_metric_rows=32,
        changed_row_positions=sorted(positions),newly_measured_metric_rows=32,appended_native_metric_rows=0,
        newly_completed_canonical_training_slots=0,additional_training_seed_slots=0,other_tracks_recaptured=False,
        all_previous7283_rows_and_capture_times_exact=True,base_numeric_csv_prefix_byte_identical=False,
        original7283_numeric_csv_prefix_byte_identical=True,newly_published_complete_native_training_slots=0,
        previous_native_release_metric_rows_and_capture_times_exact=26,all_other8066_rows_and_capture_times_exact=True,
        full_released_weight_model_replays=2,new_training_jobs=0,all_original_experiments_complete=False,
        author_equivalence_certified=False,strict_TAB_result=False,
        source_receipts=receipt['source_receipts']+config['source_receipts']+
            [dict(path=config_path,sha256=digest(config_path)),dict(path=config['increment_proof_path'],sha256=digest(proof))],
        checks=dict(all8098slots_and31fields_retained=True,exactly32missing_fixed_release_metrics_filled=True,
            all_other8066records_and_capture_times_exact=True,full_two_models_and_every_native_evaluator_replayed=True,
            no_release_evaluation_counted_as_training_seed=True,full139jobs_and29cohorts_retained=True))
    if not (target/'all_algorithm_dataset_metrics.csv').read_bytes().startswith(prefix):
        raise ValueError('All original7283 scientific CSV rows must remain byte exact')
    updated['outputs']={p.name:digest(p) for p in target.iterdir() if p.is_file()}
    (target/'publication_validation.json').write_text(json.dumps(updated,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(registered_metrics=8098,measured_metrics=2120,missing_metrics=5978,newly_measured_metrics=32,new_training_seeds=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);publish(parser.parse_args().config)
