"""Audit all newly available frozen strict cohorts and preserve other results."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import psutil

from scripts.flow_matching.publish_full_range_increment_v25 import (
    ROOT, PROTOCOL, SCOPE, METRICS as RANGE_METRICS, audit_evaluation,
    range_updates, stream_sha, legacy,
)
from scripts.flow_matching.publish_result_listing_audited_v9 import verified_capture, key, merge_audited_rows
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api

POINT_METRICS = ('precision', 'recall', 'f1', 'auroc', 'average_precision')
FIELDS = ('n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exclusive_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def assert_close(actual, expected, description):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError('Undefined metric changed: '+description)
    elif not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Independent saved-score calculation differs: '+description)


def array_shape(archive, name):
    import numpy as np
    with archive.open(name+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unregistered prepared NPY format')
    return shape


def independent_points(truth, scores, threshold):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    prediction = scores > threshold
    truth = truth.astype(bool)
    tp, fp, fn = (int(np.count_nonzero(a)) for a in
                  (truth & prediction, ~truth & prediction, truth & ~prediction))
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/(tp+fn) if tp+fn else 0.
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.,
        true_positives=float(tp), false_positives=float(fp), false_negatives=float(fn),
        auroc=float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2 else None,
        average_precision=float(average_precision_score(truth, scores)) if truth.any() else None)


def audit_points(scores, prepared, archive, dataset, result):
    import numpy as np
    strict = result['metrics']['strict']['1']
    calibration = strict['calibration']
    if (strict['protocol'] != 'strict_validation_only_no_pa_v1' or strict['coverage'] != 'complete'
            or strict['point_adjustment'] is not False or calibration['calibration_split'] != 'validation'
            or calibration['false_alarm_percent'] != 1
            or calibration['rule'] != 'validation_score_percentile_linear_strict_greater'
            or calibration['calibration_dataset_sha256'] != dataset['input_sha256']
            or calibration['model_artifact_sha256'] != result['checkpoint_sha256']):
        raise ValueError('Original complete validation-only99percentile no-PA protocol required')
    expected = {e['array_prefix']+'_'+suffix for e in dataset['entities']
                for suffix in ('train', 'validation', 'test', 'labels')}
    if set(scores.files) != expected:
        raise ValueError('Every complete original score array is required')
    validations, truths, tests, receipts = [], [], [], []
    entity_ids = {e['entity_id'] for e in dataset['entities']}
    if set(strict['per_entity']) != entity_ids:
        raise ValueError('All frozen entity identities required')
    for entity in dataset['entities']:
        prefix = entity['array_prefix']
        item = dict(entity_id=entity['entity_id'])
        for split in ('train', 'validation', 'test'):
            values = scores[prefix+'_'+split]
            shape = array_shape(archive, prefix+'_'+split)
            if values.ndim != 1 or len(values) != shape[0] or not np.isfinite(values).all():
                raise ValueError('Complete finite train/validation/test timestamp scores required')
            item[split+'_points'] = len(values)
            if split == 'validation':
                validations.append(values.astype(np.float64))
        labels, test = scores[prefix+'_labels'], scores[prefix+'_test']
        if (labels.ndim != 1 or labels.shape != test.shape or len(labels) != entity['test_points']
                or not np.isin(labels, [0, 1]).all() or not np.array_equal(labels, prepared[prefix+'_labels'])):
            raise ValueError('All original prepared labels and test timestamps required')
        metric = independent_points(labels, test, calibration['threshold'])
        for name, value in metric.items():
            assert_close(value, strict['per_entity'][entity['entity_id']][name], entity['entity_id']+'/'+name)
        item['independent_point_metrics'] = metric
        receipts.append(item)
        truths.append(labels)
        tests.append(test)
    validation = np.concatenate(validations).astype(np.float64)
    if (len(validation) != calibration['calibration_points']
            or hashlib.sha256(validation.astype('<f8').tobytes()).hexdigest() != calibration['calibration_sha256']):
        raise ValueError('All original validation scores in frozen entity order required')
    threshold = float(np.percentile(validation, 99, method='linear'))
    assert_close(threshold, calibration['threshold'], 'validation threshold')
    micro = independent_points(np.concatenate(truths), np.concatenate(tests), threshold)
    for name, value in micro.items():
        assert_close(value, strict['micro'][name], 'micro/'+name)
    assert_close(float(np.mean([r['independent_point_metrics']['f1'] for r in receipts])),
                 strict['macro_f1'], 'entity macro F1')
    return dict(validation_points=len(validation), threshold=threshold,
        full_entities=len(receipts), full_test_points=sum(r['test_points'] for r in receipts),
        micro=micro, entities=receipts, no_point_adjustment=True, no_test_calibration=True)


def point_updates(original, records, source, captured):
    groups = defaultdict(list)
    seen = set()
    for record in records:
        identity = (record['algorithm'], record['dataset'], record['seed'])
        if identity in seen:
            raise ValueError('Same canonical seed must not be counted twice')
        seen.add(identity)
        groups[identity[:2]].append(record)
    updates = []
    for (algorithm, dataset), group in groups.items():
        for metric in POINT_METRICS:
            matches = [r for r in original if (r['task'],r['algorithm'],r['dataset'],r['protocol'],r['metric']) ==
                       ('anomaly_detection',algorithm,dataset,PROTOCOL,metric)]
            if len(matches) != 1:
                raise ValueError('Exactly one existing strict metric slot required')
            old = matches[0]
            values = [r['point_audit']['micro'][metric] for r in group if r['point_audit']['micro'][metric] is not None]
            if not values:
                continue
            stats = independent_stats(values)
            if not int(old['n']) <= stats['n'] <= int(old['required_repeats']):
                raise ValueError('Original canonical seed budget must remain unchanged')
            if stats['n'] == int(old['n']):
                for field in FIELDS:
                    actual = stats[field]
                    expected = None if old[field] in ('', None) else float(old[field])
                    assert_close(actual, expected, 'same-count '+field)
                continue
            updated = dict(old)
            updated.update(stats,source=source,captured_utc=captured,
                author_equivalence_certified=False,strict_TAB_result=False)
            updates.append(updated)
    return updates


def verify_delta(original, merged, updates):
    if len(original) != len(merged) or [key(r) for r in original] != [key(r) for r in merged]:
        raise ValueError('Every original registered scientific metric identity must remain')
    allowed = {key(r) for r in updates}
    fields = set(FIELDS)|{'source','captured_utc','author_equivalence_certified','strict_TAB_result'}
    changed = []
    for index, (before, after) in enumerate(zip(original, merged)):
        if before == after:
            continue
        if (key(after) not in allowed or after['protocol'] != PROTOCOL
                or after['task'] != 'anomaly_detection'
                or after['metric'] not in POINT_METRICS+RANGE_METRICS
                or not int(before['n']) < int(after['n']) <= int(before['required_repeats'])
                or any(before.get(f) != after.get(f) for f in set(before)|set(after) if f not in fields)):
            raise ValueError('Only proved original strict repeat increments may change')
        changed.append(index)
    if len(changed) != len(updates) or len(allowed) != len(updates):
        raise ValueError('One proof per changed metric required')
    return changed


def verify_registration(config):
    if config['original_scope'] != SCOPE:
        raise ValueError('Preserve all original experiments and adopted baselines')
    for receipt in config['source_receipts']:
        if stream_sha(ROOT/receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen ready-cohort input changed: '+receipt['path'])
    identities = [(c['algorithm'],c['dataset']) for c in config['cohorts']]
    if len(identities) != len(set(identities)):
        raise ValueError('Unique original cohorts required')


def worker(config, config_path):
    import numpy as np
    target = ROOT/config['output_directory']
    proof_dir = ROOT/config['proof_directory']
    if target.exists() or proof_dir.exists():
        raise FileExistsError('Preserve every prior publication or audit attempt')
    proof_dir.mkdir(parents=True)
    base = ROOT/config['base_publication']
    old = verified_capture(base,'publication_validation.json')
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        columns, original = reader.fieldnames, list(reader)
    canonical = {}
    for cohort in config['cohorts']:
        for entry in cohort['runs']:
            for job in read(ROOT/entry['queue'])['jobs']:
                if job['id'] in canonical and canonical[job['id']] != job:
                    raise ValueError('Conflicting original canonical job')
                canonical[job['id']] = job
    records, ranges, hashes = [], [], {}
    for cohort in config['cohorts']:
        for entry in cohort['runs']:
            result_path = ROOT/entry['result']
            result = read(result_path)
            job = canonical[entry['id']]
            if result['job'] != job or stream_sha(result_path) != entry['result_sha256']:
                raise ValueError('Original complete result identity changed')
            calibration = result['metrics']['strict']['1']['calibration']
            run = dict(source_execution_id=job['id'],algorithm_config=cohort['algorithm'],dataset=cohort['dataset'],
                seed=job['seed'],result_sha256=entry['result_sha256'],scores_sha256=result['scores_sha256'],
                checkpoint_sha256=result['checkpoint_sha256'],threshold=calibration['threshold'])
            if entry['evaluation'] is not None:
                evaluation, range_proof = audit_evaluation(dict(evaluation_path=entry['evaluation'],
                    evaluation_sha256=entry['evaluation_sha256']),run,entry['science'],job,hashes)
                ranges.append(evaluation)
            else:
                from scripts.flow_matching.run_tsad_queue import complete_result
                if not complete_result(ROOT, job):
                    raise ValueError('Incomplete canonical saved-score model')
                range_proof = None
            manifest = read(ROOT/job['dataset_manifest_path'])
            dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
            data_receipts = [dict(path=dataset['input_npz'],sha256=dataset['input_sha256'])]+dataset['source_files']
            for receipt in job['frozen_files']+data_receipts:
                if receipt['path'] not in hashes:
                    hashes[receipt['path']]=stream_sha(ROOT/receipt['path'])
                if hashes[receipt['path']] != receipt['sha256']:
                    raise ValueError('Every original raw/prepared/configuration source must agree')
            with np.load(result_path.parent/'scores.npz',allow_pickle=False) as scores, \
                    np.load(ROOT/dataset['input_npz'],allow_pickle=False) as prepared, \
                    zipfile.ZipFile(ROOT/dataset['input_npz']) as archive:
                point = audit_points(scores,prepared,archive,dataset,result)
            records.append(dict(id=job['id'],algorithm=cohort['algorithm'],dataset=cohort['dataset'],seed=job['seed'],
                canonical_job=job,result_path=entry['result'],result_sha256=entry['result_sha256'],
                scores_sha256=result['scores_sha256'],checkpoint_sha256=result['checkpoint_sha256'],
                point_audit=point,range_audit=range_proof,registered_model_config=job['model_config'],
                registered_model_parameters=read(ROOT/job['model_config'])['parameters'],
                source_reproduction_status=result['reproduction_status'],
                pretrained_fixed_weights=cohort['algorithm'].startswith('fm_moment_pretrained_'),
                model_reinference_performed=False,author_equivalence_certified=False))
            exclusive_progress = dict(completed_full_saved_score_audits=len(records),required_audits=config['full_saved_score_runs'],
                latest_id=job['id'],full_test_points=point['full_test_points'])
            legacy.write(proof_dir/'progress.json',exclusive_progress)
            print(json.dumps(exclusive_progress),flush=True)
    if len(records) != config['full_saved_score_runs']:
        raise ValueError('All registered complete saved-score obligations required')
    captured = datetime.now(timezone.utc).isoformat()
    proof_path = proof_dir/'independent_complete_saved_score_audit.json'
    source = proof_path.relative_to(ROOT).as_posix()
    updates = point_updates(original,records,source,captured)+range_updates(original,ranges,source,captured)
    merged = merge_audited_rows(original,updates)
    changed = verify_delta(original,merged,updates)
    measured = sum(r['mean'] not in ('',None) for r in merged)
    evidence = dict(captured_utc=captured,configuration=config_path,configuration_sha256=stream_sha(ROOT/config_path),
        original_scope=SCOPE,full_saved_score_runs_audited=len(records),full_range_evaluations_audited=len(ranges),
        registered_cohorts=len(config['cohorts']),records=records,metric_rows=updates,changed_positions=changed,
        raw_prepared_scientific_source_receipts=[dict(path=p,sha256=s) for p,s in hashes.items()],
        additional_training_seed_slots=0,new_model_training_runs=0,model_reinference_performed=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary=config['boundary'])
    exclusive_write(proof_path,evidence)
    target.mkdir(parents=True)
    with (target/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(merged)
    for name in ('original_generation_jobs.csv','original_energy_jobs.csv','corrected_two_seed_metric_evidence.json'):
        shutil.copyfile(base/name,target/name)
    labels=['算法/消融','数据集','任务','协议','指标','完成种子','预定种子','均值','STD','SE','95%下界','95%上界',
        '论文值','作者等价确认','采集时间UTC','证据来源','掩码类型','缺失率','comparison','评分配方','轮数',
        'paper_std','指标单位','选择/协议边界','paper_spread','评价重复数','评价STD','评价SE','评价95%区间',
        '不确定性口径','完整TAB确认']
    if len(columns)!=len(labels):raise ValueError('Every existing CSV field must be displayed')
    boundary='本版逐份核验新可用的完整保存分数结果，再更新既有严格评价槽位；缺结果仍为—。SMAP历史预处理结果保留历史身份，修正重跑未完成。登记的本地转录模型、预训练权重评价与原作者管线分别解释，作者及完整TAB等价未认证。范围指标为实体宏平均，点级指标为完整测试时间点微平均；二者不可混用。'
    (target/'all_metrics.md').write_text('# 完整算法—数据集实验技术指标 v31\n\n'+boundary+'\n\n'+
        legacy.table(labels,[[r.get(c,'') for c in columns] for r in merged])+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(merged),encoding='utf-8')
    strict_groups = defaultdict(dict)
    for row in merged:
        if row['task']=='anomaly_detection' and row['protocol']==PROTOCOL:
            strict_groups[row['algorithm'],row['dataset']][row['metric']]=row
    def value(group,metric):
        row=group.get(metric)
        return '—' if row is None or row['mean'] in ('',None) else legacy.fmt(float(row['mean']))
    strict_header=['算法配置','数据集','点级完成/预定','P','R','F1','AUROC','AP','VUS-ROC(n)','VUS-PR(n)','Affiliation F(n)','F1 STD','F1 95%CI']
    strict_values=[]
    for (algorithm,dataset),group in strict_groups.items():
        f=group['f1']
        strict_values.append([algorithm,dataset,str(f['n'])+'/'+str(f['required_repeats']),
            *[value(group,m) for m in POINT_METRICS],
            *[value(group,m)+' ('+str(group[m]['n'])+')' for m in RANGE_METRICS],
            legacy.fmt(None if f['std'] in ('',None) else float(f['std'])),
            str(f['ci95_low'] or '—')+' / '+str(f['ci95_high'] or '—')])
    (target/'strict_algorithm_dataset.md').write_text('# 全部异常检测组合\n\n'+boundary+'\n\n指标0–1。\n\n'+legacy.table(strict_header,strict_values)+'\n',encoding='utf-8')
    datasets=sorted({d for _,d in strict_groups})
    algorithms=sorted({a for a,_ in strict_groups})
    overview=[]
    for algorithm in algorithms:
        line=[algorithm]
        for dataset in datasets:
            f=strict_groups.get((algorithm,dataset),{}).get('f1')
            line.append('未登记' if f is None else '—' if f['mean'] in ('',None) else
                '%.2f [%s/%s]'%(100*float(f['mean']),f['n'],f['required_repeats']))
        overview.append(line)
    (target/'overview.md').write_text('# 算法与数据集 F1 总表\n\n'+boundary+'\n\nF1百分数；单元格显示均值 [完整种子/预定种子]。预训练1/1为固定权重评价。\n\n'+legacy.table(['算法配置']+datasets,overview)+'\n',encoding='utf-8')
    newly=sum(original[i]['mean'] in ('',None) and merged[i]['mean'] not in ('',None) for i in changed)
    receipt=copy.deepcopy(old)
    receipt.update(published_utc=captured,base_publication=config['base_publication'],base_publication_utc=old['published_utc'],
        aggregate_metric_rows=len(merged),measured_metric_rows=measured,missing_metric_rows=len(merged)-measured,
        measured_task_rows=dict(Counter(r['task'] for r in merged if r['mean'] not in ('',None))),
        replaced_metric_rows=len(changed),changed_numeric_metric_rows=len(changed),newly_measured_metric_rows=newly,
        changed_row_positions=changed,full_saved_score_runs_audited=len(records),full_range_evaluations_audited=len(ranges),
        additional_training_seed_slots=0,new_model_training_runs=0,all_original_experiments_complete=False,
        author_equivalence_certified=False,strict_TAB_result=False,performance_rows_changed=True,
        all_numeric_tables_byte_identical_to_base=False,other_track_rows_and_capture_times_preserved=True)
    receipt.update(strict_groups=len(strict_groups),
        strict_complete_groups=sum(all(int(g[m]['n'])==int(g[m]['required_repeats']) and g[m]['mean'] not in ('',None)
            for m in POINT_METRICS) for g in strict_groups.values()),
        strict_partial_groups=sum(0<int(g['f1']['n'])<int(g['f1']['required_repeats']) for g in strict_groups.values()))
    receipt['checks']=dict(all_original7283_metric_identities_retained=len(merged)==7283,
        independent_full_validation_and_test_arrays_passed=True,no_PA_or_test_threshold_in_main=True,
        pretrained_evaluations_not_training_repeats=True,undefined_metrics_not_zero=True,
        all_other_rows_and_timestamps_exact=True,all31_CSV_fields_in_text=True)
    receipt['source_receipts']+=config['source_receipts']+[dict(path=config_path,sha256=stream_sha(ROOT/config_path)),
        dict(path=source,sha256=stream_sha(proof_path))]
    receipt['outputs']={p.name:stream_sha(p) for p in target.iterdir() if p.is_file()}
    exclusive_write(target/'publication_validation.json',receipt)
    (target/'README.rst').write_text('完整算法—数据集指标 v31\n==========================\n\n'+boundary+'\n\n'+
        '%d项登记指标，%d项已有实测，%d项缺结果。核验%d份完整保存分数运行，更新%d条指标，其中%d条由缺失变为实测。无新增训练或检查点重推断。\n'%(len(merged),measured,len(merged)-measured,len(records),len(changed),newly),encoding='utf-8')
    # Include the final explanatory file in the immutable export manifest.
    receipt['outputs']['README.rst']=stream_sha(target/'README.rst')
    (target/'publication_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    exclusive_write(proof_dir/'validation.json',dict(passed=True,full_saved_score_runs=len(records),full_range_evaluations=len(ranges),
        changed_metrics=len(changed),newly_measured=newly,total_metrics=len(merged),measured_metrics=measured,
        source_identity_checks_passed=True,all_original_experiments_complete=False))
    print(json.dumps(dict(rows=len(merged),measured=measured,changed=len(changed),newly_measured=newly,full_audits=len(records))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify_registration(config)
    if args.worker:return worker(config,args.config)
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full saved-score publication already has a controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting',
        configuration_sha256=stream_sha(ROOT/args.config))
    def update(values):state.update(values);legacy.write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
                PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.publish_ready_strict_metrics_v31','--config',args.config,'--worker'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid)
                update(dict(state='auditing_all_full_saved_scores',active_pid=child.pid,
                    active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                def saved_score_update(values):
                    values=dict(values)
                    if values.get('state')=='replaying_full_checkpoints':
                        values['state']='auditing_all_full_saved_scores'
                    update(values)
                code,usage=capped_wait(child,config,api,saved_score_update)
            receipt=dict(exit_code=code,full_saved_score_audit_passed=code==0,**usage)
            exclusive_write(base/'receipt.json',receipt)
            update(dict(state='completed' if code==0 else 'failed_preserved',active_pid=None,**receipt))
            if code:raise RuntimeError('Full saved-score audit failed; complete attempt preserved')
    finally:owner.close()


if __name__=='__main__':main()
