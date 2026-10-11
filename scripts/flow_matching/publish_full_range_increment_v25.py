"""Audit complete saved-score range evaluations and update only existing slots."""
import argparse
from collections import defaultdict, Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching.run_tsad_queue import complete_result

ROOT, legacy = previous.ROOT, previous.legacy
METRICS = ('VUS_ROC', 'VUS_PR', 'affiliation_f')
PROTOCOL = 'validation_only_1_percent_no_PA'
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


def stream_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_increment_proof(path, proof):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def entity_buffers(labels, record):
    import numpy as np
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Original binary labels required')
    if np.unique(labels).size != 2:
        if record['status'] != 'undefined_single_class_reference' or record['VUS_ROC'] is not None or record['VUS_PR'] is not None:
            raise ValueError('Undefined reference metrics must not become zero')
        return
    edges = np.diff(np.r_[0, labels.astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    expected = 2 * int(np.median(lengths))
    if (record['status'] != 'defined' or record['max_buffer'] != expected
            or record['buffers_evaluated'] != expected + 1 or record['threshold_samples'] != 250):
        raise ValueError('All original integer buffers and250thresholds required')


def verify_macros(evaluation):
    import numpy as np
    for records, aggregate, fields in [
        (evaluation['VUS_per_entity'], evaluation['VUS_entity_macro'], ['VUS_ROC', 'VUS_PR']),
        ({key: value['1'] for key, value in evaluation['strict_affiliation_per_entity'].items()},
         evaluation['strict_affiliation_entity_macro']['1'], ['affiliation_f', 'affiliation_precision', 'affiliation_recall'])]:
        for field in fields:
            values = [r[field] for r in records.values() if r[field] is not None]
            actual = aggregate[field]
            if actual['defined_entities'] != len(values) or actual['undefined_entities'] != len(records)-len(values):
                raise ValueError('Undefined entity counts must remain explicit')
            mean = actual['mean_over_defined_entities']
            if not values:
                if mean is not None:
                    raise ValueError('Undefined macro cannot become zero')
            elif not math.isclose(mean, float(np.mean(values)), rel_tol=1e-13, abs_tol=1e-13):
                raise ValueError('Independent full entity macro differs')


def range_updates(original, records, source, captured):
    values = defaultdict(lambda: defaultdict(list))
    identities = set()
    for e in records:
        algorithm, dataset = Path(e['model_config']).stem, e['dataset'].upper()
        identity = (algorithm, dataset, e['seed'])
        if identity in identities:
            raise ValueError('Same original seed cannot be counted twice')
        identities.add(identity)
        metrics = {**e['VUS_entity_macro'], **e['strict_affiliation_entity_macro']['1']}
        for metric in METRICS:
            value = metrics[metric]['mean_over_defined_entities']
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError('Finite defined native metrics required')
                values[algorithm, dataset][metric].append(value)
    updates = []
    for (algorithm, dataset), metrics in values.items():
        for metric, samples in metrics.items():
            matches = [r for r in original if (r['task'], r['algorithm'], r['dataset'], r['protocol'], r['metric']) ==
                       ('anomaly_detection', algorithm, dataset, PROTOCOL, metric)]
            if len(matches) != 1:
                raise ValueError('Only one existing canonical range metric slot may change')
            old = matches[0]
            stats = independent_stats(samples)
            if not int(old['n']) <= stats['n'] <= int(old['required_repeats']):
                raise ValueError('Completed repeats cannot disappear or exceed original budget')
            if stats['n'] == int(old['n']):
                for field in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'):
                    before, after = old[field], stats[field]
                    if before in ('', None) and after is None:
                        continue
                    if before in ('', None) or after is None or not math.isclose(float(before), after, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError('Existing same-count native range aggregate changed')
                continue
            row = dict(old)
            row.update(stats, source=source, captured_utc=captured,
                       author_equivalence_certified=False, strict_TAB_result=False)
            updates.append(row)
    return updates


def audit_evaluation(entry, run, science, canonical, hash_cache):
    import numpy as np
    path = ROOT / entry['evaluation_path']
    if stream_sha(path) != entry['evaluation_sha256'] or path.with_name('evaluation.sha256').read_text(encoding='ascii').strip() != entry['evaluation_sha256']:
        raise ValueError('Frozen complete native range artifact changed')
    e = legacy.read(path)
    result_path = ROOT / e['model_result_path']
    result = legacy.read(result_path)
    config = legacy.read(ROOT / science)
    settings_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (e['status'] != 'completed_range_evaluation' or result['job'] != canonical
            or e['id'] != run['source_execution_id'] or e['seed'] != run['seed']
            or Path(e['model_config']).stem != run['algorithm_config'] or e['dataset'].upper() != run['dataset']
            or stream_sha(result_path) != run['result_sha256'] or e['model_result_sha256'] != run['result_sha256']
            or e['scores_sha256'] != run['scores_sha256'] or result['checkpoint_sha256'] != run['checkpoint_sha256']
            or e['evaluation_config_sha256'] != settings_sha or e['source_receipts'] != config['source_receipts']
            or e['evaluation_identity'] != hashlib.sha256((run['result_sha256'] + settings_sha).encode()).hexdigest()
            or config['cpu_threads'] != 1 or not complete_result(ROOT, canonical)):
        raise ValueError('Original complete model, captured seed and scientific evaluation identity required')
    for source in e['source_receipts'] + canonical['frozen_files']:
        if source['path'] not in hash_cache:
            hash_cache[source['path']] = stream_sha(ROOT / source['path'])
        if hash_cache[source['path']] != source['sha256']:
            raise ValueError('Original frozen source, model configuration or data changed')
    calibration = result['metrics']['strict']['1']['calibration']
    if (calibration['calibration_split'] != 'validation' or calibration['false_alarm_percent'] != 1
            or calibration['rule'] != 'validation_score_percentile_linear_strict_greater'
            or calibration['threshold'] != run['threshold']):
        raise ValueError('Original validation-only99percentile threshold required')
    dataset = next(d for d in legacy.read(ROOT / canonical['dataset_manifest_path'])['datasets'] if d['id'] == result['dataset'])
    if dataset['input_npz'] not in hash_cache:
        hash_cache[dataset['input_npz']] = stream_sha(ROOT / dataset['input_npz'])
    if hash_cache[dataset['input_npz']] != dataset['input_sha256']:
        raise ValueError('Full prepared source data changed')
    for raw in dataset['source_files']:
        if raw['path'] not in hash_cache:
            hash_cache[raw['path']] = stream_sha(ROOT / raw['path'])
        if hash_cache[raw['path']] != raw['sha256']:
            raise ValueError('Original raw source data changed')
    entities = {item['entity_id'] for item in dataset['entities']}
    if set(e['VUS_per_entity']) != entities or set(e['strict_affiliation_per_entity']) != entities:
        raise ValueError('All original entities required')
    points = 0
    with np.load(result_path.parent / 'scores.npz', allow_pickle=False) as scores, np.load(ROOT / dataset['input_npz'], allow_pickle=False) as prepared:
        for entity in dataset['entities']:
            prefix = entity['array_prefix']
            labels, values = scores[prefix+'_labels'], scores[prefix+'_test']
            if (labels.shape != values.shape or labels.ndim != 1 or len(labels) != entity['test_points']
                    or not np.isfinite(values).all() or not np.array_equal(labels, prepared[prefix+'_labels'])):
                raise ValueError('All original entity timestamps, full labels and finite scores required')
            entity_buffers(labels, e['VUS_per_entity'][entity['entity_id']])
            points += len(labels)
    verify_macros(e)
    return e, dict(id=e['id'], seed=e['seed'], full_test_points=points, full_entities=len(entities),
                   evaluation_path=entry['evaluation_path'], evaluation_sha256=entry['evaluation_sha256'],
                   model_result_sha256=run['result_sha256'], scores_sha256=run['scores_sha256'],
                   checkpoint_sha256=run['checkpoint_sha256'], source_model_predates_this_evaluation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    if config['original_scope'] != SCOPE:
        raise ValueError('Keep every original paper, method, ablation and dataset obligation')
    for source in config['source_receipts']:
        if stream_sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen range publication input changed')
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    snapshot = legacy.read(ROOT / config['original_source_snapshot'])
    runs = {r['source_execution_id']: r for r in snapshot['strict_runs'] if r['calibration_false_alarm_percent'] == 1}
    selected = {entry['id'] for entry in config['evaluations']}
    if len(selected) != len(config['evaluations']):
        raise ValueError('Repeated original evaluation cannot add a seed')
    groups = {(runs[entry['id']]['algorithm_config'], runs[entry['id']]['dataset']) for entry in config['evaluations']}
    for run in runs.values():
        if (run['algorithm_config'], run['dataset']) in groups and run.get('VUS_ROC') is not None and run['source_execution_id'] not in selected:
            raise ValueError('Previously measured original seeds must remain included')
    canonical = {}
    for path in config['scientific_configs']:
        for queue in legacy.read(ROOT / path)['queue_paths']:
            for job in legacy.read(ROOT / queue)['jobs']:
                if job['id'] in canonical and canonical[job['id']] != job:
                    raise ValueError('Conflicting original queue identity')
                canonical[job['id']] = job
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    target, proof_path = ROOT / config['output_directory'], ROOT / config['increment_proof']
    if target.exists() or proof_path.exists():
        raise FileExistsError('Preserve all original and failed publication attempts')
    captured = datetime.now(timezone.utc).isoformat()
    records, evidence, hashes = [], [], {}
    for entry in config['evaluations']:
        e, proof = audit_evaluation(entry, runs[entry['id']], entry['scientific_config'], canonical[entry['id']], hashes)
        records.append(e)
        evidence.append(proof)
    updates = range_updates(original, records, config['increment_proof'], captured)
    rows = previous.merge_audited_rows(original, updates)
    changed = [i for i, (a, b) in enumerate(zip(original, rows)) if a != b]
    if len(changed) != len(updates) or len(rows) != len(original):
        raise ValueError('Only existing range slots may change')
    keys = {previous.key(r) for r in updates}
    if any(previous.key(rows[i]) not in keys or rows[i]['metric'] not in METRICS for i in changed):
        raise ValueError('Point metrics, author tracks and all other capture times must remain unchanged')
    proof = dict(captured_utc=captured, configuration=args.config, configuration_sha256=stream_sha(ROOT / args.config),
                 original_scope=SCOPE, native_complete_evaluations=evidence, evaluation_records=records,
                 full_saved_score_evaluations_verified=len(records), changed_existing_metric_rows=len(updates),
                 metric_rows=updates, raw_prepared_and_scientific_source_receipts=[dict(path=p, sha256=s) for p, s in hashes.items()],
                 additional_training_seed_slots=0, newly_completed_training_slots=0,
                 author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
    write_increment_proof(proof_path, proof)
    target.mkdir(parents=True)
    legacy.csv_write(target/'all_algorithm_dataset_metrics.csv', rows)
    for name in ('overview.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv'):
        shutil.copyfile(base/name, target/name)
    strict = copy.deepcopy(snapshot['strict_summary'])
    for update in updates:
        record = next(r for r in strict if r['algorithm_config'] == update['algorithm'] and r['dataset'] == update['dataset'])
        for field in ('n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high'):
            record[update['metric']+'_'+field] = update[field]
    headers = ['算法配置','数据集','完成/预定','状态','P','R','点级F1','AUROC','AP','VUS-ROC(n)','VUS-PR(n)','Affiliation F(n)','F1 STD','F1 95%CI']
    values = [[r['algorithm_config'],r['dataset'],f"{r['completed_seeds']}/{r['required_seeds']}",r['status'],
               *[legacy.fmt(r[k+'_mean']) for k in ('precision','recall','f1','auroc','average_precision')],
               *[legacy.fmt(r[k+'_mean'])+f" ({r.get(k+'_n') or 0})" for k in METRICS],
               legacy.fmt(r['f1_std']),legacy.fmt(r['f1_ci95_low'])+' / '+legacy.fmt(r['f1_ci95_high'])] for r in strict]
    (target/'strict_algorithm_dataset.md').write_text('# 全部异常检测组合\n\n验证分数第99百分位，无PA；指标0–1。点级指标及种子数保留旧快照；本次只更新VUS与Affiliation，逐指标采集时间见CSV。完整TAB训练作者等价未认证。\n\n'+legacy.table(headers,values)+'\n',encoding='utf-8')
    headers = ['任务','算法／消融','数据集','协议','轮数','掩码／缺失率','评分配方','指标','完成／预定','均值','STD','SE','95%下界','95%上界','论文值','采集时间UTC']
    values = [[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('epochs',''),str(r.get('pattern',''))+'/'+str(r.get('missing_ratio','')),r.get('score_recipe',''),r['metric'],f"{r['n']}/{r['required_repeats']}",*[legacy.fmt(r.get(k)) for k in ('mean','std','se','ci95_low','ci95_high','paper_mean')],r['captured_utc']] for r in rows]
    (target/'all_metrics.md').write_text('# 完整算法—数据集技术指标\n\n各协议及采集时间分别保留；—为缺结果，0为实测零。全部实验仍未完成。\n\n'+legacy.table(headers,values)+'\n',encoding='utf-8')
    (target/'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    measured = sum(r['mean'] not in ('',None) for r in rows)
    newly = sum(original[i]['mean'] in ('',None) and rows[i]['mean'] not in ('',None) for i in changed)
    (target/'README.rst').write_text('完整算法—数据集指标 v25\n==========================\n\n'
        f'{len(rows)}项指标；{measured}项有实测值；{len(rows)-measured}项缺结果。\n\n'
        f'本次核验{len(records)}项完整保存分数评价，更新{len(updates)}项既有VUS／Affiliation指标，其中{newly}项由缺失变为实测。'
        '点级P/R/F1、AUROC、AP、其他任务、作者协议、原训练种子预算及其采集时间均保留v24。无新增训练种子或模型训练。\n\n'
        '核验原始模型/分数/任务身份、原始和预处理数据校验、全部实体完整标签与时间点、250个阈值及每一个整数缓冲区、原验证阈值、定义/未定义计数及实体和种子宏平均。'
        'Affiliation仅使用验证阈值1%字段，测试标签比例诊断不进入本次汇总。单种子不生成不确定性，多种子使用样本STD、SE及Student-t95%区间。\n\n'
        '45篇论文、681项方法/基线/消融、273项原始数据任务及采用基线范围保留，全部实验尚未完成。完整TAB训练框架及论文作者等价未认证。\n\n'
        '全部指标见results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。完整证据见publication_validation.json及其increment_proof。\n',encoding='utf-8')
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=captured, base_publication_utc=old['published_utc'], replaced_metric_rows=len(updates),
        changed_numeric_metric_rows=len(updates), newly_measured_metric_rows=newly, aggregate_metric_rows=len(rows),
        measured_metric_rows=measured, missing_metric_rows=len(rows)-measured, measured_task_rows=dict(Counter(r['task'] for r in rows if r['mean'] not in ('',None))),
        range_full_saved_score_evaluations_verified=len(records), range_increment_proof=config['increment_proof'],
        other_track_rows_and_capture_times_preserved=True, all_numeric_tables_byte_identical_to_base=False,
        performance_rows_changed=bool(updates), newly_completed_canonical_training_slots=0, additional_training_seed_slots=0,
        original_scope=SCOPE, all_original_experiments_complete=False)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config,sha256=stream_sha(ROOT/args.config)),dict(path=config['increment_proof'],sha256=stream_sha(proof_path))]
    receipt['outputs'] = {p.name:stream_sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',receipt)
    print(dict(rows=len(rows),measured=measured,newly_measured=newly,changed=len(updates),full_evaluations=len(records),all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
