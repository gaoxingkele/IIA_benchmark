"""Publish two newly finished original seeds after full-model and array audits."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import math
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching.audit_grasp_complete_replay_union_v1 import metrics

ROOT, legacy = previous.ROOT, previous.legacy
RECIPE = dict(source_samples=5, flow_evaluations=10)
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


def validate_er_metadata(results):
    if len(results) != 2 or sorted(r['job']['seed'] for r in results) != [1103, 1104]:
        raise ValueError('Exactly the two original ER training seeds required')
    for result in results:
        job = result['job']
        if (job['id'] != 'grasp_protocol_v3__er__swan__swan_sf__seed%d' % job['seed']
                or job['profile'] != 'er' or job['dataset'] != 'swan' or job['entity'] != 'swan_sf'
                or result['status'] != 'completed' or result['diagnostic']
                or result['epochs_completed'] != 1500 or result['optimizer_updates'] != 93000
                or job['expected_optimizer_updates'] != 93000 or result['parameter_count'] != 69244
                or result['scored_points'] != 72000 or result['validation_points'] != 3935
                or job['test_points'] != 72000 or job['validation_points'] != 3935
                or job['score_recipes'] != [RECIPE] or len(result['metrics']) != 1
                or result['metrics'][0]['recipe'] != RECIPE):
            raise ValueError('Original full architecture, training and scoring budgets required')
        values = result['metrics'][0]['metrics']
        oracle, validation = values['paper_retrospective'], values['validation_only_control']
        if (not oracle['test_labels_used_for_threshold_optimization'] or oracle['point_adjustment']
                or validation['point_adjustment'] or validation['quantile'] != .99
                or validation['threshold_comparator'] != '>'
                or validation['test_features_used_for_calibration']
                or validation['test_labels_used_for_calibration']
                or result['paper_equivalence_certified'] or result['strict_TAB_result']):
            raise ValueError('Oracle and validation-only protocols must stay separate and uncertified')


def audit_er_cohort(config):
    import numpy as np
    from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import SCIENTIFIC_FIELDS
    queue = legacy.read(ROOT / config['grasp_queue'])
    canonical = {j['id']: j for j in queue['jobs']}
    old_proof = legacy.read(ROOT / config['grasp_preserved_full_array_proof'])
    new_proof = legacy.read(ROOT / config['grasp_new_full_model_proof'])
    if (old_proof['full_original_models'] != 13 or old_proof['full_original_scoring_recipes'] != 31
            or not old_proof['all_complete_arrays_point_mapping_and_metrics_independently_verified']
            or not old_proof['original_full_native_worker_exit_success_verified']
            or not old_proof['no_new_training_seed_slots']
            or not new_proof['full_checkpoint_contents_and_re_inference_passed']
            or new_proof['full_job_count'] != 1 or len(new_proof['records']) != 1
            or new_proof['cuda_context_initialized'] or new_proof['additional_training_seed_slots'] != 0
            or new_proof['score_tolerance'] != config['score_tolerance']
            or new_proof['metric_tolerance'] != config['metric_tolerance']):
        raise ValueError('Actual preserved and new full native replays required')
    old = [r for r in old_proof['records'] if r['canonical_id'] ==
           'grasp_protocol_v3__er__swan__swan_sf__seed1103']
    if len(old) != 1:
        raise ValueError('Preserved original ER seed must have complete native evidence')
    new = new_proof['records'][0]
    if (new['canonical_id'] != 'grasp_protocol_v3__er__swan__swan_sf__seed1104'
            or not new['full_checkpoint_restoration_and_reinference_passed']):
        raise ValueError('Only the newly completed original ER seed1104 may enter')
    guard = legacy.read(ROOT / config['grasp_new_guard_receipt'])
    status = legacy.read(ROOT / config['grasp_new_terminal_status'])
    if (guard['exit_code'] != 0 or not guard['full_replay_passed'] or guard['resource_guard_aborted']
            or status['state'] != 'completed' or status['active_pid'] is not None
            or not status['full_replay_passed']):
        raise ValueError('Successful terminal full native replay required')
    results, records = [], []
    for proof in [old[0], new]:
        job = canonical[proof['canonical_id']]
        path = ROOT / job['output_directory'] / 'result.json'
        expected_sha = proof.get('result_sha256', proof.get('original_complete_training_result_sha256'))
        if legacy.sha(path) != expected_sha:
            raise ValueError('Preserved original full training result changed')
        result = legacy.read(path)
        if any(result['job'][field] != job[field] for field in SCIENTIFIC_FIELDS):
            raise ValueError('Canonical original ER science or budget changed')
        results.append(result)
        for artifact in result['artifacts']:
            if legacy.sha(ROOT / artifact['path']) != artifact['sha256']:
                raise ValueError('Full trained model, history or saved scores changed')
        data = result['data_audit']
        if legacy.sha(ROOT / data['prepared_path']) != data['prepared_sha256']:
            raise ValueError('Frozen full prepared data changed')
        for raw in data['raw_sources']:
            if legacy.sha(ROOT / raw['path']) != raw['sha256']:
                raise ValueError('Frozen raw source changed')
        if legacy.sha(ROOT / job['model_config']) != job['model_config_sha256']:
            raise ValueError('Original model configuration changed')
        if len(proof['recipes']) != 1:
            raise ValueError('One full original ER scoring recipe required')
        recipe = proof['recipes'][0]
        original_path = recipe['original_score_path']
        replay_path = recipe.get('replay_score_path')
        if (recipe['recipe'] != RECIPE or recipe['complete_test_points'] != 72000
                or recipe['complete_validation_points'] != 3935
                or legacy.sha(original_path) != recipe['original_score_sha256']
                or legacy.sha(replay_path) != recipe['replay_score_sha256']):
            raise ValueError('Full native scoring recipe binding differs')
        with np.load(ROOT / data['prepared_path'], allow_pickle=False) as prepared, \
                np.load(original_path, allow_pickle=False) as original, \
                np.load(replay_path, allow_pickle=False) as replay:
            if not np.array_equal(original['labels'], prepared['labels']):
                raise ValueError('All evaluated labels must match original full data')
            for field, shape in [('test_scores', (72000,)), ('validation_scores', (3935,)),
                                 ('labels', (72000,)), ('test_row_ids', (72000,)),
                                 ('validation_raw_row_ids', (3935,))]:
                if (original[field].shape != shape or replay[field].shape != shape
                        or not np.isfinite(original[field]).all() or not np.isfinite(replay[field]).all()):
                    raise ValueError('Every original test and validation point required')
                if field.endswith('scores'):
                    if not np.allclose(original[field], replay[field], **config['score_tolerance']):
                        raise ValueError('Full scores exceed frozen native replay tolerance')
                elif not np.array_equal(original[field], replay[field]):
                    raise ValueError('Original full point mapping changed')
            if (not np.array_equal(original['test_row_ids'], np.arange(72000))
                    or not np.array_equal(original['validation_raw_row_ids'],
                                          np.arange(data['split_cut'], data['raw_train_points']))):
                raise ValueError('Full grouped test/validation row identity required')
            actual = metrics(original['labels'], original['test_scores'], original['validation_scores'])
            replayed = metrics(replay['labels'], replay['test_scores'], replay['validation_scores'])
            original_prediction = original['test_scores'] > np.quantile(original['validation_scores'], .99)
            replay_prediction = replay['test_scores'] > np.quantile(replay['validation_scores'], .99)
            disagreements = int(np.count_nonzero(original_prediction != replay_prediction))
            if disagreements / 72000 > config['metric_tolerance']['maximum_prediction_disagreement_rate']:
                raise ValueError('Full validation-calibrated classifications differ')
            for protocol, values in actual.items():
                for name, value in values.items():
                    saved = result['metrics'][0]['metrics'][protocol][name]
                    if name == 'threshold':
                        tolerance = dict(rtol=config['metric_tolerance']['threshold_rtol'],
                                         atol=config['metric_tolerance']['threshold_atol'])
                        if not np.isclose(value, saved, **tolerance) or not np.isclose(value, replayed[protocol][name], **tolerance):
                            raise ValueError('Independent validation-only threshold differs')
                    elif (abs(value-saved) > config['metric_tolerance']['metric_atol']
                          or abs(value-replayed[protocol][name]) > config['metric_tolerance']['metric_atol']):
                        raise ValueError('Independent original/replayed metric differs')
        records.append(dict(canonical_id=job['id'], seed=job['seed'], result_path=path.relative_to(ROOT).as_posix(),
            result_sha256=legacy.sha(path), full_test_points=72000, full_validation_points=3935,
            full_epochs=1500, optimizer_updates=93000, independent_original_metrics=actual,
            independent_replayed_metrics=replayed, prediction_disagreements=disagreements,
            original_score_sha256=legacy.sha(original_path), replay_score_sha256=legacy.sha(replay_path),
            full_native_checkpoint_replay_and_original_arrays_verified=True))
    validate_er_metadata(results)
    return results, records


def increment_rows(original, cfm, er_results, source, captured):
    validate_er_metadata(er_results)
    groups = [g for g in cfm['seed_aggregates'] if (g['track'], g['dataset'], g['variant'], g['epochs']) ==
              ('appendix_d', 'pendulum', 'node', 10)]
    if len(groups) != 1 or groups[0]['completed_seeds'] != 5 or groups[0]['required_seeds'] != 5:
        raise ValueError('Complete original five-seed NODE10 cohort required')
    group = groups[0]
    updates = []
    for metric in ['MSE', 'MAE', 'RMSE']:
        matches = [r for r in original if r['algorithm'] == 'CFM-TS/node' and r['dataset'] == 'pendulum'
                   and r['protocol'] == 'appendix_d' and str(r['epochs']) == '10' and r['metric'] == metric]
        if len(matches) != 1 or int(matches[0]['n']) != 4 or int(matches[0]['required_repeats']) != 5:
            raise ValueError('Only the original partial NODE10 metric slots may change')
        row = dict(matches[0])
        row.update({key: group[metric + '_' + key] for key in ['n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high']})
        row.update(source=source, captured_utc=captured, author_equivalence_certified=False, strict_TAB_result=False,
                   comparison='all_five_full_models_replayed_author_alignment_unverified')
        updates.append(row)
    for protocol, field, names in [('paper_test_label_oracle_no_PA', 'paper_retrospective', ['AP', 'ROC', 'Best_F1']),
                                   ('validation_only_1_percent_no_PA_control', 'validation_only_control', ['precision', 'recall', 'f1'])]:
        for metric in names:
            matches = [r for r in original if r['algorithm'] == 'GRASP/er' and r['dataset'] == 'swan'
                       and r['protocol'] == protocol and r['metric'] == metric
                       and json_recipe(r['score_recipe']) == RECIPE]
            if len(matches) != 1 or int(matches[0]['n']) != 1 or int(matches[0]['required_repeats']) != 10:
                raise ValueError('Only original one-of-ten ER metric slots may change')
            values = [r['metrics'][0]['metrics'][field][metric] for r in er_results]
            row = dict(matches[0])
            if not math.isclose(float(row['mean']), values[0], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Previously published original ER seed must remain in the cohort')
            row.update(independent_stats(values))
            row.update(source=source, captured_utc=captured, author_equivalence_certified=False, strict_TAB_result=False,
                       comparison='two_of_ten_full_native_models_replayed_author_alignment_unverified')
            updates.append(row)
    return updates


def json_recipe(value):
    import json
    return json.loads(value) if isinstance(value, str) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    if config['original_scope'] != SCOPE:
        raise ValueError('All original paper, method, ablation and data obligations remain required')
    for receipt in config['source_receipts']:
        if legacy.sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen native publication evidence changed: ' + receipt['path'])
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    cfm = legacy.read(ROOT / config['cfm_capture'] / 'execution_audit.json')
    arrays = legacy.read(ROOT / config['cfm_capture'] / 'independent_full_metrics_audit.json')
    union = legacy.read(ROOT / config['cfm_full_model_union'])
    cfm_sha = legacy.sha(ROOT / config['cfm_capture'] / 'execution_audit.json')
    if (cfm['registered_jobs'] != 285 or cfm['job_counts']['completed'] != 85
            or arrays['audited_full_canonical_slots'] != 85 or arrays['full_five_seed_groups'] != 17
            or not arrays['all_completed_full_arrays_and_metrics_verified']
            or not arrays['all_seed_aggregates_independently_verified']
            or arrays['source_capture_sha256'] != cfm_sha or union['source_capture_sha256'] != cfm_sha
            or union['full_canonical_models_verified'] != 85 or union['bitwise_identical_models'] != 85
            or not union['full_checkpoint_contents_and_re_inference_passed']
            or not union['all_completed_canonical_models_covered']
            or union['original_canonical_slots'] != 285 or union['required_original_scope'] != SCOPE
            or union['original_prediction_tolerance'] != dict(rtol=1e-7, atol=1e-8)):
        raise ValueError('All85 complete canonical full models and full original arrays required')
    er_results, er_records = audit_er_cohort(config)
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    captured = datetime.now(timezone.utc).isoformat()
    updates = increment_rows(original, cfm, er_results, config['increment_proof'], captured)
    rows = previous.merge_audited_rows(original, updates)
    keys = {previous.key(row) for row in updates}
    changed = [i for i, (a, b) in enumerate(zip(original, rows)) if a != b]
    if len(rows) != len(original) or len(changed) != 9 or any(previous.key(rows[i]) not in keys for i in changed):
        raise ValueError('Only nine existing original metric slots may change')
    measured = sum(r['mean'] not in ('', None) for r in rows)
    if measured != old['measured_metric_rows']:
        raise ValueError('Seed increments must not invent measured metric slots')
    target = ROOT / config['output_directory']
    proof_path = ROOT / config['increment_proof']
    if target.exists() or proof_path.exists():
        raise FileExistsError('Native increment proofs and published tables are immutable')
    target.mkdir(parents=True)
    legacy.write(proof_path, dict(captured_utc=captured, configuration=args.config,
        configuration_sha256=legacy.sha(ROOT / args.config), cfm_full_model_union=config['cfm_full_model_union'],
        cfm_completed_canonical_models=85, cfm_full_five_seed_groups=17, grasp_er_records=er_records,
        metric_rows=updates, additional_training_seed_slots=0, newly_completed_existing_training_slots=2,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        original_scope=SCOPE))
    legacy.csv_write(target / 'all_algorithm_dataset_metrics.csv', rows)
    for name in ['overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv']:
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '评分配方', '指标',
               '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集时间UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs', ''),
               str(r.get('pattern', '')) + '/' + str(r.get('missing_ratio', '')), r.get('score_recipe', ''),
               r['metric'], '%s/%s' % (r['n'], r['required_repeats']),
               *[legacy.fmt(r.get(k)) for k in ['mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean']],
               r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text('# 完整算法—数据集实验技术指标\n\n'
        '—表示缺结果，0表示实测零。各协议、预算及采集时间分别保留；全部实验仍未完成。\n\n'
        + legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    summary = [[r['algorithm'], r['metric'], '%s/%s' % (r['n'], r['required_repeats']),
                *[legacy.fmt(r.get(k)) for k in ['mean', 'std', 'ci95_low', 'ci95_high']], r['protocol']] for r in updates]
    (target / 'README.rst').write_text('完整算法—数据集指标 v23\n==========================\n\n'
        '%d项指标；%d项有实测值；%d项缺结果。本次只更新九项既有指标。\n\n' % (len(rows), measured, len(rows)-measured)
        + legacy.table(['算法', '指标', '完成／预定', '均值', 'STD', '95%下界', '95%上界', '协议'], summary)
        + '\n\nCFM-TS全部85份已完成模型均通过完整原始测试网格重推断，逐点完全一致；'
        'Pendulum/NODE/附录D/10轮补齐5/5种子，17组五种子完整。全部285项原始任务仍须完成。\n\n'
        'GRASP/ER/SWAN为2/10原始训练种子，每份1500轮、93000次更新、72000测试点、3935验证点。'
        '完整CPU重推断在冻结容差内通过，验证阈值分类无差异；保存原始训练分数并独立重算指标。'
        '测试标签选阈值与验证集第99百分位阈值、无PA的对照分别列出。\n\n'
        '其余7274项指标及采集时间保留v22。45篇论文、681项方法基线消融、273项原始数据任务及采用基线'
        '范围不变，尚未全部完成；作者实现等价和严格TAB认证仍未取得。\n\n'
        '完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。\n', encoding='utf-8')
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=captured, base_publication_utc=old['published_utc'], replaced_metric_rows=9,
        changed_numeric_metric_rows=9, newly_measured_metric_rows=0, aggregate_metric_rows=len(rows),
        measured_metric_rows=measured, missing_metric_rows=len(rows)-measured,
        measured_task_rows=dict(Counter(r['task'] for r in rows if r['mean'] not in ('', None))),
        cfm_completed_canonical_slots=85, cfm_complete_groups=17, cfm_all_completed_full_model_replays_verified=85,
        grasp_complete_cohorts=old['grasp_complete_cohorts']+1, grasp_er_completed_training_seeds=2,
        grasp_er_required_training_seeds=10, grasp_er_full_native_models_replayed=2,
        other_track_rows_and_capture_times_preserved=True, newly_completed_canonical_training_slots=2,
        additional_training_seed_slots=0, all_original_experiments_complete=False, original_scope=SCOPE)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=legacy.sha(ROOT / args.config)),
        dict(path=config['increment_proof'], sha256=legacy.sha(proof_path))]
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, changed=9, cfm_full_models=85, grasp_er_seeds=2,
               all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
