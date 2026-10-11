"""Publish the complete original Electricity SAITS cohort and first BRITS seed."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import math
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

ROOT, legacy = previous.ROOT, previous.legacy


def validated_cohorts(proofs):
    cohorts = {'saits': [], 'brits': []}
    seen = set()
    for proof in proofs:
        if (not proof['full_checkpoint_contents_and_re_inference_passed']
                or proof['full_models_replayed'] != len(proof['records'])
                or proof['additional_training_seed_slots'] != 0
                or proof['cuda_context_initialized']):
            raise ValueError('Actual complete readonly native model replays required')
        for record in proof['records']:
            identity = record['id']
            if identity in seen:
                raise ValueError('A training seed cannot be counted twice')
            seen.add(identity)
            method = next((name for name in cohorts if identity ==
                'saits_paper_%s_electricity_author_missing0.1_seed%d' % (name, record['seed'])), None)
            if method is None:
                raise ValueError('Only the frozen original Electricity 10-percent slots may change')
            if (record['model_parameters'] != {'saits': 11511000, 'brits': 6993328}[method]
                    or record['full_test_shape'] != [290, 100, 370]
                    or record['held_out_coordinates'] != 1020954
                    or not record['full_original_checkpoint_and_data_identity_verified']
                    or not record['full_test_grid_and_grouped_statistics_replayed']
                    or not record['full_training_completion']['full_training_finished']
                    or record['full_training_completion']['benchmark_smoke']):
                raise ValueError('Every original parameter, test window and training completion required')
            for metric in ('mae', 'rmse', 'mre'):
                original = record['original_metrics'][metric]
                if not math.isfinite(original) or any(not math.isclose(
                        record[field][metric], original, rel_tol=2e-6, abs_tol=2e-6)
                        for field in ('replayed_metrics', 'independently_recomputed_metrics')):
                    raise ValueError('All independently replayed metrics must match')
            cohorts[method].append(record)
    if (sorted(r['seed'] for r in cohorts['saits']) != list(range(26, 31))
            or [r['seed'] for r in cohorts['brits']] != [26]):
        raise ValueError('All five original SAITS seeds and exactly the first original BRITS seed required')
    if len({r['original_full_dataset_sha256'] for rows in cohorts.values() for r in rows}) != 1:
        raise ValueError('Both methods must use the identical frozen full dataset')
    return cohorts


def audited_updates(original, proofs, source, captured_utc):
    cohorts = validated_cohorts(proofs)
    updates = []
    for method, records in cohorts.items():
        for metric in ('mae', 'rmse', 'mre'):
            matches = [row for row in original if row['algorithm'] == method
                and row['dataset'] == 'electricity' and row['task'] == 'imputation'
                and row['protocol'] == 'author' and row['pattern'] == 'paper_mask'
                and row['missing_ratio'] == '0.1' and row['metric'] == metric]
            if len(matches) != 1 or int(matches[0]['required_repeats']) != 5:
                raise ValueError('Exactly the original five-seed registered metric slot required')
            row = dict(matches[0])
            row.update(independent_stats([record['original_metrics'][metric] for record in records]))
            row.update(captured_utc=captured_utc, source=source,
                author_equivalence_certified=False, strict_TAB_result=False,
                comparison='full_native_models_audited_%d_of_5_author_alignment_unverified' % len(records))
            updates.append(row)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full-model publication input changed')
    for path in config['full_model_guard_receipts']:
        guard = legacy.read(ROOT / path)
        if (guard['exit_code'] != 0 or not guard['full_native_model_replay_passed']
                or guard['resource_guard_aborted']):
            raise ValueError('Every actual full native replay must finish without a resource abort')
    proofs = [legacy.read(ROOT / path) for path in config['full_model_proofs']]
    cohorts = validated_cohorts(proofs)
    for records in cohorts.values():
        for record in records:
            for field in ('result', 'checkpoint', 'replay_arrays'):
                path = record[field] if field == 'replay_arrays' else record[field + '_path']
                if legacy.sha(ROOT / path) != record[field + '_sha256']:
                    raise ValueError('Actual full model, result or replay array changed')
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, original = reader.fieldnames, list(reader)
    captured = max(proof['captured_utc'] for proof in proofs)
    proof_path = config['cohort_summary_path']
    updates = audited_updates(original, proofs, proof_path, captured)
    rows = previous.merge_audited_rows(original, updates)
    changed = [i for i, (a, b) in enumerate(zip(original, rows)) if a != b]
    keys = {previous.key(row) for row in updates}
    if len(rows) != len(original) or len(changed) != 6 or any(previous.key(rows[i]) not in keys for i in changed):
        raise ValueError('Only the six existing original Electricity metrics may change')
    measured = sum(row['mean'] not in ('', None) for row in rows)
    if measured != old['measured_metric_rows'] + 3:
        raise ValueError('Exactly the three original BRITS missing metric slots must become measured')
    target = ROOT / config['output_directory']
    if target.exists() or (ROOT / proof_path).exists():
        raise FileExistsError('Preserve all full cohort and metric publications')
    target.mkdir(parents=True)
    cohort_summary = dict(captured_utc=captured, source_proofs=config['full_model_proofs'],
        saits_original_training_seeds=[r['seed'] for r in cohorts['saits']],
        brits_original_training_seeds=[r['seed'] for r in cohorts['brits']],
        full_native_models_replayed=6, full_checkpoint_replay_and_original_completion_verified=True,
        additional_training_seed_slots=0, author_equivalence_certified=False,
        all_original_experiments_complete=False, metric_rows=updates)
    legacy.write(ROOT / proof_path, cohort_summary)
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    for name in ('overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '评分配方',
        '指标', '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集时间UTC']
    values = [[row['task'], row['algorithm'], row['dataset'], row['protocol'], row.get('epochs', ''),
        str(row.get('pattern', '')) + '/' + str(row.get('missing_ratio', '')), row.get('score_recipe', ''),
        row['metric'], '%s/%s' % (row['n'], row['required_repeats']),
        *[legacy.fmt(row.get(key)) for key in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
        row['captured_utc']] for row in rows]
    (target / 'all_metrics.md').write_text('# 完整算法—数据集实验技术指标\n\n'
        '—表示缺结果，0表示实测零。各协议和轨道采集时间分别保留；'
        '清单完整不表示全部实验完成。\n\n' + legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    summary = [[row['algorithm'], row['metric'], '%s/%s' % (row['n'], row['required_repeats']),
        *[legacy.fmt(row.get(key)) for key in ('mean', 'std', 'ci95_low', 'ci95_high', 'paper_mean')]] for row in updates]
    (target / 'README.rst').write_text('完整算法—数据集指标 v22\n==========================\n\n'
        '%d项指标；%d项有实测值；%d项缺结果。仅更新Electricity作者协议、名义10%%缺失率的'
        'SAITS和BRITS六项指标。SAITS为5/5个原始独立训练种子；BRITS为1/5。\n\n' %
        (len(rows), measured, len(rows)-measured) + legacy.table(
        ['算法', '指标', '完成／预定', '均值', 'STD', '95%下界', '95%上界', '论文值'], summary) +
        '\n\n全部六份完整模型通过原环境CPU重推断，每份覆盖290窗口、1020954缺失坐标和全部月份误差统计。'
        'SAITS为11511000参数，BRITS为6993328参数；原训练上限、验证集早停、数据、批大小和既有容差保留。'
        '重推断不增加训练种子；BRITS单种子不提供种子STD或置信区间。'
        '1397与论文1400预处理窗口差异和历史调参缺口仍保留，不能认证作者效果等价。\n\n'
        '其余7277项指标及其采集时间与v21逐行一致。45篇论文、681方法基线消融、273原始任务及采用基线'
        '仍未全部完成。完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。\n', encoding='utf-8')
    counts = Counter(row['task'] for row in rows if row['mean'] not in ('', None))
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=datetime.now(timezone.utc).isoformat(),
        base_publication_utc=old['published_utc'], replaced_metric_rows=6, changed_numeric_metric_rows=6,
        newly_measured_metric_rows=3, aggregate_metric_rows=len(rows), measured_metric_rows=measured,
        missing_metric_rows=len(rows)-measured, measured_task_rows=dict(counts),
        newly_completed_canonical_training_slots=4, saits_electricity_completed_training_seeds=5,
        saits_electricity_required_training_seeds=5, brits_electricity_completed_training_seeds=1,
        brits_electricity_required_training_seeds=5, saits_full_native_models_replayed=5,
        brits_full_native_models_replayed=1, other_track_rows_and_capture_times_preserved=True,
        additional_training_seed_slots=0, all_original_experiments_complete=False)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=legacy.sha(ROOT / args.config)),
        dict(path=proof_path, sha256=legacy.sha(ROOT / proof_path))]
    receipt['outputs'] = {path.name: legacy.sha(path) for path in target.iterdir() if path.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, missing=len(rows)-measured,
        saits_seeds=5, brits_seeds=1, full_models_replayed=6))


if __name__ == '__main__':
    main()
