"""Merge a full-model-audited SAITS seed increment into the complete inventory."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

ROOT, legacy = previous.ROOT, previous.legacy


def audited_updates(original, proof, source):
    if (not proof['full_checkpoint_contents_and_re_inference_passed']
            or proof['full_models_replayed'] != 2 or len(proof['records']) != 2
            or sorted(r['seed'] for r in proof['records']) != [26, 27]
            or proof['additional_training_seed_slots'] != 0):
        raise ValueError('Both originally registered full trained-model proofs required')
    for r in proof['records']:
        if (r['model_parameters'] != 11511000 or r['full_test_shape'] != [290, 100, 370]
                or r['held_out_coordinates'] != 1020954 or r['full_test_batches'] != 3
                or not r['full_original_checkpoint_and_data_identity_verified']
                or not r['full_test_grid_and_grouped_statistics_replayed']):
            raise ValueError('Every native full test point/model parameter is required')
    updates = []
    for metric in ('mae', 'rmse', 'mre'):
        matches = [r for r in original if r['algorithm'] == 'saits' and r['dataset'] == 'electricity'
                   and r['protocol'] == 'author' and r['missing_ratio'] == '0.1' and r['metric'] == metric]
        if len(matches) != 1 or int(matches[0]['required_repeats']) != 5:
            raise ValueError('Exactly the frozen original five-seed metric slot is required')
        row = dict(matches[0])
        row.update(independent_stats([r['original_metrics'][metric] for r in proof['records']]))
        row.update(captured_utc=proof['captured_utc'], source=source, author_equivalence_certified=False,
                   comparison='full_native_models_audited_partial_2_of_5_author_alignment_unverified',
                   strict_TAB_result=False)
        updates.append(row)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full-model publication evidence changed')
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers, original = reader.fieldnames, list(reader)
    proof = legacy.read(ROOT / config['full_model_proof'])
    updates = audited_updates(original, proof, config['full_model_proof'])
    rows = previous.merge_audited_rows(original, updates)
    changed = [i for i, (a, b) in enumerate(zip(original, rows)) if a != b]
    if len(changed) != 3 or any(previous.key(rows[i]) not in {previous.key(r) for r in updates} for i in changed):
        raise ValueError('Only the three frozen SAITS metrics may change')
    measured = sum(r['mean'] not in ('', None) for r in rows)
    if len(rows) != old['aggregate_metric_rows'] or measured != old['measured_metric_rows']:
        raise ValueError('Existing partially measured slots must be retained')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Result publications are immutable')
    target.mkdir(parents=True)
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    for name in ('overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '评分配方',
               '指标', '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集时间UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs', ''),
               str(r.get('pattern', '')) + '/' + str(r.get('missing_ratio', '')), r.get('score_recipe', ''),
               r['metric'], f"{r['n']}/{r['required_repeats']}",
               *[legacy.fmt(r.get(k)) for k in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
               r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text('# 完整算法—数据集技术指标\n\n'
        '—为缺结果，0为实测零。不同协议分列；每个轨道保留自己的采集时间。'
        '清单完整不表示全部原始实验完成。\n\n' + legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    summary = [[r['metric'], f"{r['n']}/{r['required_repeats']}",
                *[legacy.fmt(r[k]) for k in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')]] for r in updates]
    (target / 'README.md').write_text('# 完整算法—数据集实验指标\n\n'
        f'发布UTC：{now}。共{len(rows)}条指标，{measured}条有实测值、{len(rows)-measured}条缺完整结果。'
        '覆盖137个算法／配置名称；指标行数不是独立训练次数。全部原始实验尚未完成。\n\n'
        '[可搜索全部结果](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；'
        '[全部文字指标](all_metrics.md)；[异常检测F1总览](overview.md)；'
        '[异常检测全部指标](strict_algorithm_dataset.md)。\n\n'
        '本版仅更新SAITS／Electricity／作者协议／名义10%缺失率的三个既有指标槽位，'
        '从1/5增至2/5训练种子。两份实际11511000参数权重均通过全部290窗口、'
        '1020954个缺失坐标和逐月误差统计的完整CPU重推断。'
        '[模型和训练完整性证据](../saits_electricity_full_models_v1/README.md)。\n\n' +
        legacy.table(['指标', '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值'], summary) + '\n\n'
        '均值和模型种子不确定性使用原完整训练结果；CPU重推断只核验保存权重与结果，'
        '不另计训练种子。仅两个种子的t区间较宽，不能据此认证作者效果一致。'
        '原作者预处理窗口1397与论文1400的差异、历史设置缺口仍保留。\n\n'
        'CFM-TS仍为84/285个规范任务完成；84个已完成模型已通过全量权重重推断，'
        '[独立证据](../cfm_full_model_and_recovery_v3/README.md)。'
        '其他轨道的结果、缺口和采集时间与v16逐行保留；本次不重新采集异常检测或生成轨道。'
        '验证集定阈值无PA、测试标签选阈值、PA和作者原生协议均分别列出；'
        '尚未获完整TAB预算或作者等价认证。\n', encoding='utf-8')
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=now, base_publication_utc=old['published_utc'],
                   replaced_metric_rows=3, changed_numeric_metric_rows=3,
                   aggregate_metric_rows=len(rows), measured_metric_rows=measured,
                   missing_metric_rows=len(rows)-measured, newly_measured_metric_rows=0,
                   newly_completed_canonical_training_slots=1,
                   saits_electricity_completed_training_seeds=2,
                   saits_electricity_required_training_seeds=5,
                   saits_full_native_models_replayed=2,
                   cfm_all_completed_full_model_replays_verified=84,
                   other_track_rows_and_capture_times_preserved=True,
                   all_original_experiments_complete=False)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=legacy.sha(ROOT / args.config))]
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, missing=len(rows)-measured, changed_metric_rows=3,
               saits_original_training_seeds=2))


if __name__ == '__main__':
    main()
