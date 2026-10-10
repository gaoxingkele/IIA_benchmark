"""Update four released GiFlow aggregates after a second complete native seed."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

ROOT, legacy = previous.ROOT, previous.legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen second native seed publication input changed')
    base = ROOT / config['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    capture = ROOT / config['giflow_capture']
    receipt = previous.verified_capture(capture, 'validation.json')
    audit = legacy.read(ROOT / config['full_array_audit'])
    if (receipt['canonical_slots'] != 140 or receipt['seed_groups'] != 28 or receipt['job_counts']['completed'] != 3
            or audit['job_id'] != 'giflow_native__released_mirror__air36__main__seed394'
            or audit['seed'] != 394 or audit['track'] != 'released_mirror'
            or audit['actual_full_training_epochs'] != 196 or audit['actual_optimizer_updates'] != 9408
            or audit['configured_maximum_epochs'] != 300 or audit['full_test_batches'] != 14
            or audit['masked_targets_with_window_repetitions'] != 264837
            or audit['full_test_arrays_and_metrics_passed'] is not True or audit['author_equivalence_certified']):
        raise ValueError('Complete native second seed, original scope and entire test audit required')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, original = reader.fieldnames, list(reader)
    with (capture / 'algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        captured = list(csv.DictReader(stream))
    updates = [r for r in captured if r['algorithm'] == 'GiFlow/main' and r['dataset'] == 'air36'
               and r['protocol'] == 'giflow_native/released_mirror']
    if len(updates) != 4 or any(int(r['n']) != 2 or int(r['required_repeats']) != 5 for r in updates):
        raise ValueError('Exactly four original aggregates, two of five native training seeds required')
    for row in updates:
        row.update(source=config['giflow_capture'] + '/algorithm_dataset_metrics.csv',
                   comparison='two_full_native_seeds_arrays_audited_seed394_model_replay_in_flight',
                   strict_TAB_result='False')
    merged = previous.merge_audited_rows(original, updates)
    changed = [i for i, (a, b) in enumerate(zip(original, merged)) if a != b]
    if len(merged) != 7283 or len(changed) != 4 or any(previous.key(merged[i]) not in {previous.key(r) for r in updates} for i in changed):
        raise ValueError('Only four released native metrics may change; all other tasks remain')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every immutable metric publication')
    target.mkdir(parents=True)
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{k: r.get(k, '') for k in headers} for r in merged])
    with (target / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        exported = list(csv.DictReader(stream))
    if any(a != b for i, (a, b) in enumerate(zip(original, exported)) if i not in changed):
        raise ValueError('Other numerical rows, metadata or capture timestamps changed')
    for name in ['overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv']:
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '评分配方', '指标',
               '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs', ''),
               str(r.get('pattern', '')) + '/' + str(r.get('missing_ratio', '')), r.get('score_recipe', ''),
               r['metric'], f"{r['n']}/{r['required_repeats']}",
               *[legacy.fmt(r.get(k)) for k in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
               r['captured_utc']] for r in merged]
    (target / 'all_metrics.md').write_text('# 全部算法—数据集技术指标\n\n各协议单列；—为缺失，0为实测零。'
        '本轮仅更新GiFlow发布源码分支四项聚合，其余逐行保留。完整清单不表示全部实验已完成。\n\n' +
        legacy.table(columns, values) + '\n', encoding='utf-8', newline='\n')
    (target / 'results.html').write_text(legacy.interactive_html(merged), encoding='utf-8', newline='\n')
    now = datetime.now(timezone.utc).isoformat()
    summary = legacy.table(['原生指标', '均值', 'STD', '完成／预定'],
                           [[r['metric'], legacy.fmt(r['mean']), legacy.fmt(r['std']), '2/5'] for r in updates])
    (target / 'README.rst').write_text('完整算法—数据集指标 v21\n==========================\n\n'
        f'发布UTC：{now}。7283项指标、1790项有实测值、5493项尚缺完整结果。仅更新GiFlow发布源码分支'
        'Air36/main/20%point的四项聚合，独立完整训练种子由1/5增至2/5；条目数不是实验次数。\n\n'
        '全部指标：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。\n\n' + summary + '\n\n'
        '新增seed394按原300轮上限、40轮早停完成196轮/9408次更新；全量1747窗口/14批次/'
        '264837缺失目标数组与原生批次均值指标独立核验通过。seed393为247轮/11856次更新。'
        'STD/SE/Student-t区间来自两个独立训练种子；不增加重复推断种子。'
        '发布源码的test-in-validation和每批EMA重置限制保留，不能称作者等价或严格TAB成绩。\n\n'
        'seed394完整原生检查点重推断已启动，当前未声明完成；原始20步Euler、完整模型、数据和冻结容差均保留。'
        '此前两个seed393分支的完整模型重推断证据继续保留。\n\n'
        '新增证据：../giflow_released_seed394_full_arrays_v1/full_array_audit.json；'
        '全部140原始槽：../giflow_canonical_listing_v4/canonical_jobs.csv；'
        '完整模型执行：../giflow_released_seed394_native_execution_v1/launch.json。\n\n'
        '其余7279条指标与各轨道原采集时间不变。45论文/681方法基线消融/273原始任务及采用基线'
        '仍未全部完成；所有未完成范围继续保留。\n', encoding='utf-8', newline='\n')
    validation = copy.deepcopy(old)
    validation.update(published_utc=now, base_publication_utc=old['published_utc'],
        giflow_complete_slots=3, giflow_released_air36_main_training_seeds=2,
        giflow_released_air36_main_required_training_seeds=5,
        newly_completed_canonical_training_slots=1, replaced_metric_rows=4, changed_numeric_metric_rows=4,
        newly_measured_metric_rows=0, additional_training_seed_slots=0,
        performance_rows_changed=True, added_pending_metric_rows=0,
        giflow_seed394_full_arrays_verified=True, giflow_seed394_full_model_replay_complete=False,
        other_track_rows_and_capture_times_preserved=True, all_numeric_tables_byte_identical_to_base=False,
        all_original_experiments_complete=False)
    validation['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=legacy.sha(ROOT / args.config))]
    validation['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', validation)
    print(dict(rows=7283, measured=1790, updated_metrics=4, released_seed_count=2, required_seeds=5))


if __name__ == '__main__':
    main()
