"""Add every registered native Energy command without changing previous scores."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import shutil
import psutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.gtgan_energy_native_common_v1 import full_budget, verify_recipe
from scripts.flow_matching.run_gtgan_energy_native_queue_v1 import construction_ready

ROOT, legacy = previous.ROOT, previous.legacy


def pending_energy(model, queue, stamp, headers, source):
    cases = model['cases'] + model['release_evaluation_cases']
    if len(cases) != 4 or len(queue['jobs']) != 4 or {j['case_id'] for j in queue['jobs']} != {c['id'] for c in cases}:
        raise ValueError('All four original Energy training and released-weight commands required')
    modes = Counter((c.get('execution_mode') == 'native_release_checkpoint_evaluation', c['missing_rate']) for c in cases)
    if modes != Counter({(release, rate): 1 for release in [False, True] for rate in [0.0, 0.5]}):
        raise ValueError('Both Energy masks and both original execution modes required')
    rows = []
    for case in cases:
        budget = full_budget(case)
        release = budget['native_release_checkpoint_evaluation']
        for metric in ['discriminative', 'predictive']:
            row = dict.fromkeys(headers, '')
            row.update(algorithm='GT-GAN/native-original/released-checkpoint' if release else 'GT-GAN/native-original',
                dataset='energy', task='unconditional_time_series_generation',
                protocol='GT-GAN-author-command/' + case['id'], metric=metric,
                n=0, required_repeats=1, captured_utc=stamp, source=source,
                missing_ratio=case['missing_rate'], author_equivalence_certified=False, strict_TAB_result=False,
                comparison='pending_full_original_release_evaluation' if release else 'pending_full_original_from_scratch_training_and_evaluation',
                score_recipe='original_energy_discriminator_300_steps' if metric == 'discriminative' else budget['predictive_function'] + '_5000_steps',
                evaluator_repeats=10,
                metric_unit='native_abs_accuracy_minus_0.5' if metric == 'discriminative' else 'native_normalized_last_feature_MAE' if case['missing_rate'] == 0 else 'native_normalized_all28_features_MAE',
                selection_boundary='original global normalization and all overlapping windows; fixed released checkpoint,zero generator training updates' if release else 'original global normalization and all overlapping windows; full5000 embedding plus6000/8500 joint updates; checkpoint choice unproved',
                uncertainty_scope='one fixed released generator; evaluator repeats are not generator seeds' if release else 'one original hardcoded7777 generator seed; evaluator repeats are not generator seeds')
            rows.append(row)
    legacy.validate_rows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = legacy.read(ROOT / a.config)
    for item in c['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen Energy publication evidence changed')
    base = ROOT / c['base_publication']
    old = previous.verified_capture(base, 'publication_validation.json')
    queue = legacy.read(ROOT / c['energy_queue'])
    for item in queue['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Full Energy execution registration changed')
    model = legacy.read(ROOT / queue['model_config'])
    for case in model['cases'] + model['release_evaluation_cases']:
        verify_recipe(case, ROOT / model['source_root'])
    ready = construction_ready(queue)
    if not ready:
        expected = queue['construction_audit_parent']
        parent = psutil.Process(expected['pid'])
        if parent.create_time() != expected['create_time'] or queue['construction_audit_config'] not in parent.cmdline():
            raise ValueError('The registered full data audit must be currently live or fully passed')
    for job in queue['jobs']:
        if (ROOT / job['output_directory'] / 'result.json').exists():
            raise ValueError('New full result needs an independent complete audit, not a pending row')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, original = reader.fieldnames, list(reader)
    stamp = datetime.now(timezone.utc).isoformat()
    added = pending_energy(model, queue, stamp, headers, c['energy_queue'])
    rows = original + added
    if len({previous.key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate canonical metric slots')
    measured = sum(r['mean'] not in ('', None) for r in rows)
    if len(rows) != old['aggregate_metric_rows'] + 8 or measured != old['measured_metric_rows']:
        raise ValueError('Only eight pending original Energy metric slots may be added')
    target = ROOT / c['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every published metric inventory')
    target.mkdir(parents=True)
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with (target / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        captured = list(csv.DictReader(stream))
    if captured[:len(original)] != original:
        raise ValueError('An existing result or capture timestamp changed')
    for name in ['overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv']:
        shutil.copyfile(base / name, target / name)
    columns = ['任务', '算法／消融', '数据集', '协议', '缺失率', '评分配方', '指标', '完成／预定',
        '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '采集UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], str(r.get('missing_ratio', '')),
        r.get('score_recipe', ''), r['metric'], f"{r['n']}/{r['required_repeats']}",
        *[legacy.fmt(r.get(k)) for k in ['mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean']], r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text('# 完整算法—数据集技术指标\n\n'
        '本版逐行保留v18全部结果，仅增加GT-GAN原始Energy四命令的八项待完成指标。'
        '—表示缺结果；发布权重评估与从头训练分别登记，评估器重复不另计生成器种子。\n\n'
        + legacy.table(columns, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    job_rows = []
    for case in model['cases'] + model['release_evaluation_cases']:
        job = next(j for j in queue['jobs'] if j['case_id'] == case['id'])
        b = full_budget(case)
        job_rows.append(dict(job_id=job['id'], dataset='energy', missing_rate=case['missing_rate'],
            mode='released_checkpoint_evaluation' if b['native_release_checkpoint_evaluation'] else 'from_scratch_full_training',
            complete=False, embedding_updates=b['native_embedding_steps'], joint_steps=b['native_joint_steps'],
            full_evaluations=b['full_periodic_evaluations'], windows=b['full_generation_windows'],
            variables=b['variables'], sequence_length=b['sequence_length'], evaluator_repeats=10,
            discriminative_updates=300, predictive_updates=5000, queue=c['energy_queue']))
    with (target / 'original_energy_jobs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(job_rows[0]))
        writer.writeheader()
        writer.writerows(job_rows)
    counts = Counter(r['task'] for r in rows)
    ready = Counter(r['task'] for r in rows if r['mean'] not in ('', None))
    proof_boundary = '完整两个缺失率的数据/五网络构造/发布权重兼容检查通过。' if ready else '完整两个缺失率的数据/模型核验仍在运行，尚未称全量核验通过。'
    (target / 'README.rst').write_text('完整算法—数据集指标 v19\n==========================\n\n'
        f'发布UTC：{stamp}。共{len(rows)}项指标，{measured}项有实测值，{len(rows)-measured}项缺完整结果；'
        f'{len({r["algorithm"] for r in rows})}个算法／配置名称。逐行保留v18全部成绩、协议及采集时间，没有新增实测成绩。\n\n'
        '全部结果：results.html；完整CSV：all_algorithm_dataset_metrics.csv；全部文字指标：all_metrics.md。'
        '异常检测矩阵及原生成任务CSV保留原快照；新增Energy四原始命令见original_energy_jobs.csv。\n\n'
        'GT-GAN Energy' + proof_boundary + '该核验不等于训练或推断完成。'
        '四条README原命令包括两条发布权重评估及两条从头训练；总31次全量评估、620次原生评估器训练。'
        '从头训练各5000次嵌入及6000/8500联合步，完整19712x24x28数据。'
        'Energy判别器原生300步，预测器5000步；0%预测末变量，50%预测全部28变量，不能将两者当成同一目标的MAE。\n\n'
        '登记及证据：../gtgan_energy_native_execution_v1/README.rst。仅有一个原生7777训练种子的预定槽；'
        '固定发布权重没有新的生成器训练种子，十次评估器训练不是十次生成器训练。'
        '原方向45论文/681方法基线消融/273原始任务及其采用基线仍未全部完成；不声称作者等价或完整TAB认证。\n', encoding='utf-8')
    receipt = copy.deepcopy(old)
    receipt.update(published_utc=stamp, base_publication_utc=old['published_utc'], aggregate_metric_rows=len(rows),
        measured_metric_rows=measured, missing_metric_rows=len(rows) - measured, task_rows=dict(counts), measured_task_rows=dict(ready),
        algorithm_configuration_names=len({r['algorithm'] for r in rows}), replaced_metric_rows=0, changed_numeric_metric_rows=0,
        newly_measured_metric_rows=0, added_pending_metric_rows=8, gtgan_energy_original_commands=4,
        gtgan_energy_original_from_scratch_commands=2, gtgan_energy_original_release_evaluation_commands=2,
        gtgan_energy_full_native_data_model_proof_completed=ready,
        gtgan_energy_native_full_command_completions=0, other_track_rows_and_capture_times_preserved=True,
        all_original_experiments_complete=False)
    receipt['source_receipts'] += c['source_receipts'] + [dict(path=a.config, sha256=legacy.sha(ROOT / a.config))]
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=measured, pending_energy=8, existing_metric_rows_changed=0))


if __name__ == '__main__':
    main()
