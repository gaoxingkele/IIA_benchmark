"""Append all520 original image-table obligations without inventing performance."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import shutil

from scripts.flow_matching.publish_brits_two_seed_listing_v27 import ROOT, legacy, previous


def missing_rows(queue, captured, source):
    rows = []
    for target in queue['paper_targets']:
        for metric, printed in target['paper_reported'].items():
            status = ('original_A10080GB_efficiency_pending' if metric == 'FPS' else
                'full_native_original_checkpoint_score_and_replay_pending' if target['full_score_runner_registered'] else
                'original_comparator_training_or_native_adapter_settings_pending')
            rows.append(dict(algorithm='flow_mismatching_original_image/' + target['id'],
                dataset=target['original_dataset_name'], task='image_anomaly_detection',
                protocol='original_image_tables_local_recipe_v1', metric=metric, n=0, required_repeats=5,
                mean='', std='', se='', ci95_low='', ci95_high='', paper_mean=printed,
                author_equivalence_certified=False, strict_TAB_result=False, captured_utc=captured,
                source=source, pattern='paper_table_%d' % target['paper_table'], missing_ratio='',
                comparison=status, score_recipe=json.dumps(target['score_parameters'], sort_keys=True),
                epochs=target.get('required_training_epochs', target['checkpoint_epoch']),
                paper_std='', metric_unit='FPS' if metric == 'FPS' else 'percent',
                selection_boundary='test-label oracle threshold sweep' if metric.endswith('F1_max') else
                    'full original test ranking/region metric; frozen model and hyperparameters',
                paper_spread='', evaluator_repeats='', evaluator_std='', evaluator_se='', evaluator_ci95='',
                uncertainty_scope='five local independent full-training seeds required; no completed native image score',
            ))
    if len(rows) != 520 or any(row['mean'] or row['n'] != 0 for row in rows):
        raise ValueError('Every original520 target must be retained as unmeasured')
    return rows


def publish(config_path):
    config = legacy.read(ROOT / config_path)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen original image inventory source differs')
    queue = legacy.read(ROOT / config['image_queue'])
    readiness = legacy.read(ROOT / config['complete_test_readiness_audit'])
    if (not readiness['passed'] or readiness['complete_original_test_images'] != 3887
            or readiness['complete_original_categories'] != 27 or readiness['native_benchmark_performance']):
        raise ValueError('Complete original-data readiness is not performance')
    if any((ROOT / case['output_directory'] / 'fully_verified_score_result.json').exists() for case in queue['cases']):
        raise ValueError('New measured results require a separate complete performance publication')
    base, target = ROOT / config['base_publication'], ROOT / config['output_directory']
    old_receipt = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); header = reader.fieldnames; before = list(reader)
    if len(before) != 8098 or len(header) != 31 or sum(bool(row['mean']) for row in before) != 2120:
        raise ValueError('Preserve complete original scientific inventory')
    captured = datetime.now(timezone.utc).isoformat()
    added = missing_rows(queue, captured, config['image_queue'])
    if any(row['algorithm'].startswith('flow_mismatching_original_image/') for row in before):
        raise ValueError('Do not duplicate existing original image obligations')
    if target.exists():
        raise FileExistsError('Preserve prior scientific publications')
    target.mkdir(parents=True)
    prefix = (base / 'all_algorithm_dataset_metrics.csv').read_bytes()
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer, fieldnames=header); writer.writerows(added)
    (target / 'all_algorithm_dataset_metrics.csv').write_bytes(prefix + buffer.getvalue().encode('utf-8'))
    with (target / 'flow_mismatching_original_image_metric_inventory.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=header); writer.writeheader(); writer.writerows(added)
    rows = before + [{key: str(row.get(key, '')) for key in header} for row in added]
    for file in base.iterdir():
        if file.is_file() and file.name not in {'all_algorithm_dataset_metrics.csv', 'all_metrics.md', 'results.html', 'README.rst', 'publication_validation.json'}:
            shutil.copyfile(file, target / file.name)
    boundary = ('原8,098项指标与采集时间完整保留为字节相同前缀；新增原始图像表格520项待测指标，'
        '共8,618项，已有实测2,120项，缺结果6,498项。新增项均n=0、均值/不确定性留空。'
        '对应全量VisA/MVTec，325个设置-种子评分与完整模型重放任务已入队，485个论文目标接入评分代码，'
        '其余35个竞争方法目标及全部其他义务保留。30个原A10080GB FPS目标不能用本地RTX3090时间冒充。'
        '3,887个原始测试图像/27类别的完整真值审计通过，但没有新原始图像性能结果。')
    (target / 'all_metrics.md').write_text('# 完整算法—数据集实验指标 v40\n\n' + boundary + '\n\n' +
        legacy.table(header, [[row.get(k) or '—' for k in header] for row in rows]) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    (target / 'README.rst').write_text('完整算法—数据集指标 v40\n==========================\n\n' + boundary +
        '\n\n严格TAB与作者实现等价仍未认证。完整45论文/681方法和消融/273原始数据任务及全部附加任务保持。'
        '图像F1-max为测试集阈值扫描；AUPRO连通性/截断、推断精度和配对探针是已公开登记的本地选择。'
        '全部每类别/玩具/定性/其他算法义务仍需完成，520项只覆盖当前逐项转录的论文表格。\n', encoding='utf-8')
    metadata = ('strict_groups', 'strict_complete_groups', 'strict_partial_groups', 'cfm_completed_canonical_slots',
        'cfm_complete_groups', 'grasp_complete_cohorts', 'brits_electricity_completed_training_seeds',
        'brits_electricity_required_training_seeds', 'brits_full_native_models_replayed')
    receipt = {key: old_receipt[key] for key in metadata if key in old_receipt}
    receipt.update(published_utc=captured, base_publication=config['base_publication'],
        base_receipt_sha256=legacy.sha(base / 'publication_validation.json'), aggregate_metric_rows=8618,
        measured_metric_rows=2120, missing_metric_rows=6498, appended_original_image_metric_rows=520,
        newly_measured_metric_rows=0, changed_numeric_metric_rows=0, newly_completed_canonical_training_slots=0,
        algorithm_configuration_names=len({row['algorithm'] for row in rows}),
        task_rows=dict(Counter(row['task'] for row in rows)),
        measured_task_rows=dict(Counter(row['task'] for row in rows if row['mean'])),
        original_scope=queue['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False, performance_rows_changed=False,
        base_numeric_csv_prefix_byte_identical=True, all_previous8098_records_and_capture_times_exact=True,
        original_image_full_native_case_seed_evaluations=325, original_image_performance_results=0,
        original_A10080GB_FPS_targets=30, original_A100_hardware_equivalence_certified=False,
        original_remaining_comparator_metric_targets=35, original_runnable_image_metric_targets=485,
        source_receipts=old_receipt['source_receipts'] + config['source_receipts'] + [
            dict(path=config_path, sha256=legacy.sha(ROOT / config_path))])
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(dict(total_metric_slots=8618, measured_slots=2120, missing_slots=6498,
               appended_unmeasured_original_image_targets=520, old8098_prefix_byte_identical=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    publish(parser.parse_args().config)
