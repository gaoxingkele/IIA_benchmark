"""Independently verify industrial raw/derived identity and full result coverage."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.model_registry import load_model_config
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import complete_result
from scripts.flow_matching.run_tsad_experiment import verify_job, evaluate_saved
from scripts.flow_matching.run_industrial_tsad_experiment import pooled_training_windows
from scripts.flow_matching.register_industrial_tsad import model_window
from scripts.flow_matching.summarize_tsad_execution import intervals


def audit(root, settings_path):
    settings = load_model_config(settings_path)
    registration = load_model_config(root/settings['registration_report'])
    if sha(settings_path) != registration['config_sha256']:
        raise ValueError('Industrial registration configuration changed')
    for receipt in registration['queues'].values():
        if sha(root/receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen industrial queue/protocol changed')
    manifest_path = root/settings['prepared_root']/'manifest.json'
    manifest = load_model_config(manifest_path)
    records, counts, raw, datasets = [], {'completed': 0, 'without_complete_result': 0}, {}, []
    for dataset in manifest['datasets']:
        if sha(root/dataset['input_npz']) != dataset['input_sha256']:
            raise ValueError('Industrial prepared arrays changed')
        role_groups = [dataset['training_groups'], dataset['validation_groups'], dataset['entities']]
        group_sets = [{g['group_id'] for g in groups} for groups in role_groups]
        if any(group_sets[a] & group_sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError('Industrial source group role contamination')
        for source in dataset['source_files']:
            if sha(root/source['path']) != source['sha256']:
                raise ValueError('Industrial raw source changed')
            raw[source['path']] = source['sha256']
        datasets.append({'dataset': dataset['id'], 'input_sha256': dataset['input_sha256'],
                         'training_points': dataset['scaler']['fit_points'], 'test_entities': len(dataset['entities']),
                         'test_points': sum(g['test_points'] for g in dataset['entities']), 'selection_audit': dataset['selection_audit']})
    frozen = {}
    queues = [load_model_config(root/p) for p in settings['queue_paths'].values()]
    for queue in queues:
        for job in queue['jobs']:
            for source in job['frozen_files']:
                if source['path'] in frozen and frozen[source['path']] != source['sha256']:
                    raise ValueError('Industrial frozen identities inconsistent')
                frozen[source['path']] = source['sha256']
            if not complete_result(root, job):
                counts['without_complete_result'] += 1
                continue
            output = root/job['output_directory']
            result = load_model_config(output/'result.json')
            dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
            with np.load(root/dataset['input_npz'], allow_pickle=False) as source:
                data = {k: source[k].copy() for k in source.files}
            with np.load(output/'scores.npz', allow_pickle=False) as source:
                scores = {k: source[k].copy() for k in source.files}
            entities = {}
            for role, groups in [('train', dataset['training_groups']), ('validation', dataset['validation_groups']), ('test', dataset['entities'])]:
                for group in groups:
                    prefix = group['array_prefix']
                    values = scores[prefix+'_'+role]
                    if values.shape != (group['points'],) or not np.isfinite(values).all():
                        raise ValueError('Industrial result lacks complete native timestamps')
                    for suffix in ['_labels', '_timestamps', '_source_rows', '_fault_labels']:
                        if prefix+suffix in data and not np.array_equal(scores[prefix+suffix], data[prefix+suffix]):
                            raise ValueError('Industrial native identity/labels changed')
                    if role == 'test':
                        entities[group['entity_id']] = (scores[prefix+'_labels'], values)
            training = np.concatenate([scores[g['array_prefix']+'_train'] for g in dataset['training_groups']])
            validation = np.concatenate([scores[g['array_prefix']+'_validation'] for g in dataset['validation_groups']])
            independent = evaluate_saved(entities, validation, training, dataset['input_sha256'], result['checkpoint_sha256'], queue['evaluation'], job['seed'])
            if independent != result['metrics']:
                raise ValueError('Industrial saved metrics do not reproduce from full saved scores')
            configuration = load_model_config(root/job['model_config'])
            fit, budget = pooled_training_windows(data, dataset, model_window(configuration, settings['default_window']))
            if budget != result['fit_group_budget'] or len(fit) != result['unique_fit_windows']:
                raise ValueError('Industrial fit group budget differs from full native groups')
            expected_epochs = configuration['parameters']['epochs']+(configuration['parameters'].get('reflow_stages', 1)-1)*configuration['parameters'].get('reflow_epochs', 0)
            if result['epoch_audit']['declared_epochs'] != expected_epochs:
                raise ValueError('Industrial training budget changed')
            if result['training_losses'] is not None and (len(result['training_losses']) != expected_epochs or not np.isfinite(result['training_losses']).all()):
                raise ValueError('Industrial loss history lacks complete finite epochs')
            counts['completed'] += 1
            records.append({'id': job['id'], 'dataset': job['dataset'], 'seed': job['seed'], 'model_config': job['model_config'],
                            'result_path': (output/'result.json').relative_to(root).as_posix(), 'result_sha256': sha(output/'result.json'),
                            'checkpoint_sha256': result['checkpoint_sha256'], 'scores_sha256': result['scores_sha256'],
                            'unique_fit_windows': len(fit), 'test_points': result['test_points'], 'epoch_audit': result['epoch_audit'],
                            'parameter_count': result['parameter_count'], 'training_seconds': result['training_seconds'], 'score_seconds': result['score_seconds'],
                            'primary_metrics': result['metrics']['strict'][str(queue['evaluation']['primary_false_alarm_percent'])]['micro'],
                            'metrics_independently_recomputed': True, 'original_labels_and_rows_identical': True})
    verify_job(root, {'frozen_files': [{'path': p, 'sha256': digest} for p, digest in frozen.items()]})
    groups = defaultdict(list)
    for record in records:
        groups[record['model_config'], record['dataset']].append(record)
    aggregates = [{'model_config': model, 'dataset': dataset, 'completed_seeds': len(values), 'required_seeds': len(settings['seeds']),
                   'metrics': {name: intervals([v['primary_metrics'][name] for v in values if v['primary_metrics'][name] is not None])
                               for name in ['precision', 'recall', 'f1', 'auroc', 'average_precision']
                               if any(v['primary_metrics'][name] is not None for v in values)}}
                  for (model, dataset), values in groups.items()]
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'registered_jobs': sum(len(q['jobs']) for q in queues),
              'counts': counts, 'frozen_files_verified': len(frozen), 'raw_sources_verified': len(raw),
              'data_manifest_sha256': sha(manifest_path), 'datasets': datasets, 'completed_records': records, 'seed_aggregates': aggregates,
              'goal_achieved': False, 'boundary': settings['boundary']}
    target = root/'projects/flow_matching_research/results/2026-10-09/industrial_detection'
    write_json(target/'validation.json', report)
    (target/'data_manifest.json').write_bytes(manifest_path.read_bytes())
    lines = ['# 工业异常检测完整数据执行', '',
             f"快照：{report['captured_utc']}。登记 {report['registered_jobs']} 项，完整结果 {counts['completed']} 项。", '',
             '56个配置 × 5种数据划分 × 5个种子。CPU550项、GPU850项。保留完整容量和训练轮数；完整结果必须保存模型和每个原始时间点的分数。', '',
             '| 数据划分 | 训练点数 | 测试运行/天数 | 完整测试点数 |', '|---|---:|---:|---:|']
    for dataset in datasets:
        lines.append(f"| {dataset['dataset']} | {dataset['training_points']} | {dataset['test_entities']} | {dataset['test_points']} |")
    lines += ['', '训练/验证/测试按整个运行或日期隔离。TEP训练d00.dat、验证d00_te.dat、测试全部21个故障运行；SKAB验证other/1.csv，其余33个实验测试；PRONTO轮换整日测试，训练/验证只保留各自日期的全部Normal连续段。所有分段单独开窗，不跨故障间隙；共享训练只拟合一次。', '',
              'SKAB验证包含异常，1%是验证总体尾部分位数，不能称正常总体误报率保证。PRONTO训练/校准的正常段选择使用原标签，明确属于标签辅助协议；三个日期轮换有关联。PSM基线超参数直接迁移，未经目标数据调优。经典TEP不代表多工况HDF5，工业迁移不代表原论文等效复现。', '',
              '| 算法配置 | 数据划分 | 种子 | Precision | Recall | 点级F1 | AUROC | AP |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        m = record['primary_metrics']
        values = ' | '.join('—' if m[k] is None else f'{m[k]:.6f}' for k in ['precision', 'recall', 'f1', 'auroc', 'average_precision'])
        lines.append(f"| {Path(record['model_config']).stem} | {record['dataset']} | {record['seed']} | {values} |")
    lines += ['', '这里只列已通过完整源数据、模型/分数哈希、覆盖和指标独立重算检查的完整结果；不是全部队列已经结束。预检不计性能。TAB VUS/Affiliation由独立固定版本队列跟随补算。', '',
              'validation.json的seed_aggregates保存跨种子均值、STD、SE与t区间；这些区间描述初始化变异，不证明跨算法等效或不同运行的独立性。', '',
              '[完整审计和逐次结果](validation.json)、[原始数据与划分审计](data_manifest.json)。原始论文、消融、多工况以及其他任务的已知缺口继续保留。', '']
    (target/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'counts': counts, 'registered_jobs': report['registered_jobs'], 'raw_sources_verified': len(raw), 'frozen_files_verified': len(frozen)}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    args = parser.parse_args()
    audit(ROOT, args.config)


if __name__ == '__main__':
    main()
