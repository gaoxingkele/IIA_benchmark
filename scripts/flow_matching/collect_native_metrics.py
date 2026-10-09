"""Separate native protocols from strict TSAD, with full entity coverage audits."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from scripts.flow_matching.mtsbench_stat_protocol import sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_stat_result(job, result, manifest, root, raw_checked):
    """An artifact-valid result must also contain every registered raw test point."""
    import numpy as np
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    expected = {e['entity_id']: e for e in dataset['entities']}
    if [e['entity'] for e in result['entities']] != [e['id'] for e in job['entities']]:
        raise ValueError('Native statistical entity coverage differs from registration')
    for registered, actual in zip(job['entities'], result['entities']):
        entity = expected[registered['id']]
        for key, value in [('points', entity['test_points']), ('features', entity['features']),
                           ('anomaly_points', entity['test_anomaly_points'])]:
            if actual[key] != value:
                raise ValueError('Native statistical raw coverage differs: ' + key)
        raw = root / registered['test_path']
        if str(raw) not in raw_checked:
            if sha(raw) != registered['test_sha256']:
                raise ValueError('Native statistical raw data changed')
            raw_checked.add(str(raw))
        score = next(a for a in result['artifacts']
                     if Path(a['path']).name == registered['id'] + '_scores.npz')
        with np.load(root / score['path'], allow_pickle=False) as data:
            if any(data[k].shape != (actual['points'],) for k in ('scores', 'labels', 'timestamps')):
                raise ValueError('Native statistical score coverage differs')
            if not np.isfinite(data['scores']).all() or not np.isin(data['labels'], [0, 1]).all():
                raise ValueError('Invalid native statistical score or label')
            if int(data['labels'].sum()) != actual['anomaly_points']:
                raise ValueError('Native statistical anomaly coverage differs')
    for key in ('PRC', 'ROC', 'Best_F1'):
        values = [e['metrics'][key] for e in result['entities']]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError('Invalid native statistical metric')
        if not math.isclose(result['metrics'][key], sum(values) / len(values), abs_tol=1e-12):
            raise ValueError('Native statistical entity macro differs')


def native_tables(root, jobs, settings):
    from scripts.flow_matching.export_complete_metrics import numeric_leaves
    from scripts.flow_matching.summarize_grasp_protocol import aggregate
    rows, leaves, entities = [], [], []
    groups = defaultdict(list)
    manifest_path = root / settings['statistical_coverage_manifest'] if settings.get('statistical_coverage_manifest') else None
    manifest = read(manifest_path) if manifest_path else None
    checked = set()
    for job in jobs:
        groups[job['algorithm'], job['recipe'], job['dataset'], job['track']].append(job)
        if job['status'] != 'completed':
            continue
        result_path = root / job['output_directory'] / 'result.json'
        result = read(result_path)
        if sha(result_path) != job['result_sha256']:
            raise ValueError('Native result changed during capture')
        if job['algorithm'] == 'grasp_mtsbench_statistical_replay':
            if manifest is None:
                raise ValueError('Full statistical coverage manifest required')
            queue = read(root / job['queue'])
            registered = next(j for j in queue['jobs'] if j['id'] == job['id'])
            receipt = next(r for r in queue['source_receipts'] if r['path'] == settings['statistical_coverage_manifest'])
            if sha(manifest_path) != receipt['sha256']:
                raise ValueError('Frozen statistical coverage manifest changed')
            audit_stat_result(registered, result, manifest, root, checked)
        common = {k: job[k] for k in ('algorithm', 'recipe', 'dataset', 'seed', 'id', 'track', 'result_sha256')}
        common['result_path'] = result_path.relative_to(root).as_posix()
        for field, value in numeric_leaves(result):
            leaves.append(dict(common, metric_path=field, value=value))
        for entity in result.get('entities', []):
            entities.append(dict(common, **entity))
    for (algorithm, recipe, dataset, protocol), members in sorted(groups.items()):
        ready = [j for j in members if j['status'] == 'completed']
        keys = sorted({k for j in ready for k, _ in numeric_leaves(j['metrics'])})
        if not keys:
            keys = ['PRC', 'ROC', 'Best_F1'] if algorithm == 'grasp_mtsbench_statistical_replay' else []
        for metric in keys:
            values = [dict(numeric_leaves(j['metrics']))[metric] for j in ready]
            stats = aggregate(values)
            deterministic = algorithm == 'grasp_mtsbench_statistical_replay'
            if deterministic:
                # Ten deterministic refits are not ten independent stochastic experiments.
                stats.update(se=None, ci95_low=None, ci95_high=None)
            rows.append({'algorithm': algorithm, 'recipe': recipe, 'dataset': dataset, 'protocol': protocol,
                         'metric': metric, **stats, 'required_seeds': len(members),
                         'status': 'completed' if len(ready) == len(members) else 'partial' if ready else 'pending',
                         'repeat_interpretation': 'deterministic actual refits; no stochastic confidence interval'
                         if deterministic else 'registered seed repetitions', 'strict_TAB_result': False})
    return rows, leaves, entities


def full_grasp_tables(root, queue_path):
    from scripts.flow_matching.summarize_grasp_protocol_v3 import summarize
    from scripts.flow_matching.export_complete_metrics import numeric_leaves
    queue = read(root / queue_path)
    capture = summarize(queue, root)
    capture.update(queue=queue_path, queue_sha256=sha(root / queue_path))
    summaries, leaves = [], []
    for group in capture['seed_aggregates']:
        recipes = next(j['score_recipes'] for j in queue['jobs']
                       if j['profile'] == group['profile'] and j['dataset'] == group['dataset'])
        for recipe in recipes:
            metrics = next((m['ten_seed_metrics'] for m in group['metrics'] or [] if m['recipe'] == recipe), {})
            for key in ('AP', 'ROC', 'Best_F1', 'precision', 'recall', 'f1'):
                summaries.append({'algorithm': 'GRASP', 'recipe': group['profile'], 'dataset': group['dataset'],
                                  'score_recipe': recipe, 'metric': key,
                                  'protocol': 'paper_test_label_oracle_no_PA' if key in ('AP', 'ROC', 'Best_F1')
                                  else 'validation_only_1_percent_no_PA_control',
                                  'completed_seeds': group['completed_seeds'], 'required_seeds': group['required_seeds'],
                                  **metrics.get(key, {'n': 0, 'mean': None, 'std': None, 'se': None,
                                                       'ci95_low': None, 'ci95_high': None}),
                                  'paper_equivalence_certified': False})
    for job in capture['jobs']:
        if job['status'] != 'completed':
            continue
        result = read(root / job['result_path'])
        if sha(root / job['result_path']) != job['result_sha256']:
            raise ValueError('Full GRASP result changed during capture')
        for metric, value in numeric_leaves(result):
            if metric == 'training_losses' or metric.startswith('training_history'):
                continue
            leaves.append(dict(job, metric_path=metric, value=value))
    return capture, summaries, leaves
