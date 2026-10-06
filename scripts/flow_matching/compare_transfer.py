"""Report all preregistered paired seeds, including reversals and uncertainty."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]


def paired_summary(baseline, candidate, expected_seeds):
    if set(baseline) != set(expected_seeds) or set(candidate) != set(expected_seeds):
        return {'verdict': 'incomplete_pairs', 'completed_baseline': len(baseline), 'completed_candidate': len(candidate)}
    a = np.array([baseline[s] for s in expected_seeds], dtype=float)
    b = np.array([candidate[s] for s in expected_seeds], dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite paired metrics')
    delta = b - a
    se = float(delta.std(ddof=1) / len(delta) ** .5)
    margin = float(t.ppf(.975, len(delta) - 1) * se)
    low, high = float(delta.mean() - margin), float(delta.mean() + margin)
    return {'verdict': 'candidate_lower_error' if high < 0 else 'candidate_higher_error' if low > 0 else 'insufficient_evidence',
            'baseline_mean': float(a.mean()), 'candidate_mean': float(b.mean()),
            'baseline_standard_error': float(a.std(ddof=1) / len(a) ** .5),
            'candidate_standard_error': float(b.std(ddof=1) / len(b) ** .5),
            'paired_difference': float(delta.mean()), 'paired_standard_error': se,
            'paired_confidence_interval_95': [low, high], 'seeds': expected_seeds,
            'boundary': 'Exploratory paired seed interval on these fixed groups; no multiplicity correction or cross-plant claim.'}


def main():
    queue = json.loads((ROOT / 'configs/experiments/fm_transfer_queue.v1.json').read_text(encoding='utf-8'))
    groups = {}
    for job in queue['jobs']:
        cfg_path = ROOT / job['config']
        if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError('Frozen transfer configuration changed')
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        group = (cfg['dataset'], cfg['missing_pattern'], cfg['missing_ratio'])
        entry = groups.setdefault(group, {'expected_seeds': set(), 'csdi': {}, 'cfmi': {}})
        entry['expected_seeds'].add(cfg['seed'])
        path = ROOT / cfg['output_root'] / 'result.json'
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['status'] != 'completed' or record['config_sha256'] != job['config_sha256'] or record.get('metric_protocol') != 'transfer_ensemble_v1':
                raise ValueError('Transfer result has incompatible protocol')
            if record.get('config', {}).get('diagnostic') or record.get('config', {}).get('limit_test_cases'):
                raise ValueError('Diagnostic output cannot enter transfer comparison')
            method = cfg['id'].split('_')[0]
            if cfg['seed'] in entry[method]:
                raise ValueError('Duplicate seed result')
            entry[method][cfg['seed']] = record
    rows = []
    for (dataset, pattern, ratio), entry in groups.items():
        for seed in set(entry['csdi']) & set(entry['cfmi']):
            a, b = entry['csdi'][seed], entry['cfmi'][seed]
            if a['config']['processed_sha256'] != b['config']['processed_sha256'] or a['test_cases'] != b['test_cases'] or a['eval_count'] != b['eval_count']:
                raise ValueError('Paired methods used different data, masks or evaluation targets')
        for metric in ['mae', 'rmse', 'crps']:
            maps = [{seed: record['metrics'][metric] for seed, record in entry[method].items()} for method in ['csdi', 'cfmi']]
            rows.append({'dataset': dataset, 'missing_pattern': pattern, 'missing_ratio': ratio, 'metric': metric,
                         'baseline': 'csdi', 'candidate': 'cfmi', **paired_summary(*maps, sorted(entry['expected_seeds']))})
    path = ROOT / 'docs/reports/flow_matching_transfer_status_2026-10-06.json'
    path.write_text(json.dumps({'rows': rows}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'complete_rows': sum(row['verdict'] != 'incomplete_pairs' for row in rows)}))


if __name__ == '__main__':
    main()
