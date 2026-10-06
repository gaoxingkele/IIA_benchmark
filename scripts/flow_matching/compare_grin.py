"""Compare all frozen GRIN seeds with visually verified out-of-sample references."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def compare(queue_path, reference_path, root=ROOT):
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    expected, actual = defaultdict(set), defaultdict(list)
    pending = []
    for job in queue['jobs']:
        path = root / job['config']
        if hashlib.sha256(path.read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError('Frozen GRIN job changed')
        cfg = json.loads(path.read_text(encoding='utf-8'))
        expected[cfg['dataset']].add(cfg['seed'])
        result_path = root / cfg['output_root'] / 'result.json'
        if not result_path.exists():
            pending.append(cfg['id'])
            continue
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status') != 'completed' or result.get('config_sha256') != job['config_sha256'] or result.get('diagnostic') or result.get('config', {}).get('diagnostic'):
            raise ValueError('Invalid or diagnostic GRIN result')
        if result['config']['seed'] != cfg['seed'] or result['config']['dataset'] != cfg['dataset']:
            raise ValueError('Result seed/dataset differs from job')
        actual[cfg['dataset']].append(result)
    records = []
    for dataset, seeds in expected.items():
        results = actual[dataset]
        observed = [r['config']['seed'] for r in results]
        if len(set(observed)) != len(observed):
            raise ValueError('Duplicate formal seed')
        for metric, paper in reference['datasets'][dataset].items():
            record = {'dataset': dataset, 'metric': metric, 'paper_reference': paper, 'required_seeds': sorted(seeds),
                      'completed_seeds': observed, 'verdict': 'incomplete'}
            if set(observed) == seeds:
                values = [r['metrics'][metric] for r in results]
                if len(values) != reference['runs'] or not all(math.isfinite(v) for v in values):
                    raise ValueError('Incomplete or non-finite repeats')
                mean = statistics.mean(values)
                se = statistics.stdev(values) / math.sqrt(len(values))
                difference = mean - paper['mean']
                record.update(mean=mean, standard_error=se, seed_95_ci=[mean - 2.776445105 * se, mean + 2.776445105 * se],
                              difference=difference, relative_difference=difference / paper['mean'],
                              within_paper_reported_spread_and_rounding=abs(difference) <= paper['reported_plus_minus'] + paper['rounding_half_unit'],
                              verdict='numeric_comparison_only_protocol_alignment_unverified')
            records.append(record)
    return {'status': 'in_progress' if pending else 'completed_numeric_comparisons', 'pending_jobs': pending,
            'records': records, 'boundary': reference['boundary']}


if __name__ == '__main__':
    result = compare(ROOT / 'configs/experiments/fm_grin_original_queue.v1.json', ROOT / 'configs/reproducibility/grin_references.v1.json')
    (ROOT / 'docs/reports/flow_matching_grin_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'pending_jobs': len(result['pending_jobs'])}))
