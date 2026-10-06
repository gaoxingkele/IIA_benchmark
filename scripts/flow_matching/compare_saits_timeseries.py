"""Keep author/corrected protocol comparisons separate and include every seed."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_saits_physio import sha


def compare(queue_path, reference_path, root=ROOT):
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    references = json.loads(reference_path.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    expected = defaultdict(set)
    pending = []
    for job in queue['jobs']:
        path = root / job['config']
        if sha(path) != job['config_sha256']:
            raise ValueError('Frozen time-series job changed')
        config = json.loads(path.read_text(encoding='utf-8'))
        key = (config['dataset'], config['method'], config['variant'], str(config['missing_ratio']))
        expected[key].add(config['seed'])
        result_path = root / config['output_root'] / 'result.json'
        if not result_path.exists():
            pending.append(config['id'])
            continue
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status') != 'completed' or result.get('config_sha256') != job['config_sha256'] or result.get('config', {}).get('diagnostic'):
            raise ValueError('Invalid or diagnostic result cannot enter formal comparison')
        groups[key].append(result)
    records = []
    for key, seeds in expected.items():
        dataset, method, variant, ratio = key
        values = groups[key]
        observed_seeds = [v['config']['seed'] for v in values]
        if len(set(observed_seeds)) != len(observed_seeds):
            raise ValueError('Duplicate formal seed')
        reference = references['datasets'][dataset][ratio][method]
        for metric, paper_value in reference.items():
            record = dict(dataset=dataset, method=method, variant=variant, missing_ratio=float(ratio), metric=metric,
                          paper_value=paper_value, completed_seeds=observed_seeds, required_seeds=sorted(seeds), verdict='incomplete')
            if set(observed_seeds) == seeds:
                numbers = [v['metrics'][metric] for v in values]
                if not all(math.isfinite(x) for x in numbers):
                    raise ValueError('Non-finite formal metric')
                mean, se = statistics.mean(numbers), statistics.stdev(numbers) / len(numbers) ** 0.5
                record.update(mean=mean, standard_error=se, model_seed_95_ci=[mean - 2.776445105 * se, mean + 2.776445105 * se],
                              absolute_difference=mean - paper_value, relative_difference=(mean - paper_value) / paper_value,
                              verdict='corrected_protocol_numeric_comparison_only' if variant == 'corrected' else 'author_protocol_alignment_unverified_numeric_comparison_only')
            records.append(record)
    return {'status': 'in_progress' if pending else 'completed_numeric_comparisons', 'pending_jobs': pending, 'records': records,
            'boundary': 'Original and corrected protocols remain separate. All preregistered seeds included; missing historical hyperparameters/ordering/seeds and unreported paper uncertainty prevent equivalence claims.'}


if __name__ == '__main__':
    report = compare(ROOT / 'configs/experiments/fm_saits_timeseries_queue.v1.json',
                     ROOT / 'configs/reproducibility/saits/timeseries_references.v1.json')
    (ROOT / 'docs/reports/flow_matching_saits_timeseries_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'pending_jobs': len(report['pending_jobs'])}))
