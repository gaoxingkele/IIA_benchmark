"""Compare all five frozen seeds with SAITS paper points without selecting runs."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def compare(queue_path, references_path, root=ROOT):
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    references = json.loads(references_path.read_text(encoding='utf-8'))
    results = {'saits': [], 'brits': []}
    pending = []
    for job in queue['jobs']:
        path = root / job['config']
        if hashlib.sha256(path.read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError('Frozen job changed')
        config = json.loads(path.read_text(encoding='utf-8'))
        result_path = root / config['output_root'] / 'result.json'
        if not result_path.exists():
            pending.append(config['id'])
            continue
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status') != 'completed' or result.get('config_sha256') != job['config_sha256'] or result.get('config', {}).get('diagnostic'):
            raise ValueError('Diagnostic or invalid result cannot enter original comparison')
        results[config['method']].append(result)
    records = []
    for method, values in results.items():
        seeds = [v['config']['seed'] for v in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate model seed')
        for metric, reference in references['datasets']['physio'][method].items():
            record = {'method': method, 'dataset': 'physio', 'metric': metric, 'paper_value': reference,
                      'completed_seeds': seeds, 'required_seeds': [26, 27, 28, 29, 30], 'verdict': 'incomplete'}
            if set(seeds) == {26, 27, 28, 29, 30}:
                observed = [v['metrics'][metric] for v in values]
                mean, se = statistics.mean(observed), statistics.stdev(observed) / len(observed) ** 0.5
                record.update(mean=mean, standard_error=se, model_seed_95_ci=[mean - 2.776445105 * se, mean + 2.776445105 * se],
                              absolute_difference=mean - reference, relative_difference=(mean - reference) / reference,
                              verdict='numeric_comparison_only_protocol_alignment_unverified')
            records.append(record)
    return {'status': 'in_progress' if pending else 'completed_numeric_comparison', 'pending_jobs': pending,
            'references': references, 'records': records,
            'boundary': 'All preregistered seeds included. Patient split order and worker RNG differ; paper Table 2 supplies no uncertainty. Numeric similarity cannot establish statistical equivalence.'}


if __name__ == '__main__':
    report = compare(ROOT / 'configs/experiments/fm_saits_physio_original_queue.v1.json',
                     ROOT / 'configs/reproducibility/saits/references.v1.json')
    path = ROOT / 'docs/reports/flow_matching_saits_reproduction_status.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'pending_jobs': len(report['pending_jobs'])}))
