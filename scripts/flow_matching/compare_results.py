"""Compare frozen, protocol-aligned repeated runs; never pick the best test run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]


def compare(records, reference):
    if not records:
        return {'verdict': 'not_run'}
    for record in records:
        cfg = record['config']
        if cfg.get('diagnostic') or cfg.get('limit_test_cases') or record['status'] != 'completed':
            return {'verdict': 'not_comparable', 'reason': 'Diagnostic, incomplete or truncated run'}
        if cfg.get('reference_id') != reference['id'] or cfg['num_samples'] != reference['num_samples']:
            return {'verdict': 'not_comparable', 'reason': 'Reference protocol or sample count differs'}
        if reference.get('requires_protocol_alignment_review') and not cfg.get('protocol_alignment_verified'):
            return {'verdict': 'not_comparable', 'reason': 'Historical validation indices are unpublished; broad author-protocol run retained without exact reproduction claim'}
    values = [record['metrics'][reference['metric']] for record in records]
    if not np.isfinite(values).all():
        raise ValueError('Invalid metric values')
    result = {'reference': reference, 'mean': float(np.mean(values)), 'runs': len(values),
              'absolute_difference': abs(float(np.mean(values)) - reference['mean'])}
    if len(values) != reference['runs']:
        return {**result, 'verdict': 'incomplete_repeats', 'reason': 'All preregistered folds/repeats are required'}
    folds = [record['config'].get('fold') for record in records]
    if reference.get('expected_folds') is not None and sorted(folds) != reference['expected_folds']:
        return {**result, 'verdict': 'not_comparable', 'reason': 'Duplicate or missing test folds'}
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    combined_se = math.sqrt(se ** 2 + reference['standard_error'] ** 2)
    band = max(reference['rounding_half_unit'], float(t.ppf(.975, len(values) - 1)) * combined_se)
    return {**result, 'standard_error': se, 'comparison_band': band,
            'verdict': 'numerically_compatible' if result['absolute_difference'] <= band else 'differs_from_paper',
            'boundary': 'Compatibility screen, not proof of statistical equivalence. Code/environment deviations must still be reviewed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=ROOT / 'configs/reproducibility/flow_matching_references.v1.json')
    parser.add_argument('--runs', type=Path, default=ROOT / 'experiments/runs/flow_matching_campaign')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/reports/flow_matching_reproduction_status_2026-10-06.json')
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding='utf-8')) for path in args.runs.glob('*/result.json')]
    results = []
    for reference in json.loads(args.references.read_text(encoding='utf-8'))['references']:
        relevant = [record for record in records if record['config'].get('reference_id') == reference['id']]
        results.append({'id': reference['id'], **compare(relevant, reference)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'results': results}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': len(results), 'completed_rows': sum(r['verdict'] in ('numerically_compatible', 'differs_from_paper') for r in results)}))


if __name__ == '__main__':
    main()
