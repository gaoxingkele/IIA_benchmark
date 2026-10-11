"""Summarize every fully verified native WT step/seed and independently review statistics."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import statistics
import sys

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new
from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import complete
from scripts.flow_matching.wt_full_cohort_summary_contract_v1 import (
    validate_configuration, METRICS, SEEDS, STEPS, UNCERTAINTY)


def checked_records(records, category_names, category_counts, paper):
    """Reject missing seeds/settings, changed pairing and category-weighted macro means."""
    if (len(records) != 30 or len(category_names) != 15 or len(set(category_names)) != 15
            or sum(category_counts) != 1725 or any(n < 1 for n in category_counts)
            or len(category_counts) != 15
            or {(r['seed'], r['steps']) for r in records} != {(s, t) for s in SEEDS for t in STEPS}):
        raise ValueError('Every original WT seed/step/category/test image required')
    by_seed = {}
    for record in records:
        identity = (record['checkpoint_sha256'], record['training_binding'])
        if record['seed'] in by_seed and by_seed[record['seed']] != identity:
            raise ValueError('Paired native-step comparisons must use identical frozen training checkpoints')
        by_seed[record['seed']] = identity
        if (record['paper_reported'] != paper or set(record['macro']) != set(METRICS)
                or len(record['categories']) != 15
                or [row['category'] for row in record['categories']] != category_names
                or [row['original_test_images'] for row in record['categories']] != category_counts):
            raise ValueError('Original complete category names/counts/paper metrics must agree')
        for row in record['categories']:
            if set(row['metrics']) != set(METRICS):
                raise ValueError('All seven metrics in every original category required')
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in row['metrics'].values()):
                raise ValueError('Finite full-native unit-scale metrics required')
        for metric in METRICS:
            mean = statistics.fmean(row['metrics'][metric] for row in record['categories'])
            if not math.isclose(mean, record['macro'][metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Unweighted complete15category macro mean required')
    if len({identity[0] for identity in by_seed.values()}) != 5:
        raise ValueError('Five independent original training checkpoints, not duplicated inference repeats')
    bindings = [identity[1] for identity in by_seed.values()]
    if any(set(binding) != {'job_sha256', 'dataset_configuration_sha256', 'model_configuration_sha256', 'optimizer_configuration_sha256'} for binding in bindings):
        raise ValueError('Complete original training/data/backbone/optimizer bindings required')
    for field in ('dataset_configuration_sha256', 'model_configuration_sha256', 'optimizer_configuration_sha256'):
        if len({binding[field] for binding in bindings}) != 1:
            raise ValueError('Five seeds must share original data/backbone/optimizer identity')
    if len({binding['job_sha256'] for binding in bindings}) != 5:
        raise ValueError('Five unique original training job bindings required')
    return {(r['seed'], r['steps']): r for r in records}


def load_verified_records(config, queue):
    import numpy as np
    environment = read(ROOT / queue['environment_configuration'])
    if (sha(sys.executable) != environment['python_sha256'] or sys.version != environment['python_version']
            or any(importlib.metadata.version(p) != v for p, v in environment['packages'].items())):
        raise ValueError('Frozen full native scoring/summary environment differs')
    job = read(ROOT / queue['cases'][0]['training_configuration'])
    dataset = read(ROOT / job['dataset_configuration'])
    verify_receipts(dataset['source_receipts']); verify_receipts(dataset['array_receipts'])
    derived = next(x for x in dataset['source_receipts'] if Path(x['path']).name == 'full_derived_image_preprocessing.json')
    names = read(ROOT / derived['path'])['original_categories']
    categories_path = next(x['path'] for x in dataset['array_receipts'] if x['split'] == 'test' and x['field'] == 'categories')
    categories = np.load(categories_path, allow_pickle=False)
    if categories.shape != (1725,) or set(np.unique(categories)) != set(range(15)):
        raise ValueError('Original full MVTec category identity array required')
    counts = [int((categories == i).sum()) for i in range(15)]
    records = []
    for case in queue['cases']:
        if not complete(case):
            raise ValueError('All30 actual full checkpoint and every-image score/replay proofs required before summary')
        location_path = ROOT / case['output_directory'] / 'fully_verified_score_result.json'
        location = read(location_path); report = read(location['report'])
        records.append(dict(case_id=case['id'], seed=case['seed'], steps=case['score_parameters']['integration_steps'],
            checkpoint_sha256=report['full_original_checkpoint_sha256'], training_binding=report['training_binding'],
            paper_reported=case['paper_reported'], macro=report['category_macro_mean'],
            categories=report['independently_recomputed_per_category_metrics'],
            score_completion_receipt=dict(path=str(location_path), sha256=sha(location_path)),
            full_score_report=dict(path=location['report'], sha256=sha(location['report']))))
    paper = queue['cases'][0]['paper_reported']
    checked_records(records, names, counts, paper)
    return records, names, counts, paper


def build_summary(records, names, counts, paper):
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
    grid = checked_records(records, names, counts, paper)
    aggregate, categories, paired = [], [], []
    for steps in STEPS:
        for metric in METRICS:
            values = [grid[seed, steps]['macro'][metric] for seed in SEEDS]
            stats = independent_stats(values)
            aggregate.append(dict(steps=steps, metric=metric, **stats, independent_training_seeds=SEEDS,
                seed_values=values, mean_percent=100*stats['mean'], paper_percent=paper[metric],
                difference_percentage_points=100*stats['mean']-paper[metric],
                author_equivalence_certified=False, strict_TAB_result=False))
            for index, name in enumerate(names):
                values = [grid[seed, steps]['categories'][index]['metrics'][metric] for seed in SEEDS]
                categories.append(dict(steps=steps, category=name, original_test_images=counts[index], metric=metric,
                    **independent_stats(values), independent_training_seeds=SEEDS, seed_values=values))
            for reference in UNCERTAINTY['paired_reference_steps']:
                deltas = [grid[seed, steps]['macro'][metric] - grid[seed, reference]['macro'][metric] for seed in SEEDS]
                paired.append(dict(steps=steps, reference_steps=reference, metric=metric,
                    **independent_stats(deltas), independent_training_seeds=SEEDS, paired_seed_differences=deltas,
                    identical_checkpoint_per_seed=True, uncertainty_is_significance_test=False))
    robustness = []
    for metric in METRICS:
        rows = [row for row in aggregate if row['metric'] == metric]
        robustness.append(dict(metric=metric, frozen_grid=STEPS,
            mean_min_over_fixed_steps=min(row['mean'] for row in rows),
            mean_max_over_fixed_steps=max(row['mean'] for row in rows),
            mean_range_over_fixed_steps=max(row['mean'] for row in rows)-min(row['mean'] for row in rows),
            step_selected_for_final_performance=False))
    return dict(aggregate_metrics=aggregate, per_category_metrics=categories,
        paired_step_differences=paired, fixed_grid_descriptive_robustness=robustness)


def independent_statistics(values):
    """Independent implementation: stdlib mean/stdev, SciPy t distribution."""
    from scipy.stats import t
    if len(values) != 5 or any(not math.isfinite(v) for v in values):
        raise ValueError('Five finite independent training-seed observations required')
    mean = statistics.mean(values); std = statistics.stdev(values); se = std / math.sqrt(5)
    half = float(t.ppf(.975, 4)) * se
    return dict(n=5, mean=mean, std=std, se=se, ci95_low=mean-half, ci95_high=mean+half)


def review_summary(candidate, records, names, counts, paper):
    grid = checked_records(records, names, counts, paper)
    expected_keys = {
        'aggregate_metrics': {(steps, metric) for steps in STEPS for metric in METRICS},
        'per_category_metrics': {(steps, name, metric) for steps in STEPS for name in names for metric in METRICS},
        'paired_step_differences': {(steps, ref, metric) for steps in STEPS for ref in [1, 50] for metric in METRICS}}
    for field, expected in expected_keys.items():
        rows = candidate[field]
        def key(row):
            if field == 'aggregate_metrics': return row['steps'], row['metric']
            if field == 'per_category_metrics': return row['steps'], row['category'], row['metric']
            return row['steps'], row['reference_steps'], row['metric']
        if len(rows) != len(expected) or {key(row) for row in rows} != expected:
            raise ValueError('Every fixed step/metric/category/pair required without subset or duplicates')
        for row in rows:
            metric, steps = row['metric'], row['steps']
            if row['independent_training_seeds'] != SEEDS:
                raise ValueError('Training seeds, not30inference cases, are uncertainty units')
            if field == 'aggregate_metrics':
                values = [grid[seed, steps]['macro'][metric] for seed in SEEDS]
                if (row['paper_percent'] != paper[metric] or row['author_equivalence_certified'] or row['strict_TAB_result']
                        or not math.isclose(row['mean_percent'], statistics.mean(values)*100, abs_tol=1e-12)
                        or not math.isclose(row['difference_percentage_points'], statistics.mean(values)*100-paper[metric], abs_tol=1e-12)):
                    raise ValueError('Paper values, units and uncertified comparison boundary required')
            elif field == 'per_category_metrics':
                index = names.index(row['category'])
                if row['original_test_images'] != counts[index]:
                    raise ValueError('Every original category image count must remain unchanged')
                values = [grid[seed, steps]['categories'][index]['metrics'][metric] for seed in SEEDS]
            else:
                values = [grid[seed, steps]['macro'][metric]-grid[seed, row['reference_steps']]['macro'][metric] for seed in SEEDS]
                if not row['identical_checkpoint_per_seed'] or row['uncertainty_is_significance_test']:
                    raise ValueError('Descriptive paired steps from identical original checkpoints required')
            if row.get('seed_values', row.get('paired_seed_differences')) != values:
                raise ValueError('All five ordered source observations must match')
            for key, value in independent_statistics(values).items():
                if not math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Independent five-seed uncertainty differs: '+key)
    rows = candidate['fixed_grid_descriptive_robustness']
    if len(rows) != 7 or {row['metric'] for row in rows} != set(METRICS):
        raise ValueError('Seven complete fixed-grid robustness records required')
    for row in rows:
        values = [statistics.mean(grid[seed, step]['macro'][row['metric']] for seed in SEEDS) for step in STEPS]
        if row['frozen_grid'] != STEPS or row['step_selected_for_final_performance']:
            raise ValueError('All fixed steps remain; no test-selected winning solver')
        for key, value in [('mean_min_over_fixed_steps', min(values)), ('mean_max_over_fixed_steps', max(values)),
                           ('mean_range_over_fixed_steps', max(values)-min(values))]:
            if not math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Independent fixed-grid robustness differs')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--attempt-directory', required=True)
    parser.add_argument('--audit', action='store_true'); args = parser.parse_args()
    config = read(ROOT / args.config); queue = validate_configuration(config)
    if config['configuration_path'] != args.config:
        raise ValueError('Frozen cohort configuration path identity differs')
    output = (ROOT / config['output_directory']).resolve(); attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_relative_to(output / 'attempts') or not attempt.is_dir():
        raise ValueError('Fresh preserved cohort summary/review attempt required')
    records, names, counts, paper = load_verified_records(config, queue)
    if not args.audit:
        value = build_summary(records, names, counts, paper)
        value.update(captured_utc=datetime.now(timezone.utc).isoformat(), config=args.config, config_sha256=sha(ROOT / args.config),
            score_queue=config['score_queue'], score_queue_sha256=sha(ROOT / config['score_queue']),
            full_source_case_records=records, uncertainty=UNCERTAINTY, original_scope=config['original_scope'],
            full_verified_case_seed_evaluations=30, independent_original_training_seeds=SEEDS, native_step_grid=STEPS,
            original_category_names=names, original_category_counts=counts,
            independent_cohort_statistics_review_passed=False, additional_training_seed_slots=0,
            author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
        path = attempt / 'full_native_wt_cohort_candidate.json'; write_new(path, value)
        write_new(output / 'cohort_candidate_location.json', dict(path=str(path), sha256=sha(path)))
    else:
        location = read(output / 'cohort_candidate_location.json'); verify_receipts([location])
        candidate = read(location['path'])
        if (candidate['config_sha256'] != sha(ROOT / args.config) or candidate['full_source_case_records'] != records
                or candidate['score_queue_sha256'] != sha(ROOT / config['score_queue'])
                or candidate['uncertainty'] != UNCERTAINTY or candidate['original_category_names'] != names
                or candidate['original_category_counts'] != counts or candidate['additional_training_seed_slots'] != 0
                or candidate['author_equivalence_certified'] or candidate['strict_TAB_result']
                or candidate['all_original_experiments_complete']):
            raise ValueError('Actual full cohort candidate/source/config/uncertainty binding changed')
        review_summary(candidate, records, names, counts, paper)
        report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
            candidate_receipt=location, score_queue_sha256=sha(ROOT / config['score_queue']),
            full_verified_case_seed_evaluations=30, independent_original_training_seeds=SEEDS, native_step_grid=STEPS,
            aggregate_metric_rows=42, per_category_metric_rows=630, paired_metric_rows=84,
            all_source_checkpoint_data_and_full_score_proofs_verified=True,
            all_seed_uncertainty_and_pairing_independently_verified=True,
            additional_training_seed_slots=0, original_scope=config['original_scope'],
            author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
        path = attempt / 'independent_full_native_wt_cohort_review.json'; write_new(path, report)
        write_new(output / 'fully_verified_cohort_summary.json', dict(passed=True, candidate=location,
            independent_review=dict(path=str(path), sha256=sha(path))))
    print(dict(full_verified_cases=30, independent_training_seeds=5, fixed_solver_settings=6, audit=args.audit))


if __name__ == '__main__':
    main()
