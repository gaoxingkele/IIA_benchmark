"""Descriptive training-seed statistics for a fully verified original cohort."""
import math
import statistics

from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

METRICS = ('mae', 'rmse', 'mre')
SEEDS = (26, 27, 28, 29, 30)


def summarize(rows):
    if len(rows) != 5 or sorted(row['seed'] for row in rows) != list(SEEDS):
        raise ValueError('All five unique original model seeds26..30 are required')
    identity = {(row['method'], row['dataset'], row['nominal_missing_ratio'], row['dataset_sha256']) for row in rows}
    if len(identity) != 1:
        raise ValueError('One frozen method/dataset/missing-rate/split cohort required')
    for row in rows:
        for key in METRICS:
            source = row['original_metrics'][key]
            reviewed = row['independent_metrics'][key]
            if not math.isfinite(source) or source < 0 or not math.isfinite(reviewed) or reviewed < 0:
                raise ValueError('All complete original and reviewed metrics must be finite')
            if not math.isclose(source, reviewed, rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError('Preserve the original full-replay metric tolerance')
    output = {}
    # An independent scalar reduction and fixed df4 quantile check the existing
    # NumPy/SciPy estimator. This interval describes five models, not ten months.
    critical_df4 = 2.7764451051977987
    for key in METRICS:
        values = [row['original_metrics'][key] for row in sorted(rows, key=lambda row: row['seed'])]
        primary = independent_stats(values)
        mean = statistics.mean(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / 4)
        se = std / math.sqrt(5)
        check = dict(mean=mean, std=std, se=se,
            ci95_low=mean - critical_df4 * se, ci95_high=mean + critical_df4 * se)
        if primary['n'] != 5 or any(not math.isclose(primary[name], value, rel_tol=1e-12, abs_tol=1e-12)
                for name, value in check.items()):
            raise ValueError('Independent five-model scalar mean/STD/t95 interval must agree')
        output[key] = dict(**primary, degrees_of_freedom=4, per_seed_original_values=values,
            independently_reviewed_metric_mean=statistics.mean(row['independent_metrics'][key] for row in rows),
            independent_scalar_reduction_and_df4_interval_verified=True)
    return dict(original_training_seeds=list(SEEDS), full_five_seed_cohort_complete=True,
        independently_trained_models=5, summaries=output,
        uncertainty_scope='sampleSTD and Student-t95 interval over five training models on one frozen split; descriptive small-n uncertainty',
        within_model_month_block_uncertainty_is_separate=True, author_equivalence_certified=False, strict_TAB_result=False)
