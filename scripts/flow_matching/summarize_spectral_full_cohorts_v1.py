"""Keep training-seed uncertainty separate from repeated learned evaluators."""
import math
import statistics

from scipy.stats import t

from scripts.flow_matching.audit_spectral_completed_generation_v1 import METRICS


def summarize_cohort(proofs, required_seeds=(42, 43, 44, 45, 46)):
    seeds = [proof['seed'] for proof in proofs]
    if not proofs or len(seeds) != len(set(seeds)) or not set(seeds).issubset(required_seeds):
        raise ValueError('Unique original trained-model seeds required')
    if len({(p['model'], p['dataset'], p['data_track']) for p in proofs}) != 1:
        raise ValueError('Do not pool models, datasets or data tracks')
    for proof in proofs:
        if not proof['passed'] or proof['actual_full_native_model_count'] != 1 or proof['cuda_context_initialized']:
            raise ValueError('Complete independently reviewed original models required')
        if set(proof['metrics']) != set(METRICS):
            raise ValueError('All four complete evaluator families required')
        for values in proof['metrics'].values():
            raw = values['raw_repeats']
            if len(raw) != 5 or not all(math.isfinite(value) for value in raw):
                raise ValueError('All five finite evaluator repetitions required')
            if not math.isclose(statistics.mean(raw), values['mean'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Frozen full evaluator mean changed')
    complete = set(seeds) == set(required_seeds)
    rows = {}
    for metric in METRICS:
        per_model = [dict(seed=p['seed'], evaluator_mean=p['metrics'][metric]['mean'],
            five_evaluator_repetitions=p['metrics'][metric]['raw_repeats']) for p in sorted(proofs, key=lambda p: p['seed'])]
        summary = dict(full_cohort_mean=None, between_training_model_sample_STD=None,
            between_training_model_CI95=None, degrees_of_freedom=None)
        if complete:
            values = [row['evaluator_mean'] for row in per_model]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            df = len(values) - 1
            margin = float(t.ppf(.975, df)) * std / math.sqrt(len(values))
            summary.update(full_cohort_mean=mean, between_training_model_sample_STD=std,
                between_training_model_CI95=[mean - margin, mean + margin], degrees_of_freedom=df)
        rows[metric] = dict(independent_training_models=len(proofs),
            evaluator_runs=5 * len(proofs), per_training_model=per_model, **summary)
    return dict(model=proofs[0]['model'], dataset=proofs[0]['dataset'], data_track=proofs[0]['data_track'],
        independently_reviewed_training_seeds=sorted(seeds), required_original_training_seeds=list(required_seeds),
        missing_original_training_seeds=sorted(set(required_seeds) - set(seeds)), full_five_seed_cohort_complete=complete,
        metrics=rows, uncertainty_scope='t interval over five model-specific evaluator means; descriptive small-n uncertainty',
        learned_evaluators_independently_retrained=False, author_equivalence_certified=False, strict_TAB_result=False)
