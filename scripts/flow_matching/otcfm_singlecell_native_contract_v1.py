"""Explicit axes for full released-source and paper-described single-cell runs.

The paper's batch128/patience30epochs/RK4 recipe is never silently replaced by
the generic released YAML batch512/patience100checks/Euler recipe. Historical
paper input identity is not certified for either available data artifact.
"""
import math

METHODS = ('cfm', 'otcfm', 'sbcfm')
DATASETS = ('released_author_donor32606', 'independent_public_donor13176')
PROTOCOLS = ('released_yaml', 'paper_declared')
SEEDS = (42, 43, 44, 45, 46)
LEAVEOUTS = (1, 2)


def protocol(name):
    base = dict(max_epochs=1000, validation_every_n_epochs=10, early_stopping_monitor='val/loss',
        early_stopping_min_delta=0., optimizer='AdamW', learning_rate=.001, weight_decay=1e-5,
        ode_grid_points_per_interval=101, data_split_fractions=[.8, .1, .1], data_split_seed=42,
        full_test_timepoint_arrays=True, whitening_before_timepoint_splits=True,
        validation_loss_includes_leftout_timepoint=True)
    if name == 'released_yaml':
        return dict(base, batch_size=512, early_stopping_patience_validation_checks=100,
            maximum_training_wall_seconds=None, ode_solver='euler',
            boundary='The full released generic CFM experiment/model/default callback settings, not the paper single-cell batch/stopping/solver recipe.')
    if name == 'paper_declared':
        return dict(base, batch_size=128, early_stopping_patience_validation_checks=3,
            maximum_training_wall_seconds=6000, ode_solver='rk4',
            boundary='Paper E.2 batch128, at most1000epochs, validation every10epochs, no improvement for30epochs represented by3checks,100-minute training cap; rk4 literal from ambiguous Runge-Kutta45(rk4) wording. LR/sigma/split details inherited from released source, not claimed historical settings.')
    raise ValueError('An explicit original-source or paper-described protocol is required')


def fraction_lengths(size, fractions):
    """PyTorch1.13+ fractional split allocation for the tested Torch1.12 runner.

    This fixes the release source's unsupported fractional random_split call in
    Torch1.12. The original generator42 and original integer random_split execute
    unchanged after allocation. No observations are dropped or reordered here.
    """
    if (type(size) is not int or size <= 0 or not fractions or not math.isclose(sum(fractions), 1.)
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 or v > 1 for v in fractions)):
        raise ValueError('Complete positive cohort and valid original split fractions required')
    lengths = [math.floor(size * value) for value in fractions]
    for i in range(size - sum(lengths)):
        lengths[i % len(lengths)] += 1
    return lengths


def compatible_random_split(dataset, lengths, generator, original_random_split):
    if (all(isinstance(x, (float, int)) for x in lengths) and math.isclose(sum(lengths), 1.)
            and all(0 <= x <= 1 for x in lengths)):
        lengths = fraction_lengths(len(dataset), lengths)
    return original_random_split(dataset, lengths, generator=generator)


def scientific_cases():
    cases = []
    for dataset in DATASETS:
        for recipe in PROTOCOLS:
            for method in METHODS:
                for leaveout in LEAVEOUTS:
                    for seed in SEEDS:
                        cases.append(dict(id=f'otcfm_singlecell__{recipe}__{method}__{dataset}__leave{leaveout}__seed{seed}',
                            dataset=dataset, protocol=recipe, method=method, leaveout_timepoint=leaveout,
                            seed=seed, scientific_protocol=protocol(recipe),
                            net=dict(hidden_dims=[64, 64, 64], batch_norm=False, activation='selu'),
                            sigma_min=1. if method == 'sbcfm' else .1,
                            ot_sampler={'cfm': None, 'otcfm': 'exact', 'sbcfm': 'sinkhorn'}[method],
                            expected_model_parameters=9093, native_dimensions=5,
                            author_equivalence_certified=False, all_original_experiments_complete=False))
    if len(cases) != 120 or len({case['id'] for case in cases}) != 120:
        raise ValueError('Keep all120 native single-cell conditions; other papers/data/baselines/ablations remain required')
    return cases


def validate_cases(cases):
    if cases != scientific_cases():
        raise ValueError('All datasets, algorithms, original five seeds, both held-out times and both scientific recipes required')


def terminal_reason(recipe, completed_epochs, stopped_epoch, elapsed_seconds, interrupted, finite_best_score):
    if (interrupted or not finite_best_score or completed_epochs <= 0
            or completed_epochs > recipe['max_epochs'] or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
        raise ValueError('Complete uninterrupted native fit with a finite selected checkpoint required')
    if completed_epochs == recipe['max_epochs']:
        return 'maximum_epochs'
    if stopped_epoch > 0:
        return 'native_early_stopping'
    if recipe['maximum_training_wall_seconds'] is not None and elapsed_seconds >= recipe['maximum_training_wall_seconds']:
        return 'native_paper_wall_time_cap'
    raise ValueError('A shortened fit reached none of the registered full stopping criteria')
