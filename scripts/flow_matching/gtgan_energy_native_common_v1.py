"""Exact README Energy recipes and budgets; no change to the author algorithm."""
from pathlib import Path

from scripts.flow_matching.register_gtgan_native_commands_v1 import native_arguments


def recipes():
    common = ['--data', 'energy', '--train', '--atol', '1e-3', '--rtol', '1e-3']
    regular = common + ['--max-steps', '6000', '--log_time', '2', '--missing_value', '0.0',
        '--reconstruction', '0.01', '--kinetic-energy', '0.5', '--jacobian-norm2', '0.1',
        '--first_epoch', '5000', '--save_dir', 'regular_energy']
    irregular = common + ['--log_time', '2', '--max-steps', '8500', '--missing_value', '0.5',
        '--reconstruction', '0.01', '--kinetic-energy', '0.5', '--jacobian-norm2', '0.1',
        '--first_epoch', '5000', '--save_dir', 'irregular_energy']
    return [(0.0, regular), (0.5, irregular)]


def release_recipes():
    common = ['--data', 'energy', '--atol', '1e-3', '--rtol', '1e-3', '--log_time', '2']
    suffix = ['--reconstruction', '0.01', '--kinetic-energy', '0.5', '--jacobian-norm2', '0.1']
    return [(rate, common + ['--max-steps', str(6000 if rate == 0 else 8500),
        '--missing_value', str(rate)] + suffix + ['--save_dir', 'regular_energy' if rate == 0 else 'irregular_energy'])
        for rate in [0.0, 0.5]]


def verify_recipe(case, source):
    release = case.get('execution_mode') == 'native_release_checkpoint_evaluation'
    matching = next((args for rate, args in (release_recipes() if release else recipes()) if rate == case['missing_rate']), None)
    if matching is None or case['author_arguments'] != matching:
        raise ValueError('The exact original README Energy command is required')
    entry = Path(case['source_entry']).name
    if entry != 'GTGAN_energy.py':
        raise ValueError('Energy must use the original Energy entry point')
    if native_arguments(source, entry, matching) != case['parsed_original_arguments']:
        raise ValueError('Original source defaults differ from the frozen Energy command')
    return full_budget(case)


def full_budget(case):
    a = case['parsed_original_arguments']
    release = case.get('execution_mode') == 'native_release_checkpoint_evaluation'
    joint = {0.0: 6000, 0.5: 8500}.get(case['missing_rate'])
    required = dict(data='energy', train=not release, first_epoch=10000 if release else 5000, max_steps=joint,
        batch_size=128, seq_len=24, hidden_dim=24, max_steps_metric=10,
        solver='sym12async', dims='32-64-64-32', log_time=2, r_layer=2, d_layer=2,
        last_activation_r='sigmoid', last_activation_d='identity', activation='exp',
        atol=0.001, rtol=0.001, reconstruction=0.01, kinetic_energy=0.5,
        jacobian_norm2=0.1, missing_value=case['missing_rate'])
    if joint is None or case['original_seed'] != 7777 or any(a.get(k) != v for k, v in required.items()):
        raise ValueError('Complete original Energy command budget and architecture are required')
    return dict(native_embedding_steps=0 if release else 5000, native_joint_steps=0 if release else joint,
        embedding_optimizer_updates=0 if release else 5000 + 2 * joint,
        generator_optimizer_updates=0 if release else joint + joint // 2,
        maximum_discriminator_updates=0 if release else 2 * joint,
        full_periodic_evaluations=1 if release else joint // 500, evaluator_repeats_per_metric=10,
        predictive_evaluator_steps=5000, discriminative_evaluator_steps=300,
        full_generation_windows=19712, variables=28, sequence_length=24,
        predictive_function='predictive_score_metrics2' if case['missing_rate'] == 0 else 'predictive_score_metrics',
        native_pretrained_training_reload=False, native_release_checkpoint_evaluation=release,
        expected_native_optimizers=0 if release else 3)
