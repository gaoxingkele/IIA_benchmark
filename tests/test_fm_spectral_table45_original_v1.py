import copy
from pathlib import Path

import pytest

from scripts.flow_matching.run_spectral_table45_original_v1 import (
    ROOT, original_args, check_worker_seed_source, validate_full_receipt,
)
from scripts.flow_matching.register_spectral_table45_original_v1 import original_commands


SOURCE = ROOT/'experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_small'


def test_all_released_commands_have_full_original_budget_and_both_stability_ablations():
    commands = list(original_commands(SOURCE))
    assert len(commands) == 18
    physics = []
    stocks = []
    for script, entry, arguments, allocation in commands:
        settings = original_args(SOURCE/entry, arguments)
        assert settings['epochs'] == 600
        assert settings['batch_size'] == 64
        assert settings['seed'] == 10
        assert settings['lr'] == .0007
        assert allocation.startswith('CUDA_VISIBLE_DEVICES=')
        check_worker_seed_source(SOURCE/entry)
        if 'pendulum' in entry:
            physics.append(('kovae' if 'kovae' in entry else 'spectral_flow', settings['w_eig']))
            if 'kovae' in entry:
                assert settings['z_dim'] == 4
            else:
                assert (settings['hidden_dim'], settings['num_layers'], settings['spec_dim']) == (32, 8, 4)
        else:
            stocks.append((entry, settings['missing_value']))
            if 'kovae' in entry:
                assert (settings['w_kl'], settings['w_pred_prior']) == (.009, .03)
            else:
                assert (settings['hidden_dim'], settings['num_layers'], settings['sampling_steps']) == (64, 4, 100)
    assert sorted(physics) == [('kovae', 0.), ('kovae', 1.), ('spectral_flow', 0.), ('spectral_flow', 1.)]
    assert len(stocks) == 14


def valid_receipt():
    return dict(generator_loaders=[dict(samples=2000, batch_size=64, num_workers=4, drop_last=False)],
        epoch_updates={str(i):32 for i in range(600)}, optimizer_updates=19200,
        metric_repeats=[.01,.02,.03,.04,.05], generated_shape=[2000,40,2], reference_shape=[2000,40,2],
        arrays_sha256='a'*64, checkpoint_sha256='b'*64)


CFG = dict(table=5, num_workers=4, author_arguments=dict(epochs=600))


def test_complete_original_receipt_requires_every_epoch_and_full_arrays():
    validate_full_receipt(valid_receipt(), CFG)


@pytest.mark.parametrize('mutation', ['partial_epoch','few_epochs','missing_repeat','nonfinite','small_data',
                                     'workers_zero','batch_reduced','missing_checkpoint','missing_arrays'])
def test_partial_or_substitute_results_cannot_pass_full_validation(mutation):
    receipt = valid_receipt()
    if mutation == 'partial_epoch':
        receipt['epoch_updates']['599'] = 31; receipt['optimizer_updates'] -= 1
    elif mutation == 'few_epochs':
        receipt['epoch_updates'].pop('599'); receipt['optimizer_updates'] -= 32
    elif mutation == 'missing_repeat':
        receipt['metric_repeats'].pop()
    elif mutation == 'nonfinite':
        receipt['metric_repeats'][0] = float('nan')
    elif mutation == 'small_data':
        receipt['generated_shape'][0] = 64
    elif mutation == 'workers_zero':
        receipt['generator_loaders'][0]['num_workers'] = 0
    elif mutation == 'batch_reduced':
        receipt['generator_loaders'][0]['batch_size'] = 32
    elif mutation == 'missing_checkpoint':
        receipt.pop('checkpoint_sha256')
    else:
        receipt.pop('arrays_sha256')
    with pytest.raises(ValueError):
        validate_full_receipt(receipt, CFG)


def test_released_sampling_and_discriminator_budget_are_retained():
    for entry in ('main_regular.py','main.py','main_joint.py','main_pendulum.py'):
        source = (SOURCE/entry).read_text(encoding='utf-8')
        assert 'method="midpoint"' in source
        assert 'ema.ema_model.compute_flow' in source
        assert 'optimizer.step()' in source
    discriminator = (SOURCE/'metrics/discriminative_torch.py').read_text(encoding='utf-8')
    assert 'iterations = 2000' in discriminator
