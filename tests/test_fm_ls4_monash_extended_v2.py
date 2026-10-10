import copy

import pytest

from scripts.flow_matching.run_ls4_monash_extended_v2 import budget, derive_config, validate_result


def config(dataset='temperature_rain', protocol='released'):
    solar=dataset=='solar_weekly';released=protocol=='released'
    return dict(dataset=dataset,protocol=protocol,train_samples=109 if solar else 25657,
        test_samples=28 if solar else 6415,length=52 if solar else 725,
        optim=dict(epochs=(100000 if released else 7000) if solar else 1000,
            batch_size=128 if released else 64,metric_iter=500 if solar else 5,
            eval_iter=100 if solar else 5))


def result(cfg):
    b=budget(cfg);epochs=cfg['optim']['epochs'];shape=[cfg['test_samples'],cfg['length'],1]
    return dict(generator_updates=b['generator_updates'],
        epoch_updates={str(e):b['batches'] for e in range(epochs)},
        epoch_samples={str(e):cfg['train_samples'] for e in range(epochs)},
        metric_history=[dict(generator_updates=step,classifier_updates=b['classifier_updates_per_metric'],
            predictor_updates=b['predictor_updates_per_metric'],reference_shape=shape,generated_shape=shape,
            metrics=dict(clf_score=14.,marginal_score=.1,predictive_score=5.)) for step in b['metric_update_positions']],
        reconstruction_history=[{} for _ in range(sum(e>0 and e%cfg['optim']['eval_iter']==0 for e in range(epochs))+1)])


def test_all32072_temperature_trajectories_and_full_multibatch_evaluators():
    b=budget(config())
    assert b['generator_updates']==201000 and b['metric_calls']==200
    assert b['classifier_updates_per_metric']==8100 and b['predictor_updates_per_metric']==5100
    assert b['classifier_test_batches']==21 and b['predictor_test_batches']==51
    validate_result(result(config()),config())


def test_full_released_solar100000_epochs_is_not_shortened_to_paper7000():
    released=budget(config('solar_weekly'));paper=budget(config('solar_weekly','paper_literal'))
    assert released['generator_updates']==100000 and released['metric_calls']==200
    assert paper['generator_updates']==14000 and paper['metric_calls']==14
    assert budget(config('temperature_rain','paper_literal'))['generator_updates']==401000


def test_paper_latent_alias_dimensions_and_ema_change_consistently():
    template=dict(data=dict(path='raw',preproc='squash_per_seq'),
        optim=dict(epochs=1000,batch_size=128,lamb=.999,lr=.001),
        model=dict(z_dim=10,sigma=.1,decoder=dict(activation='sigmoid',
            prior=dict(d_input=10,d_output=10,d_state=64,n_layers=4),
            decoder=dict(d_input=10,d_output=1,d_model=64)),
            encoder=dict(posterior=dict(d_output=10,d_state=64))))
    before=copy.deepcopy(template)
    actual=derive_config(template,dict(dataset='temperature_rain',protocol='paper_literal'),'private')
    assert actual['model']['z_dim']==actual['model']['decoder']['prior']['d_input']==5
    assert actual['model']['decoder']['prior']['d_output']==actual['model']['decoder']['decoder']['d_input']==5
    assert actual['model']['encoder']['posterior']['d_output']==5
    assert actual['model']['decoder']['activation']=='sigmoid'
    assert actual['optim']==dict(epochs=1000,batch_size=64,lamb=.999,lr=.001)
    assert template==before


@pytest.mark.parametrize('kind',['only100_updates','short_generator','lost_trajectory','missing_final','short_length'])
def test_reduced_or_single_batch_temperature_execution_cannot_satisfy_full_result(kind):
    cfg=config();r=result(cfg)
    if kind=='only100_updates':r['metric_history'][0]['classifier_updates']=100
    elif kind=='short_generator':r['generator_updates']-=1
    elif kind=='lost_trajectory':r['epoch_samples']['0']-=1
    elif kind=='missing_final':r['metric_history'].pop()
    else:r['metric_history'][0]['generated_shape']=[6415,724,1]
    with pytest.raises(ValueError):validate_result(r,cfg)
