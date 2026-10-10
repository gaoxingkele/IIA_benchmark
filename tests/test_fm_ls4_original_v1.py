import copy

import pytest

from scripts.flow_matching.run_ls4_original_v1 import budget, derive_config, validate_result


def config(dataset='fred_md', protocol='released'):
    nn5=dataset=='nn5_daily'
    return dict(dataset=dataset,protocol=protocol,train_samples=88 if nn5 else 85,
        test_samples=23 if nn5 else 22,length=791 if nn5 else 728,
        optim=dict(epochs=10000 if protocol=='released' else 7000,
            batch_size=128 if nn5 and protocol=='released' else 64,
            metric_iter=500,eval_iter=500 if nn5 else 100))


def full_result(cfg):
    required=budget(cfg);epochs=cfg['optim']['epochs']
    shape=[cfg['test_samples'],cfg['length'],1]
    return dict(generator_updates=required['generator_updates'],
        epoch_updates={str(e):required['batches'] for e in range(epochs)},
        epoch_samples={str(e):cfg['train_samples'] for e in range(epochs)},
        metric_history=[dict(generator_updates=step,classifier_updates=100,predictor_updates=100,
            reference_shape=shape,generated_shape=shape,
            metrics=dict(clf_score=.1,marginal_score=.2,predictive_score=.3))
            for step in required['metric_update_positions']],
        reconstruction_history=[{} for _ in range(sum(e>0 and e%cfg['optim']['eval_iter']==0 for e in range(epochs))+1)])


def test_full_released_nn5_batches_are88_not_padded128():
    required=budget(config('nn5_daily'))
    assert required['batches']==1 and required['generator_updates']==10000
    assert required['metric_calls']==20
    assert required['metric_update_positions'][0]==501
    assert required['metric_update_positions'][-1]==10000


def test_paper_budget_keeps7000_epochs_full64_batches_and14_evaluations():
    cfg=config('nn5_daily','paper_literal');required=budget(cfg)
    assert required['generator_updates']==14000 and required['metric_calls']==14
    validate_result(full_result(cfg),cfg)


def test_derive_only_paths_and_explicit_paper_hyperparameter_differences():
    template=dict(data=dict(path='YOUR_PATH',dataset='nn5_daily'),
        optim=dict(epochs=10000,batch_size=128,lr=.001),
        model=dict(sigma=.5,decoder=dict(d_state=64,n_layers=4)))
    before=copy.deepcopy(template)
    released=derive_config(template,dict(protocol='released'),'private')
    assert released['optim']==template['optim'] and released['model']==template['model']
    paper=derive_config(template,dict(protocol='paper_literal'),'private')
    assert paper['optim']==dict(epochs=7000,batch_size=64,lr=.001)
    assert paper['model']==dict(sigma=.1,decoder=template['model']['decoder'])
    assert template==before


@pytest.mark.parametrize('kind',['short_training','lost_row','early_eval','short_evaluator','short_sequence','missing_reconstruction'])
def test_incomplete_native_execution_cannot_be_published(kind):
    cfg=config();result=full_result(cfg)
    if kind=='short_training':result['generator_updates']-=1
    elif kind=='lost_row':result['epoch_samples']['0']-=1
    elif kind=='early_eval':result['metric_history'][0]['generator_updates']-=2
    elif kind=='short_evaluator':result['metric_history'][0]['classifier_updates']=99
    elif kind=='short_sequence':result['metric_history'][0]['generated_shape']=[22,727,1]
    else:result['reconstruction_history'].pop()
    with pytest.raises(ValueError):
        validate_result(result,cfg)
