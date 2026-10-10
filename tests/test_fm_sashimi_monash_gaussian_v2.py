import copy
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from scripts.flow_matching import run_sashimi_monash_gaussian_v2 as worker
from scripts.flow_matching import run_sashimi_monash_gaussian_queue_v2 as controller
from scripts.flow_matching.sashimi_gaussian_adapter_v1 import make_model,backbone_parameters


def stub_backbone(torch):
    class Constant(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__();self.mean=torch.nn.Parameter(torch.tensor(.4));self.inputs=[]
        def forward(self,x):return torch.zeros_like(x)+self.mean
        def default_state(self,*args,**kwargs):return None
        def step(self,previous,state=None):
            self.inputs.append(previous.detach().clone())
            return torch.zeros_like(previous)+self.mean,state
    return Constant


def test_gaussian_objective_matches_distribution_density_and_trains():
    import torch
    config=dict(backbone=backbone_parameters('paper_flat4'),activation='identity',sigma=.1)
    model=make_model(torch,stub_backbone(torch),config)
    x=torch.tensor([[[.1],[.7],[1.]]])
    expected=-torch.distributions.Normal(model.backbone.mean,.1).log_prob(x).mean()
    assert torch.allclose(model(x),expected)
    model(x).backward();assert torch.isfinite(model.backbone.mean.grad)
    assert model.backbone.mean.grad!=0


def test_autoregressive_samples_feed_back_previous_gaussian_draw():
    import torch
    model=make_model(torch,stub_backbone(torch),dict(backbone={},activation='identity',sigma=.1))
    torch.manual_seed(42);draws=torch.stack([torch.randn(2,1) for _ in range(5)],dim=1)
    torch.manual_seed(42);actual=model.generate(2,5,device='cpu')
    assert torch.allclose(actual,.4+.1*draws)
    assert torch.equal(model.backbone.inputs[0],torch.zeros(2,1))
    for i in range(1,5):assert torch.equal(model.backbone.inputs[i],actual[:,i-1])


def test_sigmoid_controls_conditional_mean_without_clipping_gaussian_support(monkeypatch):
    import torch
    model=make_model(torch,stub_backbone(torch),dict(backbone={},activation='sigmoid',sigma=.1))
    monkeypatch.setattr(torch,'randn_like',lambda x:torch.full_like(x,10))
    assert model.generate(1,1,device='cpu').item()>1


def test_unequal_test_batches_keep_native_sum_and_correct_weighted_mean_distinct():
    result=worker.test_batch_summary([dict(mean=2.,elements=128),dict(mean=8.,elements=6)],2)
    assert result['native_sum_batch_means']==10
    assert result['unweighted_mean_batch_means']==5
    assert result['element_weighted_mean']==pytest.approx(304/134)
    with pytest.raises(ValueError,match='incomplete'):worker.test_batch_summary(result['records'],3)


def test_full_temperature_evaluator_and_generator_budgets():
    full=worker.budget(dict(train_samples=25657,test_samples=6415,optim=dict(epochs=1000)))
    assert full['generator_updates']==401000
    assert full['classifier_updates']==8100 and full['predictor_updates']==5100
    assert full['classifier_test_batches']==21 and full['predictor_test_batches']==51


def full_fixture(tmp_path):
    shapes={'fred_md':(107,728),'nn5_daily':(111,791),'solar_weekly':(137,52),'temperature_rain':(32072,725)}
    queue=dict(queue_path='queue.json',source_root='source',source_receipts=[],jobs=[],
        datasets={d:dict(samples=n,length=l,yaml=d+'.yaml') for d,(n,l) in shapes.items()},
        seeds=[42,1103,1104,1105,1106],profiles=['paper_flat4','ls4_decoder_pool1_control'])
    for dataset,(n,length) in shapes.items():
        for profile in queue['profiles']:
            for seed in queue['seeds']:
                c=dict(dataset=dataset,profile=profile,seed=seed,samples=n,length=length,train_samples=int(.8*n),test_samples=n-int(.8*n),
                    optim=dict(epochs=1000 if dataset=='temperature_rain' else 7000,batch_size=64,lr=.001,weight_decay=0,lamb=.999,start_step=200),
                    sigma=.1,backbone=backbone_parameters(profile),diagnostic=False,author_equivalence_certified=False,
                    activation='sigmoid' if dataset=='temperature_rain' else 'identity',source_config='source/configs/monash/'+dataset+'.yaml')
                c['budget']=worker.budget(c);name=dataset+'__'+profile+'__'+str(seed)+'.json'
                (tmp_path/name).write_text(json.dumps(c));queue['jobs'].append(dict(id=name,model_config=name,model_config_sha256=worker.sha(tmp_path/name)))
    return queue


def test_all_four_tasks_two_profiles_five_seeds_and_no_budget_reduction(tmp_path,monkeypatch):
    queue=full_fixture(tmp_path);monkeypatch.setattr(worker,'ROOT',tmp_path)
    worker.verify(queue)
    assert sum(json.loads((tmp_path/j['model_config']).read_text())['budget']['generator_updates'] for j in queue['jobs'])==4430000
    job=queue['jobs'][0];p=tmp_path/job['model_config'];value=json.loads(p.read_text());value['optim']['epochs']=1
    p.write_text(json.dumps(value));job['model_config_sha256']=worker.sha(p)
    with pytest.raises(ValueError,match='paper budget'):worker.verify(queue)


def test_complete_training_without_both_full_evaluations_is_not_complete():
    c=dict(train_samples=85,test_samples=22,length=728,optim=dict(epochs=2));full=worker.budget(c)
    r=dict(generator_updates=full['generator_updates'],epoch_updates={'0':2,'1':2},epoch_samples={'0':85,'1':85},metric_history=[])
    with pytest.raises(ValueError,match='Both'):worker.validate_result(r,c)
    r['metric_history']=[dict(recipe=recipe,classifier_updates=100,predictor_updates=100,reference_shape=[22,728,1],generated_shape=[22,728,1],metrics={'clf_score':.7}) for recipe in full['evaluation_recipes']]
    worker.validate_result(r,c);r['metric_history'][1]['metrics']['clf_score']=float('nan')
    with pytest.raises(ValueError,match='invalid'):worker.validate_result(r,c)


def test_old_uncorrected_or_partial_data_proof_is_not_accepted(tmp_path,monkeypatch):
    queue=full_fixture(tmp_path);monkeypatch.setattr(controller,'ROOT',tmp_path)
    (tmp_path/'queue.json').write_text('{}');(tmp_path/'repair.json').write_text(json.dumps(dict(official_cauchy_source_sha256='source')))
    queue.update(data_audit_output='proof.json',cauchy_correction_config='repair.json')
    proof=dict(passed=True,diagnostic_only=True,queue_sha256=worker.sha(tmp_path/'queue.json'),cases=40,
        records=[dict(id=j['id'],trajectory_split_disjoint=True,full_original_normalization_bitwise=True) for j in queue['jobs']])
    (tmp_path/'proof.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='backend'):controller.proof_ready(queue,'data_audit')
    proof['backend_correction']=dict(algorithmic_correction=True,official_source_sha256='source')
    (tmp_path/'proof.json').write_text(json.dumps(proof));assert controller.proof_ready(queue,'data_audit')
    proof['records'].pop();(tmp_path/'proof.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='All40'):controller.proof_ready(queue,'data_audit')


def test_native_metric_frame_found_through_scheduler_optimizer_wrapper(tmp_path):
    filename=tmp_path/'metrics.py';metrics=SimpleNamespace(__file__=str(filename))
    def observed_step():
        frame=worker.metric_frame(metrics)
        return frame.f_code.co_name if frame else None
    def scheduler_wrapped_step():return observed_step()
    namespace=dict(step=scheduler_wrapped_step)
    exec(compile('def compute_classification_score():\n    return step()\n',str(filename),'exec'),namespace)
    assert namespace['compute_classification_score']()=='compute_classification_score'
    assert observed_step() is None
