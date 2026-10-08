"""Behavioral CPU checks only; synthetic data does not measure performance."""
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from iia_benchmark.models.fm_supplemental import (
    local_prior_mean, sample_jfi_prior, JFIVelocity, jfi_objective, jfi_impute,
    flow_mismatch, InfiniChannelMixer, StaticChannelEmbedding,
)
from iia_benchmark.models.fm_supplemental_author import load_sensitivehue, sensitivehue_loss, FlowMismatchUNet
from iia_benchmark.models.fm_supplemental_variants import SensitiveHUEAblation,hue_variant_loss
from iia_benchmark.models.fm_supplemental_moment import MomentICMT5


def test_lpd_uses_only_observed_and_handles_empty_columns():
    x = torch.tensor([[[float('nan'),99.],[2.,float('nan')],[float('nan'),float('nan')],[6.,float('nan')],[float('nan'),float('nan')]]])
    mask = torch.tensor([[[0,0],[1,0],[0,0],[1,0],[0,0]]],dtype=torch.bool)
    prior = local_prior_mean(x,mask)
    assert torch.equal(prior[0,:,0],torch.tensor([2.,2.,4.,6.,6.]))
    assert torch.equal(prior[0,:,1],torch.zeros(5))
    assert torch.equal(sample_jfi_prior(x,mask,variance=0),prior)


def test_euler_preserves_observed_and_step_scaling():
    x=torch.full((1,4,4),float('nan'));mask=torch.zeros_like(x,dtype=torch.bool)
    x[:,0,:]=5;mask[:,0,:]=True
    velocity=lambda state,time: torch.ones_like(state)
    one=jfi_impute(velocity,x,mask,steps=1,prior_variance=0)
    multi=jfi_impute(velocity,x,mask,steps=10,prior_variance=0)
    torch.testing.assert_close(one,multi)
    assert (one[:,0,:]==5).all() and (one[:,1:,:]==6).all()


def test_jfi_does_not_silently_skip_required_teacher():
    x=torch.zeros(2,4,4);mask=torch.zeros_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='teacher'):
        jfi_objective(lambda x,t:x,x,mask)


@pytest.mark.parametrize('mixed,ca',[(True,True),(False,False)])
def test_jfi_unet_backward_odd_shapes(mixed,ca):
    # Keep CPU thread count bounded and restore it for the shared test session.
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        net=JFIVelocity(width=4,mixed=mixed,channel_attention=ca)
        x=torch.randn(2,9,7);mask=torch.rand_like(x)>0.5
        teacher=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(63,3))
        for p in teacher.parameters():p.requires_grad_(False)
        loss=jfi_objective(net,x,mask,teacher=teacher,task='classification')
        assert torch.isfinite(loss['loss'])
        loss['loss'].backward()
        assert net.output.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in teacher.parameters())
        ablated=jfi_objective(net,x,mask,joint=False)
        assert torch.equal(ablated['loss'],ablated['flow'])
    finally:torch.set_num_threads(previous)


@pytest.mark.parametrize('aggregator',['minimum','average','percentile'])
def test_flow_mismatch_matches_independent_equations(aggregator):
    y=torch.tensor([[[[1.,2.]],[[3.,4.]]]])
    seeds=torch.stack([torch.zeros_like(y),torch.ones_like(y),-torch.ones_like(y)])
    # Zero model isolates geometric residual and known t² average.
    heat,score=flow_mismatch(lambda x,t:torch.zeros_like(x),y,paths=3,times=2,
                             aggregation=aggregator,percentile=.5,seeds=seeds)
    discrepancies=(y[None]-seeds).square().sum(2)
    if aggregator=='minimum':expected=discrepancies.min(0).values
    elif aggregator=='average':expected=discrepancies.mean(0)
    else:expected=discrepancies.sort(0).values[1]
    expected=expected*((1/3)**2+(2/3)**2)/2
    torch.testing.assert_close(heat,expected)
    torch.testing.assert_close(score,2*expected.flatten(1).max(1).values)


def test_icm_matches_channel_loop_and_has_no_cross_sample_memory():
    torch.manual_seed(7)
    q,k,v=[torch.randn(2,3,2,4,5) for _ in range(3)]
    net=InfiniChannelMixer(2)
    memory=torch.zeros(2,2,5,5);z=torch.zeros(2,2,5)
    for c in range(3):
        pk=F.elu(k[:,c])+1
        memory+=pk.transpose(-2,-1)@v[:,c];z+=pk.sum(-2)
    expected=[]
    for c in range(3):
        pq=F.elu(q[:,c])+1
        retrieved=pq@memory/((pq*z[:,:,None,:]).sum(-1,keepdim=True)+net.epsilon)
        local=(q[:,c]@k[:,c].transpose(-2,-1)/math.sqrt(5)).softmax(-1)@v[:,c]
        expected.append((retrieved+local)/2)
    torch.testing.assert_close(net(q,k,v),torch.stack(expected,1))
    torch.testing.assert_close(net(q,k,v)[0],net(q[:1],k[:1],v[:1])[0])
    net(q,k,v).square().mean().backward()
    assert net.beta.grad is not None


def test_icm_permutation_independence_and_static_variants():
    q,k,v=[torch.randn(1,3,2,4,5) for _ in range(3)]
    permutation=torch.tensor([2,0,1])
    for mode in ['icm','independent','concatenation']:
        net=InfiniChannelMixer(2,mode=mode,train_beta=False)
        assert not net.beta.requires_grad
        torch.testing.assert_close(net(q[:,permutation],k[:,permutation],v[:,permutation]),
                                   net(q,k,v)[:,permutation])
    layer=StaticChannelEmbedding(3,5)
    tokens=torch.zeros(1,3,4,5)
    assert not torch.equal(layer(tokens)[:,0],layer(tokens)[:,1])


def test_sensitivehue_author_forward_and_loss_variants():
    root=Path(__file__).resolve().parents[1]/'experiments/runs/fm_code_completion/sources/sensitivehue/original'
    if not root.exists():pytest.skip('ignored author snapshot not installed')
    x=torch.randn(2,8,4)
    for sfr in [True,False]:
        model=load_sensitivehue(root,step_num_in=8,f_in=4,dim_model=8,head_num=2,
                               dim_hidden_fc=16,statistical_removal=sfr,dropout=0)
        rec,logp=model(x)
        assert rec.shape==x.shape and torch.isfinite(logp).all()
        loss=sensitivehue_loss(rec,x,logp)
        assert torch.isfinite(loss);loss.backward()
        torch.testing.assert_close(sensitivehue_loss(rec,x,logp,probabilistic=False),F.mse_loss(rec,x))


def test_sensitivehue_probability_loss_matches_author_weighting():
    rec=torch.tensor([[[1.,2.],[3.,4.]]]);truth=torch.zeros_like(rec)
    logp=torch.tensor([[[.2,-.1],[.4,.3]]],requires_grad=True)
    variance=(-logp).exp().detach()
    expected=(variance*(rec.square()*logp.exp()-logp)/variance.mean((0,1)).pow(.75)).mean()
    torch.testing.assert_close(sensitivehue_loss(rec,truth,logp,alpha=.75),expected)


def test_flow_unet_source_adapter_and_conditional_boundary():
    root=Path(__file__).resolve().parents[1]/'experiments/runs/mtsad_protocol_audit/sources/flow_matching_library/original/examples/image/models'
    if not root.exists():pytest.skip('ignored Meta source snapshot not installed')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=FlowMismatchUNet(root,model_channels=32,num_res_blocks=1,channel_mult=(1,2),
                              attention_resolutions=(2,),num_classes=3)
        x=torch.randn(2,3,8,8);time=torch.tensor([.3,.7])
        with pytest.raises(ValueError):model(x,time)
        output=model(x,time,torch.tensor([0,2]))
        assert output.shape==x.shape and torch.isfinite(output).all()
        (output-x).square().mean().backward()
        assert model.model.out[-1].weight.grad.abs().sum()>0
    finally:torch.set_num_threads(previous)


def test_sensitivehue_relative_asset_path_ignores_cwd(monkeypatch,tmp_path):
    rel='experiments/runs/fm_code_completion/sources/sensitivehue/original'
    if not (Path(__file__).resolve().parents[1]/rel).exists():pytest.skip('snapshot absent')
    monkeypatch.chdir(tmp_path)
    model=load_sensitivehue(rel,step_num_in=8,f_in=4,dim_model=8,head_num=2,dim_hidden_fc=16)
    assert model.f_in==4


@pytest.mark.parametrize('beta',[0.,.5,1.])
def test_beta_nll_gradient_matches_paper_stop_gradient(beta):
    mean=torch.tensor([[[1.,2.]]],requires_grad=True)
    target=torch.zeros_like(mean)
    logp=torch.tensor([[[.2,-.3]]],requires_grad=True)
    loss=hue_variant_loss(mean,target,logp,mode='beta_nll',beta=beta)
    loss.backward()
    expected=(mean.detach()-target)*torch.exp((1-beta)*logp.detach())/mean.numel()
    torch.testing.assert_close(mean.grad,expected)
    expected_logp=.5*torch.exp(-beta*logp.detach())*(mean.detach().square()*logp.detach().exp()-1)/mean.numel()
    torch.testing.assert_close(logp.grad,expected_logp)


def _hue_variant(**kwargs):
    root=Path(__file__).resolve().parents[1]/'experiments/runs/fm_code_completion/sources/sensitivehue/original'
    if not root.exists():pytest.skip('SensitiveHUE author snapshot not installed')
    return SensitiveHUEAblation(str(root),step_num_in=8,f_in=4,dim_model=8,head_num=2,
                               dim_hidden_fc=16,dropout=0,**kwargs)


def test_faithful_trunk_gradient_is_mean_only_gradient():
    model=_hue_variant(loss_mode='faithful')
    x=torch.randn(2,8,4)
    preds=model(x);loss=model.objective(x,preds);loss.backward()
    trunk_grad=model.model.in_linear.weight.grad.clone()
    mean_grad=model.model.rec_linear.weight.grad.clone()
    assert model.model.sigma_linear.weight.grad.abs().sum()>0
    model.zero_grad();rec,_=model(x);(.5*(rec-x).square().mean()).backward()
    torch.testing.assert_close(model.model.in_linear.weight.grad,trunk_grad)
    torch.testing.assert_close(model.model.rec_linear.weight.grad,mean_grad)
    assert model.model.sigma_linear.weight.grad is None


def test_natural_model_heads_and_nll_match_gaussian_parameterization():
    model=_hue_variant(loss_mode='natural')
    with torch.no_grad():
        model.model.rec_linear.weight.zero_();model.model.rec_linear.bias.fill_(2)
        model.model.sigma_linear.weight.zero_();model.model.sigma_linear.bias.fill_(math.log(4))
    x=torch.randn(2,8,4);mean,logp=model(x)
    torch.testing.assert_close(mean,torch.full_like(mean,.5))
    expected=-torch.distributions.Normal(mean,torch.exp(-.5*logp)).log_prob(x).mean()
    actual=model.objective(x,(mean,logp))+.5*math.log(2*math.pi)
    torch.testing.assert_close(actual,expected)
    actual.backward();assert model.model.rec_linear.weight.grad is not None


@pytest.mark.parametrize('strategy',['sfr','revin','compression','mask','none'])
def test_hue_reconstruction_strategy_semantics(strategy):
    model=_hue_variant(strategy=strategy)
    x=torch.arange(64,dtype=torch.float32).reshape(2,8,4)
    with torch.no_grad():model.model.rec_linear.weight.zero_();model.model.rec_linear.bias.zero_()
    mask=model.create_mask(x,torch.Generator().manual_seed(5)) if strategy=='mask' else None
    predictions=model(x,mask=mask)
    expected=x.mean(1,keepdim=True).expand_as(x) if strategy=='revin' else torch.zeros_like(x)
    torch.testing.assert_close(predictions[0],expected)
    assert torch.isfinite(model.objective(x,predictions,mask))
    if strategy=='mask':
        assert mask.sum()==2 # one withheld timestamp per batch sample
        with pytest.raises(ValueError,match='same explicit'):model.objective(x,predictions)


@pytest.mark.parametrize('mixer,static',[('icm',None),('icm',3),('independent',None)])
def test_moment_t5_icm_full_forecast_forward_backward(mixer,static):
    root=Path(__file__).resolve().parents[1]/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
    if not root.exists():pytest.skip('MOMENT author snapshot not installed')
    pytest.importorskip('transformers')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=MomentICMT5(str(root),seq_len=16,patch_len=4,d_model=16,num_layers=2,
                          heads=2,d_ff=32,forecast_horizon=6,mixer=mixer,static_channels=static,dropout=0)
        x=torch.randn(2,3,16);out=model(x)
        assert out.shape==(2,3,6) and torch.isfinite(out).all()
        out.square().mean().backward()
        assert model.head.linear.weight.grad.abs().sum()>0
        if mixer=='icm':assert all(layer.beta.grad is not None for layer in model.icm_layers)
        if static is None:
            permutation=torch.tensor([2,0,1])
            torch.testing.assert_close(model(x[:,permutation]),model(x)[:,permutation],rtol=1e-4,atol=1e-5)
        # Independent batch samples must not share compressed memory.
        torch.testing.assert_close(model(x)[0],model(x[:1])[0],rtol=1e-4,atol=1e-5)
    finally:torch.set_num_threads(previous)


def test_flow_mismatch_condition_labels_repeat_in_path_major_order():
    images=torch.zeros(2,3,2,2);labels=torch.tensor([4,7]);seen=[]
    def conditional(state,time,category):
        seen.append(category.clone());return torch.zeros_like(state)
    flow_mismatch(conditional,images,paths=3,times=2,labels=labels)
    assert len(seen)==2
    assert all(torch.equal(x,torch.tensor([4,7,4,7,4,7])) for x in seen)


def test_moment_icm_adds_exactly_one_gate_per_layer_and_head():
    root=Path(__file__).resolve().parents[1]/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
    if not root.exists():pytest.skip('MOMENT source absent')
    kwargs=dict(seq_len=16,patch_len=4,d_model=16,num_layers=2,heads=2,d_ff=32,forecast_horizon=6,dropout=0)
    independent=MomentICMT5(str(root),mixer='independent',**kwargs)
    icm=MomentICMT5(str(root),mixer='icm',**kwargs)
    assert sum(p.numel() for p in icm.parameters())-sum(p.numel() for p in independent.parameters())==4
    frozen=MomentICMT5(str(root),freeze_backbone=True,train_beta=True,**kwargs)
    trainable=[name for name,p in frozen.named_parameters() if p.requires_grad]
    assert all('.beta' in name or name.startswith('head.') for name in trainable)


def test_moment_icm_padding_values_do_not_leak_into_memory():
    root=Path(__file__).resolve().parents[1]/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
    if not root.exists():pytest.skip('MOMENT source absent')
    model=MomentICMT5(str(root),seq_len=16,patch_len=4,d_model=16,num_layers=1,heads=2,
                      d_ff=32,forecast_horizon=6,dropout=0)
    values=torch.randn(1,2,16);mask=torch.ones(1,16);mask[:,-4:]=0
    changed=values.clone();changed[:,:,-4:]=1e6
    torch.testing.assert_close(model(values,input_mask=mask),model(changed,input_mask=mask))
