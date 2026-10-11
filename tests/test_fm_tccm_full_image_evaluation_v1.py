import copy
import json
import sys
import pytest
import numpy as np
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha
from scripts.flow_matching.tccm_full_image_evaluation_contract_v1 import validate_queue, complete
from scripts.flow_matching.run_tccm_full_image_evaluation_v1 import direct_replay_score, independent_metrics
from scripts.flow_matching.flow_mismatching_image_metrics_v1 import per_category_metrics


def full_queue():
    return read(ROOT/'configs/experiments/fm_tccm_full_image_evaluation_queue.v2.json')


def test_full_native_five_seed_original50epoch_queue():
    queue=full_queue()
    assert validate_queue(queue)
    assert len(queue['cases'])==5
    assert len(queue['cases'][0]['paper_reported'])==7
    assert {x['seed'] for x in queue['cases']}==set(range(1103,1108))
    for action in ('remove','duplicate'):
        broken=copy.deepcopy(queue)
        if action=='remove': broken['cases'].pop()
        else: broken['cases'][-1]=copy.deepcopy(broken['cases'][0])
        with pytest.raises(ValueError):validate_queue(broken)


@pytest.mark.parametrize('field,value',[('test_time',.5),('channel_residual','squared_L2'),('top_fraction',.1)])
def test_fixed_original_eq20_parameters_cannot_be_replaced(field,value):
    queue=full_queue();queue['cases'][0]['score_parameters'][field]=value
    with pytest.raises(ValueError):validate_queue(queue)


@pytest.mark.parametrize('field,value',[('checkpoint_epoch',49),('checkpoint','smoke.pt'),('variant','pooled'),
    ('dataset','visa'),('paper_row_index',0)])
def test_full_original_checkpoint_and_paper_binding_cannot_be_changed(field,value):
    queue=full_queue();queue['cases'][0][field]=value
    with pytest.raises(ValueError):validate_queue(queue)


def test_direct_fixed_t1_unsquared_replay_agrees_with_producer_and_analytical_scalar():
    sys.path.insert(0,str(ROOT/'src'))
    from iia_benchmark.models.fm_tccm_image import score_tccm_image
    class Velocity(torch.nn.Module):
        def forward(self,images,time,labels):
            assert torch.equal(time,torch.ones_like(time))
            assert torch.equal(labels,torch.tensor([3]))
            residual=torch.zeros_like(images)
            residual[:,0]=3.;residual[:,1]=4.
            return residual-images
    model=Velocity().eval();images=torch.randn(1,3,128,128);labels=torch.tensor([3])
    with torch.inference_mode():
        heat,score=score_tccm_image(model,images,labels=labels)
        direct_heat,direct_score=direct_replay_score(model,images,labels)
    assert torch.allclose(heat,torch.full_like(heat,5.),atol=1e-6)
    assert torch.allclose(score,torch.tensor([10.]),atol=1e-6)
    assert torch.allclose(heat,direct_heat,rtol=2e-6,atol=2e-6)
    assert torch.allclose(score,direct_score,rtol=2e-6,atol=2e-6)


def test_independent_whole_category_metrics_detect_corrupted_score_claim():
    masks=np.zeros((4,8,8),dtype=np.uint8);masks[1,2:4,2:4]=1;masks[3,3:5,3:5]=1
    labels=np.array([0,1,0,1]);categories=np.array([0,0,1,1]);names=['first','second']
    heat=np.arange(256,dtype=np.float32).reshape(4,8,8)/256
    scores=np.array([.2,.6,.9,.3],dtype=np.float32)
    arrays=dict(masks=masks,labels=labels,categories=categories)
    candidate=dict(local_metrics=per_category_metrics(heat,scores,masks,labels,categories,names,
        fpr_limit=.3,connectivity=8))
    queue=full_queue()
    rows,means=independent_metrics(arrays,names,heat,scores,queue,candidate,np)
    assert len(rows)==2 and set(means)==set(queue['cases'][0]['paper_reported'])
    for key in ('image_AP','pixel_AUROC','pixel_AUPRO'):
        broken=copy.deepcopy(candidate);broken['local_metrics']['per_category'][0]['metrics'][key]+=.01
        with pytest.raises(ValueError):independent_metrics(arrays,names,heat,scores,queue,broken,np)


def test_completion_rejects_a_short_dataset_even_with_success_flags(tmp_path,monkeypatch):
    import scripts.flow_matching.tccm_full_image_evaluation_contract_v1 as contract
    case=full_queue()['cases'][0];case['output_directory']='output'
    report=dict(passed=True,case=case,every_native_image_model_replayed=True,
        all_native_heatmaps_and_image_scores_equal=True,original_full_state_dictionary_loaded_strictly=True,
        independent_all_category_metrics_passed=True,native_performance_measured=True,
        complete_original_test_images=16,complete_original_categories=15)
    path=tmp_path/'report.json';path.write_text(json.dumps(report),encoding='utf-8')
    out=tmp_path/'output';out.mkdir()
    (out/'fully_verified_score_result.json').write_text(json.dumps(dict(report=str(path),report_sha256=sha(path),passed=True)),encoding='utf-8')
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Every native'):complete(case)
