import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_crossad_author_runtime import build_model
from iia_benchmark.models.fm_crossad_ablation import build_ablation, COMPONENTS
from iia_benchmark.models.fm_crossad_memory import install_activation_checkpointing

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT/'configs/experiments/fm_crossad_author_execution.v1.json').read_text())
PARAMETERS = json.loads((ROOT/SETTINGS['source']/'configs/GECCO/model_configs_0.json').read_text())


@pytest.fixture
def native():
    import pandas as pd
    frame = pd.read_pickle(ROOT/SETTINGS['cache_root']/'GECCO.pkl')
    return torch.from_numpy(frame.drop(columns=['label']).to_numpy()[:128][None].astype(np.float32))


@pytest.mark.parametrize('row', [1,2,3,4,5])
def test_each_reconstructed_component_combination_has_finite_native_loss_and_full_scores(row, native):
    torch.set_num_threads(2); torch.manual_seed(2025)
    model = build_ablation(ROOT/SETTINGS['runtime_root'], PARAMETERS, row)
    assert model.components_ == COMPONENTS[row] and not model.source_equivalence_
    model.train(); loss, _ = model(native, None, None, None); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if row in [1,2,3,4]:
        assert model.context_net is None
    if row in [1,2]:
        assert model.target_lengths_[-1] == 128 and model.direct_bank is None
    if row == 4:
        assert model.direct_bank.query_len == sum(model.ms_p_lens[:-1])
    model.eval(); before = {k:v.clone() for k,v in model.named_buffers()}
    with torch.no_grad():
        first, _ = model.infer(native, None, None, None)
        second, _ = model.infer(native, None, None, None)
    assert first.shape == native.shape and torch.isfinite(first).all() and torch.equal(first, second)
    assert all(torch.equal(v, before[k]) for k,v in model.named_buffers())


@pytest.mark.parametrize('deep', [False, True])
def test_checkpointing_preserves_output_gradients_rng_and_single_ema_update(native,deep):
    torch.set_num_threads(2); torch.manual_seed(2025)
    plain = build_model(ROOT/SETTINGS['runtime_root'], PARAMETERS)
    if deep:
        from iia_benchmark.models.fm_crossad_memory_v2 import install_activation_checkpointing as wrap
    else:
        wrap=install_activation_checkpointing
    wrapped = wrap(copy.deepcopy(plain))
    plain.train(); wrapped.train(); rng = torch.get_rng_state()
    first, _ = plain(native, None, None, None); first.backward(); after = torch.get_rng_state().clone()
    torch.set_rng_state(rng); second, _ = wrapped(native, None, None, None); second.backward()
    assert torch.equal(first, second) and torch.equal(torch.get_rng_state(), after)
    for (name, p), (other, q) in zip(plain.named_parameters(), wrapped.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)
    assert all(torch.equal(v, dict(wrapped.named_buffers())[name]) for name,v in plain.named_buffers())


def test_all_missing_table3_rows_have_callable_configs_and_original_full_row_bindings():
    queue = json.loads((ROOT/'configs/experiments/fm_crossad_ablation_queue.v1.json').read_text())
    assert len(queue['jobs']) == 90 and {j['ablation_row'] for j in queue['jobs']} == {1,2,3,4,5}
    assert {j['dataset'] for j in queue['jobs']} == {'PSM','MSL','GECCO'}
    report = json.loads((ROOT/'docs/reports/fm_crossad_ablation_registration_2026-10-09.json').read_text())
    assert len(report['full_row6_control_bindings']) == 18
    complete = json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v3.json').read_text())
    assert len(complete['jobs']) == 139 and all(j['activation_checkpointing'] == 'pure_encoder_decoder' for j in complete['jobs'])
    assert sum(j['reconstructed_ablation'] for j in complete['jobs']) == 90


@pytest.mark.parametrize('row',[1,2,3,4,5])
def test_deep_replay_preserves_each_reconstructed_row_gradient_and_ema(row,native):
    from iia_benchmark.models.fm_crossad_memory_v2 import install_activation_checkpointing as wrap
    torch.set_num_threads(2);torch.manual_seed(2025)
    plain=build_ablation(ROOT/SETTINGS['runtime_root'],PARAMETERS,row)
    wrapped=wrap(copy.deepcopy(plain));plain.train();wrapped.train();rng=torch.get_rng_state()
    first,_=plain(native,None,None,None);first.backward();after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);second,_=wrapped(native,None,None,None);second.backward()
    assert torch.equal(first,second) and torch.equal(torch.get_rng_state(),after)
    for (name,p),(other,q) in zip(plain.named_parameters(),wrapped.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)
    assert all(torch.equal(v,dict(wrapped.named_buffers())[name]) for name,v in plain.named_buffers())


def test_deep_queue_has_original_budgets_and_all_139_callable_experiments():
    old=json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v3.json').read_text())
    new=json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v4.json').read_text())
    assert len(new['jobs'])==139
    for a,b in zip(old['jobs'],new['jobs']):
        assert a['id']==b['id'] and a['output_directory']==b['output_directory']
        assert a['model_parameters']==b['model_parameters'] and a['train_parameters']==b['train_parameters']
        assert b['activation_checkpointing']=='pure_layer_v2'
        assert (ROOT/b['runner']).is_file()
