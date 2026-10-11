import ast
import copy
import json
import sys
import numpy as np
import pytest
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha
from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import validate_queue, complete, native_solver_facts
from scripts.flow_matching.run_wt_full_image_evaluation_v1 import direct_replay_score, independent_metrics
from scripts.flow_matching.flow_mismatching_image_metrics_v1 import per_category_metrics


def full_queue():
    return read(ROOT / 'configs/experiments/fm_wt_full_image_evaluation_queue.v1.json')


def native_core():
    sys.path.insert(0, str(ROOT / 'src'))
    from iia_benchmark.models.fm_reflect_wt_native import load_verified_wt_flow
    config = read(ROOT / 'configs/models/fm_wt_mvtec_controlled_image_domain.v1.json')
    return load_verified_wt_flow(**config['native_source'], asset_root=ROOT)


def test_all_native_steps_and_five_unique_training_seeds_without_winning_step_selection():
    queue = full_queue()
    assert validate_queue(queue)
    assert len(queue['cases']) == 30
    assert queue['solver_facts']['native_step_scan'] == [1, 10, 20, 50, 100, 200]
    assert queue['solver_facts']['active_native_step_scan_categories'] == ['transistor']
    assert not queue['solver_selected_using_test_labels']
    for action in ('remove', 'duplicate', 'overwrite_output'):
        broken = copy.deepcopy(queue)
        if action == 'remove': broken['cases'].pop()
        elif action == 'duplicate': broken['cases'][-1] = copy.deepcopy(broken['cases'][0])
        else: broken['cases'][-1]['output_directory'] = broken['cases'][0]['output_directory']
        with pytest.raises(ValueError): validate_queue(broken)


@pytest.mark.parametrize('field,value', [('integration_steps', 9), ('integration_steps', True),
    ('integration', 'midpoint'), ('epsilon', 1e-3), ('interval', [1., 0.]),
    ('energy', 'negative_half_channel_mean'), ('top_fraction', .03)])
def test_registered_related_source_budget_energy_and_normalization_cannot_be_replaced(field, value):
    queue = full_queue(); queue['cases'][0]['score_parameters'][field] = value
    with pytest.raises(ValueError): validate_queue(queue)


@pytest.mark.parametrize('field,value', [('checkpoint_epoch', 100), ('checkpoint', 'smoke.pt'),
    ('variant', 'pooled'), ('dataset', 'visa'), ('paper_row_index', 0)])
def test_full2000epoch_target_paper_and_native_data_bindings_required(field, value):
    queue = full_queue(); queue['cases'][0][field] = value
    with pytest.raises(ValueError): validate_queue(queue)


@pytest.mark.parametrize('flag', ['original_target_solver_and_step_identity_certified',
    'original_author_equivalence_certified', 'original_WT_best_feature_setting_reproduced',
    'strict_TAB_result', 'solver_selected_using_test_labels', 'all_original_experiments_complete'])
def test_related_source_sensitivity_cannot_claim_original_author_or_TAB_equivalence(flag):
    queue = full_queue(); queue[flag] = True
    with pytest.raises(ValueError): validate_queue(queue)


def test_native_AST_solver_facts_reject_changed_original_source(tmp_path):
    queue = full_queue(); model = read(ROOT / queue['evaluation_model_configuration'])
    sources = copy.deepcopy(model['native_solver_source_receipts'])
    assert native_solver_facts(sources) == queue['solver_facts']
    source = next(x for x in sources if x['role'] == 'infer')
    path = tmp_path / 'infer.py'
    path.write_bytes((ROOT / source['path']).read_bytes().replace(b'[1,10,20,50,100,200]', b'[1,10,20,50,100,199]'))
    source['path'] = str(path)
    with pytest.raises(ValueError, match='differs'): native_solver_facts(sources)
    source['sha256'] = sha(path)
    with pytest.raises(ValueError, match='step scan'): native_solver_facts(sources)


@pytest.mark.parametrize('steps', [1, 10, 20, 50, 100, 200])
def test_complete128RGB_endpoint_matches_actual_original_AST_loop_and_analytic_left_grid(steps):
    from iia_benchmark.models.fm_wt_image_evaluation import score_wt_image
    core = native_core()
    queue = full_queue(); model_config = read(ROOT / queue['evaluation_model_configuration'])
    source = next(x for x in model_config['native_solver_source_receipts'] if x['role'] == 'infer')
    tree = ast.parse((ROOT / source['path']).read_text(encoding='utf-8'))
    generator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_pdf')
    selected = [n for n in generator.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dt' for t in n.targets)) or isinstance(n, ast.For)]
    original_loop = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), '<original WT infer.py ODE body>', 'exec')
    images = torch.arange(3 * 128 * 128, dtype=torch.float32).reshape(1, 3, 128, 128) / 1000
    labels = torch.tensor([3])
    times = []
    class Velocity(torch.nn.Module):
        def forward(self, x, t, labels=None):
            if labels is not None: assert torch.equal(labels, torch.tensor([3]))
            times.append(float(t[0]))
            return torch.ones_like(x) * (.2 + t[:, None, None, None] * .1)
    velocity = Velocity().eval()
    with torch.inference_mode():
        heat, score = score_wt_image(velocity, images, native_core=core, epsilon=1e-5,
            steps=steps, top_fraction=.01, labels=labels)
        producer_times = list(times); times.clear()
        direct_heat, direct_score = direct_replay_score(velocity, images, labels, epsilon=1e-5, steps=steps, top_fraction=.01)
        assert times == producer_times
        mapped = torch.nn.functional.layer_norm(images, images.shape[1:], eps=1e-5)
        namespace = dict(torch=torch, rf=core.flow, step=steps, model=velocity, x_t=mapped,
                         c=type('Config', (), dict(device=images.device))())
        exec(original_loop, namespace)
        native_energy = .5 * namespace['x_t'].square().sum(dim=1)
        analytic_energy = .5 * (mapped + .2 + .1 * (steps - 1) / (2 * steps)).square().sum(dim=1)
    assert torch.equal(heat, native_energy)
    assert torch.equal(heat, direct_heat)
    assert torch.allclose(heat, analytic_energy, rtol=1e-4, atol=1e-4)
    assert torch.allclose(score, direct_score, rtol=2e-6, atol=2e-6)
    assert len(producer_times) == steps and producer_times[0] == 0. and producer_times[-1] < 1.


def test_Eq25_positive_channel_SUM_not_native_negative_MEAN():
    from iia_benchmark.models.fm_wt_image_evaluation import score_wt_image
    core = native_core()
    class Velocity(torch.nn.Module):
        def forward(self, images, time, labels):
            return images.new_tensor([1., 2., 3.]).reshape(1, 3, 1, 1).expand_as(images)
    heat, score = score_wt_image(Velocity(), torch.zeros(1, 3, 128, 128), native_core=core,
        epsilon=1e-5, steps=1, top_fraction=.01, labels=torch.tensor([0]))
    assert torch.equal(heat, torch.full_like(heat, 7.))
    assert torch.equal(score, torch.tensor([14.]))


def test_independent_category_metrics_reject_corrupted_performance_claim():
    masks = np.zeros((4, 8, 8), dtype=np.uint8); masks[1, 2:4, 2:4] = 1; masks[3, 3:5, 3:5] = 1
    labels = np.array([0, 1, 0, 1]); categories = np.array([0, 0, 1, 1]); names = ['first', 'second']
    heat = np.arange(256, dtype=np.float32).reshape(4, 8, 8) / 256
    scores = np.array([.2, .6, .9, .3], dtype=np.float32)
    arrays = dict(masks=masks, labels=labels, categories=categories)
    candidate = dict(local_metrics=per_category_metrics(heat, scores, masks, labels, categories, names, fpr_limit=.3, connectivity=8))
    rows, means = independent_metrics(arrays, names, heat, scores, full_queue(), candidate, np)
    assert len(rows) == 2 and len(means) == 7
    for key in ('image_AP', 'pixel_AUROC', 'pixel_AUPRO'):
        broken = copy.deepcopy(candidate); broken['local_metrics']['per_category'][0]['metrics'][key] += .01
        with pytest.raises(ValueError): independent_metrics(arrays, names, heat, scores, full_queue(), broken, np)


def test_success_flags_do_not_certify_shortened_native_test_set(tmp_path, monkeypatch):
    import scripts.flow_matching.wt_full_image_evaluation_contract_v1 as contract
    case = full_queue()['cases'][0]; case['output_directory'] = 'output'
    report = dict(passed=True, case=case, every_native_image_model_replayed=True,
        all_native_heatmaps_and_image_scores_equal=True, original_full_state_dictionary_loaded_strictly=True,
        independent_all_category_metrics_passed=True, native_performance_measured=True,
        complete_original_test_images=16, complete_original_categories=15)
    path = tmp_path / 'report.json'; path.write_text(json.dumps(report), encoding='utf-8')
    out = tmp_path / 'output'; out.mkdir()
    (out / 'fully_verified_score_result.json').write_text(json.dumps(dict(report=str(path), report_sha256=sha(path), passed=True)), encoding='utf-8')
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Every native WT'): complete(case)
