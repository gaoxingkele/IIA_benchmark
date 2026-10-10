from types import SimpleNamespace

import pytest

from scripts.flow_matching.run_spectral_table3_original_v2 import optimizer_assignments, original_optimizer


def test_original_optimizer_with_logger_scope_and_original_constructor(tmp_path):
    entry = tmp_path / 'main.py'
    entry.write_text('def main(args):\n    name = "ignored"\n    with PrintLogger() as logger:\n        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)\n', encoding='utf-8')
    observed = []
    torch = SimpleNamespace(optim=SimpleNamespace(AdamW=lambda *a, **kw: observed.append((a, kw)) or 'original'))
    model = SimpleNamespace(parameters=lambda: 'all_original_parameters')
    args = SimpleNamespace(learning_rate=.0003, weight_decay=.00001)
    assert original_optimizer(entry, model, args, torch) == 'original'
    assert observed == [(('all_original_parameters',), {'lr': .0003, 'weight_decay': .00001})]


def test_complete_spectral_optimizer_assignments_keep_original_order(tmp_path):
    entry = tmp_path / 'main.py'
    entry.write_text('def main(args):\n    with PrintLogger() as logger:\n        muon_params = model.net.spectral_weights() + model.net.spectral_gains_biases()\n        adamw_params = model.net.emb_parameters() + model.net.non_spectral_weights() + model.net.non_spectral_gains_biases()\n        param_groups = [dict(params=muon_params, use_muon=True), dict(params=adamw_params, use_muon=False)]\n        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)\n', encoding='utf-8')
    assert [n.targets[0].id for n in optimizer_assignments(entry)] == ['muon_params', 'adamw_params', 'param_groups', 'optimizer']


def test_conditional_or_incomplete_optimizer_is_not_silently_substituted(tmp_path):
    entry = tmp_path / 'main.py'
    for text in ['def main(args):\n    if args.repair:\n        optimizer = substitute()\n',
                 'def main(args):\n    with PrintLogger():\n        param_groups = []\n        optimizer = substitute(param_groups)\n']:
        entry.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError, match='Complete original optimizer'):
            optimizer_assignments(entry)
