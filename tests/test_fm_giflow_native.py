import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching.giflow_native_protocol import complete, configured_loader, fingerprint, path_only_loader, sha, verify
from scripts.flow_matching.register_giflow_native import build_jobs


def test_full_native_roster_has_paired_seeds_and_unabridged_budgets():
    jobs = build_jobs()
    assert len(jobs) == 140 and len({j['id'] for j in jobs}) == 140
    for job in jobs:
        paired = next(j for j in jobs if j['dataset'] == job['dataset'] and j['recipe'] == job['recipe']
                      and j['seed'] == job['seed'] and j['track'] != job['track'])
        assert paired['arguments'] == job['arguments']
        assert job['arguments']['batch_size'] == 128 and job['arguments']['training_epoch'] == 300
        assert job['arguments']['stride'] == 1
    assert {j['arguments']['diffusion_type'] for j in jobs} == {'graph_time', 'graph', 'time', 'none'}
    assert {j['arguments']['missing_rate'] for j in jobs} == {.2, .3, .4, .5, .6}


def test_path_adaptation_only_changes_both_roots_without_changing_branch_result(tmp_path):
    path = tmp_path / 'data.py'
    path.write_text("def dataset_loading(dataset_name, multiplier=3):\n    if dataset_name == 'small':\n        root = f'./data/{dataset_name}'\n    else:\n        root = f'./data/{dataset_name}'\n    return root, multiplier * 7\n")
    loader, receipt = path_only_loader(SimpleNamespace(__file__=str(path)), tmp_path / 'data_root')
    for name in ('small', 'large'):
        assert loader(name, 4) == (str(tmp_path / 'data_root'), 28)
    assert receipt['expressions_replaced'] == 2
    assert sha(path) == receipt['source_sha256']


def test_root_patch_refuses_unreviewed_source_change(tmp_path):
    path = tmp_path / 'data.py'
    path.write_text("def dataset_loading(dataset_name):\n    return f'./changed/{dataset_name}'\n")
    with pytest.raises(ValueError, match='needs review'):
        path_only_loader(SimpleNamespace(__file__=str(path)), tmp_path)


def test_corrected_loader_deduplicates_same_root_and_rejects_changed_root(tmp_path):
    module = SimpleNamespace(dataset_loading=lambda **kw: kw)
    loader = configured_loader(module, tmp_path)
    assert loader(root=str(tmp_path), seed=393) == {'root': str(tmp_path), 'seed': 393}
    assert loader(seed=393) == {'root': str(tmp_path), 'seed': 393}
    with pytest.raises(ValueError, match='disagrees'):
        loader(root=str(tmp_path / 'other'))


def test_native_verifier_rejects_reduced_budgets_and_block_placeholder():
    job = build_jobs()[0]
    for key, value in (('batch_size', 2), ('training_epoch', 1), ('missing_type', 'block')):
        changed = json.loads(json.dumps(job))
        changed['arguments'][key] = value
        with pytest.raises(ValueError):
            verify({'jobs': [changed], 'source_receipts': []})


def test_completion_requires_real_artifact_hashes_and_exact_job_binding(tmp_path):
    job = {'id': 'job', 'output_directory': 'out', 'arguments': {'batch_size': 128}}
    out = tmp_path / 'out'
    out.mkdir()
    model = out / 'model.pt'
    model.write_bytes(b'evaluated weights')
    result = {'status': 'completed', 'diagnostic': False, 'experiment_sha256': fingerprint(job),
              'artifacts': [{'path': 'out/model.pt', 'sha256': sha(model)}]}
    (out / 'result.json').write_text(json.dumps(result))
    assert complete(job, tmp_path)
    model.write_bytes(b'other weights')
    with pytest.raises(ValueError, match='artifact changed'):
        complete(job, tmp_path)


def test_diagnostic_cannot_satisfy_full_experiment_even_with_matching_fingerprint(tmp_path):
    job = {'id': 'job', 'output_directory': 'out'}
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'result.json').write_text(json.dumps({'status': 'completed', 'diagnostic': True,
                                                'experiment_sha256': fingerprint(job), 'artifacts': []}))
    with pytest.raises(ValueError, match='Invalid full'):
        complete(job, tmp_path)
