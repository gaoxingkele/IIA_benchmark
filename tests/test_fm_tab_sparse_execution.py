import hashlib
import json
from pathlib import Path

import pytest

from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.tab_sparse_execution import config_hash, reuse_completed


def fixture(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'scores.npz').write_bytes(b'unchanged full saved score identity fixture')
    (model / 'model.pt').write_bytes(b'unchanged model fixture')
    job = {'output_directory': 'model'}
    result = {'id': 'fixture', 'status': 'completed', 'job': job,
              'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
              'scores_sha256': sha(model / 'scores.npz'), 'checkpoint_sha256': sha(model / 'model.pt')}
    path = model / 'result.json'
    write_json(path, result)
    parent = {'output_root': 'old', 'source_receipts': [], 'all_thresholds': 250, 'all_buffers': True}
    parent_path = tmp_path / 'parent.json'
    write_json(parent_path, parent)
    config = dict(parent, output_root='new', parent_config='parent.json', parent_config_sha256=sha(parent_path), engine='sparse')
    old = {'evaluation_identity': hashlib.sha256((sha(path) + config_hash(parent)).encode()).hexdigest(),
           'model_result_sha256': sha(path), 'scores_sha256': result['scores_sha256'],
           'VUS_entity_macro': {'VUS_ROC': {'mean_over_defined_entities': .8}},
           'strict_affiliation_per_entity': {'entity': {'1': {'affiliation_f': None}}},
           'evaluation_seconds': 900.123}
    old_path = tmp_path / 'old/jobs/fixture/evaluation.json'
    write_json(old_path, old)
    old_path.with_name('evaluation.sha256').write_text(sha(old_path))
    return path, config, old_path, old


def test_reuse_preserves_every_original_number_nan_semantic_time_and_artifact(tmp_path):
    path, config, old_path, old = fixture(tmp_path)
    before = old_path.read_bytes()
    record = reuse_completed(tmp_path, path, config)
    assert record['VUS_entity_macro'] == old['VUS_entity_macro']
    assert record['strict_affiliation_per_entity'] == old['strict_affiliation_per_entity']
    assert record['evaluation_seconds'] == old['evaluation_seconds']
    assert old_path.read_bytes() == before
    assert record['evaluation_identity'] != old['evaluation_identity']
    assert record['reused_original_evaluation']['sha256'] == sha(old_path)
    assert reuse_completed(tmp_path, path, config) == record


def test_modified_original_evaluation_and_scores_are_rejected(tmp_path):
    path, config, old_path, old = fixture(tmp_path)
    old_path.write_text(json.dumps(dict(old, evaluation_seconds=0)))
    with pytest.raises(ValueError, match='evaluation changed'):
        reuse_completed(tmp_path, path, config)
    old_path.write_text(json.dumps(old))
    old_path.with_name('evaluation.sha256').write_text(sha(old_path))
    (path.parent / 'scores.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        reuse_completed(tmp_path, path, config)


def test_all_original_job_rosters_and_metric_resolution_preserved():
    root = Path(__file__).resolve().parents[1]
    pairs = [('fm_tab_range_execution.v1.json', 'fm_tab_range_execution.sparse.v2.json'),
             ('fm_reflow_tab_range.v1.json', 'fm_reflow_tab_range.sparse.v2.json'),
             ('fm_industrial_tab_range.v1.json', 'fm_industrial_tab_range.sparse.v2.json'),
             ('fm_moment_pretrained_tab_range.v1.json', 'fm_moment_pretrained_tab_range.sparse.v2.json')]
    total = 0
    for old_name, new_name in pairs:
        read = lambda name: json.loads((root / 'configs/experiments' / name).read_text(encoding='utf-8'))
        old, new = read(old_name), read(new_name)
        assert old['queue_paths'] == new['queue_paths']
        for name in ('reference_metric_root', 'tab_commit', 'metric_semantics', 'strict_calibration', 'tab_calibration', 'undefined_policy'):
            assert old[name] == new[name]
        assert old['source_receipts'] == new['source_receipts'][:len(old['source_receipts'])]
        assert old['output_root'] != new['output_root']
        total += sum(len(json.loads((root / p).read_text())['jobs']) for p in old['queue_paths'])
    assert total == 3362
