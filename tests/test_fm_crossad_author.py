import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_crossad_author_runtime import prepare_runtime, load_release
from scripts.flow_matching.run_crossad_author_job import native_scores


def test_source_preserved_and_only_two_cpu_placements_patched(tmp_path):
    source = tmp_path/'source'; target = tmp_path/'runtime'
    p = source/'models/CrossAD/Basic_CrossAD.py'; p.parent.mkdir(parents=True)
    text = '\n'.join(f'self.{name} = self.ms_utils.{name}(self.ms_p_lens).cuda()' for name in ['scale_ind_mask', 'next_scale_mask'])
    p.write_text(text)
    manifest = prepare_runtime(source, target)
    assert p.read_text() == text and not manifest['algorithm_changed']
    assert len(manifest['files'][0]['patches']) == 2
    assert '.cuda()' not in (target/p.relative_to(source)).read_text()
    assert prepare_runtime(source, target) == manifest


def test_changed_cpu_patch_anchor_is_rejected(tmp_path):
    p = tmp_path/'source/models/CrossAD/Basic_CrossAD.py'; p.parent.mkdir(parents=True); p.write_text('changed')
    with pytest.raises(ValueError, match='anchor'):
        prepare_runtime(tmp_path/'source', tmp_path/'runtime')


def test_release_all_algorithm_weights_loaded_and_only_thop_metadata_ignored(tmp_path):
    model = torch.nn.Linear(2, 1)
    state = dict(model.state_dict(), total_ops=torch.zeros(1)); path = tmp_path/'release.pt'
    torch.save(state, path)
    audit = load_release(model, path)
    assert audit['all_algorithm_weights_strictly_loaded'] and audit['ignored_thop_counter_keys'] == ['total_ops']
    state.pop('weight'); torch.save(state, path)
    with pytest.raises(ValueError, match='missing'):
        load_release(model, path)


def test_unknown_checkpoint_weights_are_rejected(tmp_path):
    model = torch.nn.Linear(2, 1); state = dict(model.state_dict(), random_head=torch.zeros(1)); path = tmp_path/'release.pt'
    torch.save(state, path)
    with pytest.raises(ValueError, match='unexpected'):
        load_release(model, path)


@pytest.mark.parametrize('length', [8, 9, 17])
def test_native_tail_controls_keep_original_nonoverlap_points(length):
    values = np.arange(length*3, dtype=np.float32).reshape(length, 3)
    model = SimpleNamespace(infer=lambda batch, *args: (batch, None))
    assert np.array_equal(native_scores(model, values, 8, 2), values.mean(axis=1))


def test_short_segment_not_silently_replaced():
    with pytest.raises(ValueError, match='shorter'):
        native_scores(None, np.zeros((2, 3)), 8, 2)


def test_all_author_datasets_and_complete_training_budgets_registered():
    root = Path(__file__).resolve().parents[1]
    queue = json.loads((root/'configs/experiments/fm_crossad_author_queue.v1.json').read_text())
    assert len(queue['jobs']) == 49 and len({j['id'] for j in queue['jobs']}) == 49
    assert sum(j['mode'] == 'release_checkpoint' for j in queue['jobs']) == 7
    assert sum(j['mode'] == 'fresh_train' for j in queue['jobs']) == 42
    assert len({j['dataset'] for j in queue['jobs']}) == 7
    for job in queue['jobs']:
        base = root/queue['settings']['source']/'configs'/job['dataset']
        assert job['train_parameters'] == json.loads((base/'train_configs.json').read_text())
        assert job['model_parameters'] == json.loads((base/'model_configs_0.json').read_text())
        assert job['train_parameters']['train_epochs'] == 20
