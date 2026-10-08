import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_moment_pretrained import MomentPretrainedWindow, reconstruct_windows, load_pretrained
from scripts.flow_matching.run_moment_pretrained_job import segment_windows, segment_scores


class IdentityReconstruction(torch.nn.Module):
    patch_len = 8
    def forward(self, x_enc, input_mask, mask):
        assert torch.equal(input_mask, mask)
        assert torch.all(mask == 1)
        self.last_length = x_enc.shape[-1]
        return SimpleNamespace(reconstruction=x_enc)


@pytest.mark.parametrize('length,policy,expected', [(100,'minimal_patch',104), (512,'minimal_patch',512), (512,'TAB_always_extra_patch',520)])
def test_padding_and_all_observed_mask_keep_each_original_point(length, policy, expected):
    model = IdentityReconstruction()
    values = np.arange(2*length*3, dtype=np.float32).reshape(2,length,3)
    scores = reconstruct_windows(model, values, 1, 'cpu', policy)
    assert scores.shape == (2,length)
    assert np.array_equal(scores, np.zeros((2,length)))
    assert model.last_length == expected


def test_short_segment_retains_native_points_and_explicit_replication():
    values = np.arange(17*3, dtype=np.float32).reshape(17,3)
    windows, starts, padding = segment_windows(values, 512)
    assert windows.shape == (1,512,3) and padding == 495
    assert np.array_equal(windows[0,:17], values)
    assert np.array_equal(windows[0,17:], np.broadcast_to(values[-1],(495,3)))
    dummy = SimpleNamespace(score=lambda x: np.tile(np.arange(512),(len(x),1)))
    scores,audit=segment_scores(dummy,values,512)
    assert np.array_equal(scores,np.arange(17))
    assert audit['short_segment_replicated_points']==495


def test_pretrained_checksums_fail_before_import_or_partial_weight_loading(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'not pretrained')
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        load_pretrained(tmp_path,tmp_path,'bad','bad')


def test_zero_shot_cannot_silently_fine_tune_and_score_before_load():
    with pytest.raises(ValueError):
        MomentPretrainedWindow('source','checkpoint','sha','sha',epochs=1)
    model=MomentPretrainedWindow('source','checkpoint','sha','sha')
    with pytest.raises(RuntimeError):
        model.score(np.zeros((1,512,3)))


def test_all_actual_sizes_contexts_and_native_datasets_are_registered():
    root=Path(__file__).resolve().parents[1]
    queue=json.loads((root/'configs/experiments/fm_moment_pretrained_queue.v1.json').read_text())
    assert len(queue['jobs'])==72
    assert len({j['id'] for j in queue['jobs']})==72
    assert len({j['model_config'] for j in queue['jobs']})==6
    assert {j['window'] for j in queue['jobs']}=={100,512}
    assert len({j['dataset'] for j in queue['jobs']})==12
    for path in {j['model_config'] for j in queue['jobs']}:
        config=json.loads((root/path).read_text())
        assert config['parameters']['epochs']==0
        assert config['parameters']['checkpoint_sha256']
        assert config['parameters']['config_sha256']
