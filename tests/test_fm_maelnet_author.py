import json

import pytest
import torch

from scripts.flow_matching.register_maelnet_author import script_arguments
from scripts.flow_matching import run_maelnet_author_queue as runner


def test_shell_recipe_is_parsed_without_execution_and_capacity_is_retained():
    text = 'export CUDA_VISIBLE_DEVICES=0\npython -u run_anomaly.py \\\n --model MaelNetS2 \\\n --d_model 512 --p_hidden_dims 128 128 --train_epochs 10\n'
    original = script_arguments(text)
    changed = runner.effective_arguments({'author_arguments': original}, {'raw_directory': 'data/native', 'seed': 1103},
                                        {'num_workers': 0, 'chunk_size': 0}, runner.ROOT / 'tmp' / 'workspace')
    assert runner.argument(changed, 'd_model') == '512'
    assert runner.argument(changed, 'train_epochs') == '10'
    assert changed[changed.index('--p_hidden_dims') + 1:changed.index('--p_hidden_dims') + 3] == ['128', '128']
    assert '--root_path' not in original
    assert runner.argument(changed, 'root_path') == str(runner.ROOT / 'data/native')
    with pytest.raises(ValueError):
        script_arguments('python -u run_anomaly.py --model X\npython -u run_anomaly.py --model Y')


def training_case(tmp_path, monkeypatch, epochs=3, steps=5, early=False, nonfinite=False):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    workspace = tmp_path / 'workspace'
    (workspace / 'checkpoints' / 'setting').mkdir(parents=True)
    (workspace / 'results').mkdir()
    torch.save({'weight': torch.tensor([float('nan') if nonfinite else 1.])}, workspace / 'checkpoints/setting/checkpoint_AnomalyTransformer.pth')
    csv = workspace / 'results/training.csv'
    csv.write_text('Epoch,Cost Time,Steps,Train Loss,Vali Loss,Test Loss\n' + ''.join(f'{i},1,{steps},0.4,0.5,0.6\n' for i in range(1, epochs + 1)))
    stdout = tmp_path / 'stdout.log'
    stdout.write_text('Early stopping' if early else '')
    return workspace, set(), ['--model', 'AnomalyTransformer', '--train_epochs', '3', '--batch_size', '32'], {'window_counts': {'train': 160}}, stdout


def test_truncated_epoch_budget_requires_author_early_stopping(tmp_path, monkeypatch):
    values = training_case(tmp_path, monkeypatch, epochs=2)
    with pytest.raises(ValueError, match='epoch budget'):
        runner.stage_artifacts(*values)
    values[-1].write_text('Early stopping')
    artifacts, details = runner.stage_artifacts(*values)
    assert details['epochs_completed'] == 2 and details['author_early_stopping']
    assert len(artifacts) == 2


def test_reduced_data_steps_cannot_be_promoted_to_full_data_result(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='cover all'):
        runner.stage_artifacts(*training_case(tmp_path, monkeypatch, steps=4))


def test_nonfinite_checkpoint_is_rejected_even_with_full_epochs(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='nonfinite learner'):
        runner.stage_artifacts(*training_case(tmp_path, monkeypatch, nonfinite=True))


def test_receipt_tampering_and_partial_stage_cannot_count_as_complete(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'weights')
    receipt = {'status': 'completed', 'artifacts': [{'path': 'model.pt', 'sha256': runner.sha(path)}]}
    runner.verify_artifacts(receipt, tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_artifacts(receipt, tmp_path)
    with pytest.raises(ValueError, match='complete stage'):
        runner.verify_artifacts({'status': 'running', 'artifacts': receipt['artifacts']}, tmp_path)


def test_author_pipeline_snapshot_keeps_pending_and_label_dependent_track_separate(tmp_path):
    from scripts.flow_matching.summarize_tsad_execution import author_pipeline_snapshot
    queue = {'settings': {'output_root': 'author'}, 'boundary': 'test-label diagnostic',
             'jobs': [{'id': 'run', 'dataset': 'PSM', 'seed': 1103, 'author_recipe': 'e3d1',
                       'output_directory': 'author/jobs/run', 'stages': [{'id': 'train'}, {'id': 'rl'}]}]}
    (tmp_path / 'queue.json').write_text(json.dumps(queue))
    snapshot = author_pipeline_snapshot(tmp_path, {'author_execution_queues': {'maelnet': 'queue.json'}})[0]
    assert snapshot['registered_stages'] == 2
    assert snapshot['counts'] == {'pending': 1}
    assert snapshot['strict_TAB_result'] is False
    assert snapshot['process_observation']['verified_live'] is False
    target = tmp_path / 'author/jobs/run'
    target.mkdir(parents=True)
    (target / 'result.json').write_text(json.dumps({'experiment_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='another job'):
        author_pipeline_snapshot(tmp_path, {'author_execution_queues': {'maelnet': 'queue.json'}})
